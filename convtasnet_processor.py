
#
# Conv-TasNet Speech Separation Processor for Pipecat
#
# Isolates the dominant (primary) speaker from overlapping voices
# in real-time mic input before audio reaches Gemini Live.
#
# Supports two modes:
#   - Browser (WebRTC): Separates 2 speakers, uses profile tracking
#   - Telephony (SmartFlo): Denoises 1 speaker, uses energy-based selection
#
# Author: Sashi Vardhan
#

import asyncio
import functools
from typing import Optional

import numpy as np
import torch
import torchaudio
from torchaudio.pipelines import CONVTASNET_BASE_LIBRI2MIX
from loguru import logger

from pipecat.frames.frames import Frame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


class ConvTasNetProcessor(FrameProcessor):
    """Pipecat FrameProcessor that separates overlapping speakers in real-time.

    Sits between transport.input() and the LLM aggregator in the pipeline.
    Intercepts InputAudioRawFrame, runs Conv-TasNet to isolate the dominant
    speaker, and pushes cleaned audio downstream.

    Two operating modes:
        Browser:   2-speaker separation with spectral profile tracking
        Telephony: 1-speaker denoising with energy-based selection

    Architecture:
        Audio Source → Transport → [THIS PROCESSOR] → Gemini Live

    The model runs at 8 kHz internally. Audio is resampled from/to the
    transport's native sample rate (default 16 kHz) automatically.
    """

    MODEL_SAMPLE_RATE = 8000  # Conv-TasNet native rate

    def __init__(
        self,
        *,
        chunk_duration: float = 0.4,
        vad_threshold: float = 0.001,
        telephony_mode: bool = False,
        telephony_strategy: str = "energy",
        profile_lock_min_energy: Optional[float] = None,
        profile_min_similarity: Optional[float] = None,
        energy_ratio_db_min: Optional[float] = None,
        speech_hold_chunks: int = 0,
        name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Conv-TasNet processor.

        Args:
            chunk_duration: Processing window in seconds. Smaller = lower
                latency but potentially worse separation quality.
                Recommended range: 0.25–0.5. Default 0.4 (400ms).
            vad_threshold: Raw energy gate threshold. Audio chunks below
                this energy level are treated as silence and skip the model.
                Auto-adjusted for telephony mode.
            telephony_mode: If True, optimize for telephony audio:
                - Lower VAD threshold (µ-law has lower energy after decoding)
                - Energy-based source selection (1 speaker + noise scenario)
                - Input gain normalization for compressed audio
                - Preserves original signal level better
            telephony_strategy: Telephony source selection strategy.
                "energy" uses the louder source (default).
                "profile" uses browser-style speaker profile tracking.
            profile_lock_min_energy: Minimum mix energy to build/track
                the speaker profile. Below this, output is suppressed.
            profile_min_similarity: Minimum cosine similarity required
                to accept a tracked speaker; lower values are suppressed.
            energy_ratio_db_min: Minimum dB difference between the chosen
                source and the other source. Below this, output is suppressed.
            speech_hold_chunks: Keep passing audio for this many chunks
                after speech drops below the VAD threshold.
            name: Optional processor name for logging.
        """
        super().__init__(name=name or "ConvTasNetProcessor", **kwargs)

        self._telephony_mode = telephony_mode
        telephony_strategy = (telephony_strategy or "energy").strip().lower()
        if telephony_strategy not in {"energy", "profile"}:
            logger.warning(
                "ConvTasNetProcessor: Unknown telephony_strategy='{}'. Using 'energy'.",
                telephony_strategy,
            )
            telephony_strategy = "energy"
        self._telephony_strategy = telephony_strategy
        self._chunk_duration = chunk_duration
        self._chunk_samples = int(self.MODEL_SAMPLE_RATE * chunk_duration)

        # --- Telephony-specific adjustments ---
        # µ-law audio after decoding to float32 has ~10-20dB lower energy
        # than native 16kHz PCM. We need a much lower VAD threshold to
        # avoid rejecting valid speech as silence.
        if telephony_mode:
            self._vad_threshold = vad_threshold * 0.1  # 10× more sensitive
        else:
            self._vad_threshold = vad_threshold

        # Audio buffer (accumulated at model sample rate, 8 kHz)
        self._buffer = np.zeros(0, dtype=np.float32)

        # Thread pool executor for blocking model inference
        self._executor = None

        # Track the input sample rate (set from first frame)
        self._input_sr: Optional[int] = None

        # --- Speaker tracking state (browser mode only) ---
        # In telephony mode, we use simple energy-based selection instead
        # because there's typically 1 speaker + ambient noise, not 2 speakers.
        self._speaker_profile: Optional[np.ndarray] = None
        self._locked_source_idx: Optional[int] = None
        self._profile_chunks_seen: int = 0
        self._profile_warmup: int = 3
        self._profile_lock_min_energy = profile_lock_min_energy
        self._profile_min_similarity = profile_min_similarity
        self._energy_ratio_db_min = energy_ratio_db_min
        self._speech_hold_chunks = max(int(speech_hold_chunks), 0)
        self._speech_hold_remaining = 0

        # --- Telephony audio stats ---
        # Track running input gain to preserve natural volume levels
        self._input_gain_ema: float = 0.0  # Exponential moving average of input RMS
        self._gain_ema_alpha: float = 0.1  # EMA smoothing factor

        # --- Precomputed resampler caches (avoid recreating per chunk) ---
        self._resamplers: dict[tuple[int, int], torchaudio.transforms.Resample] = {}

        # Load Conv-TasNet model
        logger.info("ConvTasNetProcessor: Loading Conv-TasNet model...")
        bundle = CONVTASNET_BASE_LIBRI2MIX
        self._model = bundle.get_model()
        self._model.eval()

        # Force CPU — Conv-TasNet is small enough for CPU inference
        self._device = torch.device("cpu")
        self._model = self._model.to(self._device)

        # Optimize: limit PyTorch threads to avoid contention with async loop
        torch.set_num_threads(2)

        if telephony_mode:
            mode_str = "TELEPHONY (profile tracking)" if self._telephony_strategy == "profile" else "TELEPHONY (denoising)"
        else:
            mode_str = "BROWSER (2-speaker separation)"
        logger.info(
            "ConvTasNetProcessor: Model loaded on {}. "
            "mode={}, chunk_duration={:.2f}s ({} samples), vad_threshold={:.6f}",
            self._device,
            mode_str,
            chunk_duration,
            self._chunk_samples,
            self._vad_threshold,
        )

    def _get_resampler(self, from_sr: int, to_sr: int) -> torchaudio.transforms.Resample:
        """Get or create a cached resampler. Avoids repeated kernel computation."""
        key = (from_sr, to_sr)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(
                orig_freq=from_sr,
                new_freq=to_sr,
                resampling_method="sinc_interp_kaiser",
                lowpass_filter_width=16,
                dtype=torch.float32,
            )
        return self._resamplers[key]

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame through the Conv-TasNet separation pipeline.

        Only InputAudioRawFrame frames are processed; all other frame types
        are passed through untouched.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            await self._process_audio(frame, direction)
        else:
            # Pass through all non-audio frames (system frames, control, etc.)
            await self.push_frame(frame, direction)

    async def _process_audio(self, frame: InputAudioRawFrame, direction: FrameDirection):
        """Buffer, separate, and emit cleaned audio frames."""
        # Detect input sample rate from the first frame
        if self._input_sr is None:
            self._input_sr = frame.sample_rate
            logger.info(
                "ConvTasNetProcessor: Detected input sample rate: {} Hz (telephony={})",
                self._input_sr,
                self._telephony_mode,
            )

        # Decode PCM int16 bytes → float32 [-1, 1]
        pcm_int16 = np.frombuffer(frame.audio, dtype=np.int16)
        pcm_float = pcm_int16.astype(np.float32) / 32768.0

        # Resample to model rate (8 kHz) if needed
        if self._input_sr != self.MODEL_SAMPLE_RATE:
            pcm_float = self._resample(pcm_float, self._input_sr, self.MODEL_SAMPLE_RATE)

        # Append to buffer
        self._buffer = np.concatenate([self._buffer, pcm_float])

        # Process all complete chunks in the buffer
        while len(self._buffer) >= self._chunk_samples:
            chunk = self._buffer[: self._chunk_samples]
            self._buffer = self._buffer[self._chunk_samples :]

            # Run model inference in thread pool to avoid blocking event loop
            loop = asyncio.get_running_loop()
            cleaned = await loop.run_in_executor(
                self._executor,
                functools.partial(self._separate_chunk, chunk),
            )

            # Resample back to input rate if needed
            if self._input_sr != self.MODEL_SAMPLE_RATE:
                cleaned = self._resample(cleaned, self.MODEL_SAMPLE_RATE, self._input_sr)

            # Convert float32 → PCM int16 bytes
            out_int16 = (cleaned * 32767).astype(np.int16)
            out_bytes = out_int16.tobytes()

            # Create and push a new cleaned audio frame
            cleaned_frame = InputAudioRawFrame(
                audio=out_bytes,
                sample_rate=self._input_sr,
                num_channels=frame.num_channels,
            )
            await self.push_frame(cleaned_frame, direction)

    # ------------------------------------------------------------------
    #  Source selection strategies
    # ------------------------------------------------------------------

    def _select_speaker_telephony(self, src0: np.ndarray, src1: np.ndarray) -> int:
        """Source selection for TELEPHONY mode (1 speaker + noise).

        On a phone call there is typically ONE primary speaker (the caller)
        and ambient noise (field machinery, wind, other people in background).
        Conv-TasNet separates these into two sources.

        Strategy: Pick the source with HIGHER energy — in telephony, the
        caller's voice is always the dominant signal on the phone channel.
        Background noise is attenuated by the phone's own DSP before reaching us.

        This is simpler and more reliable than profile tracking for single-speaker
        scenarios because there's no risk of locking onto the wrong source.
        """
        energy_0 = float(np.mean(src0 ** 2))
        energy_1 = float(np.mean(src1 ** 2))
        return 0 if energy_0 >= energy_1 else 1

    def _select_speaker_browser(self, src0: np.ndarray, src1: np.ndarray) -> Optional[int]:
        """Source selection for BROWSER mode (2 speakers / cross-talk).

        Uses spectral profile tracking to lock onto the primary speaker and
        avoid flipping to a louder background source (e.g. YouTube, TV).

        Phase 1 (warmup): Use correlation with mix to identify caller
        Phase 2 (tracking): Use spectral cosine similarity to maintain lock
        """
        mix = src0 + src1
        if self._profile_lock_min_energy is not None:
            mix_energy = float(np.mean(mix ** 2))
            if mix_energy < self._profile_lock_min_energy:
                return None

        if self._profile_chunks_seen < self._profile_warmup:
            # Warmup: pick source most correlated with the mix
            corr0 = float(np.abs(np.corrcoef(src0, mix)[0, 1]))
            corr1 = float(np.abs(np.corrcoef(src1, mix)[0, 1]))
            chosen = 0 if corr0 >= corr1 else 1
            chosen_src = src0 if chosen == 0 else src1

            # Build spectral profile
            profile = np.abs(np.fft.rfft(chosen_src))
            if self._speaker_profile is None:
                self._speaker_profile = profile
            else:
                alpha = 1.0 / (self._profile_chunks_seen + 1)
                self._speaker_profile = (1 - alpha) * self._speaker_profile + alpha * profile

            self._profile_chunks_seen += 1
            self._locked_source_idx = chosen

            if self._profile_chunks_seen >= self._profile_warmup:
                logger.info(
                    "ConvTasNetProcessor: Speaker profile locked (source {})",
                    chosen,
                )

            return chosen
        else:
            # Tracking: compare to locked profile via cosine similarity
            spec0 = np.abs(np.fft.rfft(src0))
            spec1 = np.abs(np.fft.rfft(src1))
            profile = self._speaker_profile
            if profile is None:
                return None

            sim0 = float(np.dot(spec0, profile) / (np.linalg.norm(spec0) * np.linalg.norm(profile) + 1e-8))
            sim1 = float(np.dot(spec1, profile) / (np.linalg.norm(spec1) * np.linalg.norm(profile) + 1e-8))

            max_sim = max(sim0, sim1)
            if self._profile_min_similarity is not None and max_sim < self._profile_min_similarity:
                return None

            chosen = 0 if sim0 >= sim1 else 1

            # Slowly update profile
            chosen_spec = spec0 if chosen == 0 else spec1
            self._speaker_profile = 0.95 * self._speaker_profile + 0.05 * chosen_spec

            return chosen

    # ------------------------------------------------------------------
    #  Core separation
    # ------------------------------------------------------------------

    def _separate_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Run Conv-TasNet separation on a single audio chunk (blocking).

        This runs in a thread executor. It:
        1. Checks energy (VAD gate) — skips model for silence
        2. Normalizes the chunk (with gain tracking for telephony)
        3. Runs Conv-TasNet to get 2 separated sources
        4. Selects the speech source (strategy depends on mode)
        5. Restores original gain level and returns cleaned waveform

        Args:
            chunk: Float32 audio chunk, shape (N,), at 8 kHz.

        Returns:
            Cleaned float32 audio chunk, shape (N,).
        """
        # VAD gate: check raw energy BEFORE normalization
        raw_energy = float(np.mean(chunk ** 2))
        speech_active = raw_energy >= self._vad_threshold
        if speech_active:
            if self._speech_hold_chunks > 0:
                self._speech_hold_remaining = self._speech_hold_chunks
        elif self._speech_hold_remaining > 0:
            self._speech_hold_remaining -= 1
            speech_active = True
        else:
            return np.zeros(len(chunk), dtype=np.float32)

        # Track input RMS for gain restoration (telephony mode)
        input_rms = float(np.sqrt(raw_energy))
        if self._telephony_mode:
            if self._input_gain_ema == 0.0:
                self._input_gain_ema = input_rms
            else:
                self._input_gain_ema = (
                    (1 - self._gain_ema_alpha) * self._input_gain_ema
                    + self._gain_ema_alpha * input_rms
                )

        # Normalize input for model
        peak = np.max(np.abs(chunk)) + 1e-8
        chunk_norm = chunk / peak

        # Convert to tensor [1, 1, T] for Conv-TasNet
        waveform = torch.tensor(chunk_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        waveform = waveform.to(self._device)

        # Run separation
        with torch.no_grad():
            sources = self._model(waveform)[0]  # shape: [2, T]

        src0 = sources[0].cpu().numpy().flatten()
        src1 = sources[1].cpu().numpy().flatten()

        # Select speech source based on operating mode
        if self._telephony_mode:
            if self._telephony_strategy == "profile":
                chosen_idx = self._select_speaker_browser(src0, src1)
            else:
                chosen_idx = self._select_speaker_telephony(src0, src1)
        else:
            chosen_idx = self._select_speaker_browser(src0, src1)

        if chosen_idx is None:
            return np.zeros(len(chunk), dtype=np.float32)

        if (
            self._energy_ratio_db_min is not None
            and self._energy_ratio_db_min > 0
            and raw_energy >= self._vad_threshold
        ):
            energy_0 = float(np.mean(src0 ** 2))
            energy_1 = float(np.mean(src1 ** 2))
            chosen_energy = energy_0 if chosen_idx == 0 else energy_1
            other_energy = energy_1 if chosen_idx == 0 else energy_0
            if chosen_energy <= 0:
                return np.zeros(len(chunk), dtype=np.float32)
            ratio_db = 10.0 * np.log10((chosen_energy + 1e-12) / (other_energy + 1e-12))
            if ratio_db < self._energy_ratio_db_min:
                return np.zeros(len(chunk), dtype=np.float32)

        main_np = src0 if chosen_idx == 0 else src1

        # --- Output gain management ---
        # The model's output is normalized differently from the input.
        # For telephony, we restore the original signal level to avoid
        # volume jumps that confuse downstream components.
        out_peak = np.max(np.abs(main_np)) + 1e-8

        if self._telephony_mode:
            # Restore to input RMS level — prevents volume distortion
            out_rms = float(np.sqrt(np.mean(main_np ** 2))) + 1e-8
            target_rms = self._input_gain_ema
            gain = target_rms / out_rms
            # Clamp gain to prevent extreme amplification
            gain = min(gain, 3.0)
            main_np = main_np * gain
            # Final clip to prevent clipping
            main_np = np.clip(main_np, -0.95, 0.95)
        else:
            # Browser mode: simple peak normalization (original behavior)
            main_np = main_np / out_peak

        return main_np

    def _resample(self, audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """Resample audio between sample rates using cached high-quality resamplers.

        Uses kaiser-windowed sinc interpolation for better quality than
        the default linear interpolation, especially important for
        telephony where we're resampling compressed audio.

        Args:
            audio: Float32 audio array, shape (N,).
            from_sr: Source sample rate.
            to_sr: Target sample rate.

        Returns:
            Resampled float32 audio array.
        """
        if from_sr == to_sr:
            return audio
        resampler = self._get_resampler(from_sr, to_sr)
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        resampled = resampler(waveform)
        return resampled.squeeze(0).numpy()
