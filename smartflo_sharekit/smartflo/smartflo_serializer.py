"""SmartFlo (Tata Tele Business Services) Media Streams serializer for Pipecat.

Protocol differences vs Exotel / Twilio:
- Audio encoding is µ-law / G.711 PCMU  (Exotel uses raw PCM — different codecs!)
- Outgoing audio payloads MUST be multiples of 160 bytes (enforced by _audio_buffer)
- Incoming `mark` events must be acknowledged with a `mark` response
- `clear` events from SmartFlo interrupt outgoing audio (mapped to InterruptionFrame)
- `connected`, `start`, and `stop` events are silently ignored

Audio format:
- Encoding:    mulaw / G.711 µ-law (PCMU)
- Sample rate: 8000 Hz
- Channels:    mono
- Payload:     base64-encoded JSON field

Critical mistake to avoid:
    Do NOT send raw PCM to SmartFlo. SmartFlo expects µ-law (PCMU).
    Sending PCM produces loud static noise on the caller's end.
    This serializer handles PCM ↔ µ-law conversion automatically.
"""

import base64
import json
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.utils import create_stream_resampler, pcm_to_ulaw, ulaw_to_pcm
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

# SmartFlo requires all outgoing audio payloads to be multiples of this size.
_CHUNK_BYTES = 160


class SmartFloFrameSerializer(FrameSerializer):
    """Serializer for SmartFlo (Tata Tele) bi-directional audio streaming WebSocket protocol.

    Drop-in replacement for ExotelFrameSerializer — swap the serializer,
    everything else (transport, pipeline, LLM, TTS, STT) stays the same.

    Key difference from Exotel:
        SmartFlo uses µ-law (PCMU/G.711) encoding.
        Exotel uses raw PCM.
        Sending PCM to SmartFlo = static noise on the call.
    """

    class InputParams(BaseModel):
        smartflo_sample_rate: int = 8000
        sample_rate: Optional[int] = None

    def __init__(
        self,
        stream_sid: str,
        call_sid: Optional[str] = None,
        params: Optional[InputParams] = None,
    ):
        self._stream_sid = stream_sid
        self._call_sid = call_sid
        self._params = params or SmartFloFrameSerializer.InputParams()

        self._smartflo_sample_rate = self._params.smartflo_sample_rate
        self._sample_rate = 0

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()

        # Buffer outgoing audio — flush only when buffer is a multiple of _CHUNK_BYTES
        self._out_buffer = b""

    async def setup(self, frame: StartFrame):
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Convert a Pipecat frame → SmartFlo WebSocket JSON message.

        Outgoing audio flow:
            PCM (pipeline) → resample to 8kHz → pcm_to_ulaw → base64 → JSON
        """
        if isinstance(frame, InterruptionFrame):
            # Bot was interrupted — clear our buffer and tell SmartFlo to clear its queue
            self._out_buffer = b""
            return json.dumps({"event": "clear", "streamSid": self._stream_sid})

        if isinstance(frame, AudioRawFrame):
            # Convert PCM → µ-law at SmartFlo's sample rate
            ulaw_bytes = await pcm_to_ulaw(
                frame.audio, frame.sample_rate, self._smartflo_sample_rate, self._output_resampler
            )
            if not ulaw_bytes:
                return None

            self._out_buffer += ulaw_bytes
            if not self._out_buffer:
                return None

            # Pad to next multiple of _CHUNK_BYTES using µ-law silence (0xFF)
            remainder = len(self._out_buffer) % _CHUNK_BYTES
            if remainder:
                self._out_buffer += b"\xff" * (_CHUNK_BYTES - remainder)

            payload = base64.b64encode(self._out_buffer).decode("ascii")
            self._out_buffer = b""

            return json.dumps({
                "event": "media",
                "streamSid": self._stream_sid,
                "media": {"payload": payload},
            })

        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            return json.dumps(frame.message)

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Convert SmartFlo WebSocket JSON message → Pipecat frame.

        Incoming audio flow:
            base64 → µ-law → ulaw_to_pcm → resample to pipeline rate → InputAudioRawFrame
        """
        message = json.loads(data)
        event = message.get("event", "")

        if event == "media":
            payload = base64.b64decode(message["media"]["payload"])
            # Convert µ-law → PCM at the pipeline's sample rate
            pcm_bytes = await ulaw_to_pcm(
                payload, self._smartflo_sample_rate, self._sample_rate, self._input_resampler
            )
            if not pcm_bytes:
                return None
            return InputAudioRawFrame(
                audio=pcm_bytes,
                num_channels=1,
                sample_rate=self._sample_rate,
            )

        if event == "dtmf":
            digit = message.get("dtmf", {}).get("digit")
            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError:
                logger.info(f"SmartFlo: invalid DTMF digit: {digit}")
                return None

        if event == "mark":
            # SmartFlo requires every mark it sends to be echoed back
            mark_name = message.get("mark", {}).get("name", "")
            logger.debug(f"SmartFlo: mark received: {mark_name}")
            ack = {"event": "mark", "streamSid": self._stream_sid, "mark": {"name": mark_name}}
            return OutputTransportMessageFrame(message=ack)

        if event in ("connected", "start", "stop"):
            return None  # Expected lifecycle events — no frame needed

        logger.debug(f"SmartFlo: unhandled event type: {event}")
        return None
