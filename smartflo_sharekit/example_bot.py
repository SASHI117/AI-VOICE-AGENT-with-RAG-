"""SmartFlo bot customized for apsagent / rnf_agnet Gemini + RAG stack.

This keeps the SmartFlo serializer and transport, but reuses your existing
agent logic (prompt, function calling, and RAG behavior).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, Frame, LLMRunFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams, FastAPIWebsocketTransport

from smartflo.smartflo_serializer import SmartFloFrameSerializer


# Ensure imports like `import apsagent` and `import rnf_gemini.rnf_agnet` work
# when running from smartflo_sharekit directory.
SERVER_ROOT = Path(__file__).resolve().parent.parent
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

load_dotenv(SERVER_ROOT / ".env", override=False)


TRANSFER_MARKER = "[TRANSFER_TO_AGENT]"
ESCALATION_NUMBER = os.getenv("ESCALATION_NUMBER", "+919154708539")
SMARTFLO_API_BASE_URL = os.getenv(
    "SMARTFLO_API_BASE_URL", "https://api-smartflo.tatateleservices.com"
).rstrip("/")
SMARTFLO_API_TOKEN = os.getenv("SMARTFLO_API_TOKEN", "").strip()


def _sanitize_intercom(raw_value: str) -> str:
    return str(raw_value or "").strip()


def _post_smartflo_call_options(payload: dict) -> dict:
    url = f"{SMARTFLO_API_BASE_URL}/v1/call/options"
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
    }
    if SMARTFLO_API_TOKEN:
        headers["Authorization"] = SMARTFLO_API_TOKEN

    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=6) as response:
        raw = response.read().decode("utf-8", errors="ignore")
        return json.loads(raw) if raw else {}


async def _transfer_smartflo_call(call_id: str, intercom: str) -> bool:
    if not call_id:
        logger.error("SmartFlo transfer skipped: missing call_id")
        return False

    intercom_value = _sanitize_intercom(intercom)
    if not intercom_value:
        logger.error("SmartFlo transfer skipped: missing intercom")
        return False

    if not SMARTFLO_API_TOKEN:
        logger.warning("SMARTFLO_API_TOKEN is not set; transfer may fail")

    payload = {
        "type": 4,
        "call_id": call_id,
        "intercom": intercom_value,
    }

    try:
        result = await asyncio.to_thread(_post_smartflo_call_options, payload)
        logger.info("SmartFlo transfer response: {}", result)
        return bool(result.get("success", False)) if isinstance(result, dict) else False
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
        logger.error("SmartFlo transfer failed: status={} body='{}'", exc.code, body)
        return False
    except Exception:
        logger.exception("SmartFlo transfer failed")
        return False


class EscalationDetector(FrameProcessor):
    """Detect transfer marker in streaming LLM text and trigger SmartFlo transfer."""

    def __init__(self, task: PipelineTask | None, call_id: str = ""):
        super().__init__()
        self._task = task
        self._call_id = call_id
        self._text_buffer = ""
        self._detection_phase = True
        self._transfer_started = False
        self._disconnect_scheduled = False
        self._last_was_text = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        is_text = isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM

        if not is_text:
            self._last_was_text = False
            await self.push_frame(frame, direction)
            return

        text = frame.text or ""
        if not text:
            await self.push_frame(frame, direction)
            return

        if not self._last_was_text and not self._detection_phase:
            self._detection_phase = True
            self._text_buffer = ""

        self._last_was_text = True

        if not self._detection_phase:
            await self.push_frame(frame, direction)
            return

        self._text_buffer += text

        marker_index = self._text_buffer.find(TRANSFER_MARKER)
        if marker_index != -1 and not self._text_buffer[:marker_index].strip():
            self._detection_phase = False

            logger.info(
                "Escalation detected; transferring call_id={} -> {}",
                self._call_id,
                ESCALATION_NUMBER,
            )

            if not self._transfer_started:
                self._transfer_started = True

                async def _transfer_and_maybe_end():
                    ok = await _transfer_smartflo_call(self._call_id, ESCALATION_NUMBER)
                    if not ok:
                        logger.warning("SmartFlo transfer was not accepted; keeping call connected")
                        return

                    if self._task is None:
                        return

                    if self._disconnect_scheduled:
                        return

                    self._disconnect_scheduled = True

                    await asyncio.sleep(2)
                    logger.info("Closing SmartFlo session after successful escalation")
                    await self._task.queue_frames([EndFrame()])

                asyncio.create_task(_transfer_and_maybe_end())

            clean = self._text_buffer.replace(TRANSFER_MARKER, "").strip()
            self._text_buffer = ""
            if clean:
                frame.text = clean
                await self.push_frame(frame, direction)

            return

        if len(self._text_buffer) >= len(TRANSFER_MARKER):
            self._detection_phase = False
            frame.text = self._text_buffer
            self._text_buffer = ""
            await self.push_frame(frame, direction)
            return

        stripped = self._text_buffer.lstrip()
        if stripped and not TRANSFER_MARKER.startswith(stripped):
            self._detection_phase = False
            frame.text = self._text_buffer
            self._text_buffer = ""
            await self.push_frame(frame, direction)
            return

        return


def _load_agent_module():
    profile = os.getenv("SMARTFLO_AGENT_PROFILE", "rnf_agnet").strip().lower()
    if profile == "apsagent":
        import apsagent as agent_mod

        return "apsagent", agent_mod

    import rnf_gemini.rnf_agnet as agent_mod

    return "rnf_agnet", agent_mod


def _limits_for_profile(profile: str) -> tuple[int, int, int, int]:
    if profile == "apsagent":
        return (
            int(os.getenv("RAG_MAX_CONTEXT_CHARS", "5000")),
            int(os.getenv("RAG_MAX_CONTEXT_CHARS_DETAIL", "7500")),
            int(os.getenv("RAG_MAX_CHUNKS", "4")),
            int(os.getenv("RAG_MAX_CHUNKS_DETAIL", "6")),
        )

    return (
        int(os.getenv("RNF_RAG_MAX_CONTEXT_CHARS", os.getenv("RAG_MAX_CONTEXT_CHARS", "5000"))),
        int(
            os.getenv(
                "RNF_RAG_MAX_CONTEXT_CHARS_DETAIL",
                os.getenv("RAG_MAX_CONTEXT_CHARS_DETAIL", "8000"),
            )
        ),
        int(os.getenv("RNF_RAG_MAX_CHUNKS", os.getenv("RAG_MAX_CHUNKS", "4"))),
        int(os.getenv("RNF_RAG_MAX_CHUNKS_DETAIL", os.getenv("RAG_MAX_CHUNKS_DETAIL", "6"))),
    )


def _configure_smartflo_rag_mode(profile: str) -> None:
    """Prefer FastRAG for telephony sessions to avoid call-setup timeouts.

    SmartFlo calls can be disconnected quickly if startup blocks on remote embedding
    generation. For this bridge, default to FastRAG unless explicitly disabled.
    """
    force_fast_rag = os.getenv("SMARTFLO_FORCE_FAST_RAG", "1").strip().lower()
    if force_fast_rag not in {"1", "true", "yes", "on"}:
        return

    threshold = os.getenv("SMARTFLO_FAST_RAG_CHUNK_THRESHOLD", "1000000000")
    if profile == "apsagent":
        os.environ["FAST_RAG_CHUNK_THRESHOLD"] = threshold
    else:
        os.environ["RNF_FAST_RAG_CHUNK_THRESHOLD"] = threshold

    logger.info("SmartFlo FastRAG mode enabled for profile='{}'", profile)


def _configure_smartflo_live_ws_timeout(profile: str) -> None:
    """Set a safer default handshake timeout for Gemini Live over WebSocket."""
    os.environ.setdefault("GEMINI_WS_OPEN_TIMEOUT_SECS", "35")
    if profile != "apsagent":
        os.environ.setdefault("RNF_GEMINI_WS_OPEN_TIMEOUT_SECS", os.environ["GEMINI_WS_OPEN_TIMEOUT_SECS"])

    logger.info(
        "SmartFlo Gemini WS open timeout set to {}s",
        os.getenv("GEMINI_WS_OPEN_TIMEOUT_SECS"),
    )


async def parse_smartflo_start(websocket) -> dict:
    """Read initial SmartFlo lifecycle events and extract stream metadata.

    SmartFlo deployments can vary slightly in payload shape; some send `streamSid`
    under `start`, while others include it at the top level. We accept either to
    avoid stalling the call setup.
    """
    # Some providers include metadata as URL query params instead of an initial
    # JSON frame. Prefer this immediately when present.
    query_stream_id = str(
        websocket.query_params.get("streamSid")
        or websocket.query_params.get("stream_id")
        or ""
    ).strip()
    query_call_id = str(
        websocket.query_params.get("callSid")
        or websocket.query_params.get("call_id")
        or ""
    ).strip()
    if query_stream_id:
        logger.info(
            "SmartFlo preflight metadata from query params: stream_id_present={} call_id_present={}",
            True,
            bool(query_call_id),
        )
        return {"stream_id": query_stream_id, "call_id": query_call_id}

    for _ in range(25):
        try:
            message = await asyncio.wait_for(websocket.receive(), timeout=2.0)
        except asyncio.TimeoutError:
            continue
        except Exception:
            logger.exception("Failed while waiting for SmartFlo start metadata")
            break

        message_type = str(message.get("type", "")).strip().lower()
        if message_type == "websocket.disconnect":
            logger.warning(
                "SmartFlo disconnected before start metadata: code={} reason='{}'",
                message.get("code"),
                message.get("reason", ""),
            )
            return {"stream_id": "", "call_id": ""}

        if message_type != "websocket.receive":
            continue

        raw_text = message.get("text")
        if raw_text is None and message.get("bytes") is not None:
            raw_text = message["bytes"].decode("utf-8", errors="ignore")
        if not raw_text:
            continue

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning("Ignoring non-JSON SmartFlo preflight message")
            continue

        event = str(data.get("event", "")).strip().lower()
        start = data.get("start") or {}
        if not isinstance(start, dict):
            start = {}

        stream_id = str(
            start.get("streamSid")
            or data.get("streamSid")
            or start.get("stream_id")
            or data.get("stream_id")
            or ""
        ).strip()
        call_id = str(
            start.get("callSid")
            or data.get("callSid")
            or start.get("call_id")
            or data.get("call_id")
            or ""
        ).strip()

        logger.info(
            "SmartFlo preflight event='{}' stream_id_present={} call_id_present={}",
            event,
            bool(stream_id),
            bool(call_id),
        )

        if stream_id and event in {"connected", "start", "media", "mark", "dtmf", "stop"}:
            return {"stream_id": stream_id, "call_id": call_id}

        if event == "start":
            return {"stream_id": stream_id, "call_id": call_id}

    logger.error("SmartFlo start metadata not found before call setup timeout")
    return {"stream_id": "", "call_id": ""}


async def run_bot(websocket, handle_sigint: bool = False):
    profile, agent_mod = _load_agent_module()
    logger.info("SmartFlo profile selected: {}", profile)
    _configure_smartflo_rag_mode(profile)
    _configure_smartflo_live_ws_timeout(profile)

    call_data = await parse_smartflo_start(websocket)
    if not call_data["stream_id"]:
        logger.error("Missing SmartFlo stream_id; closing call to avoid silent timeout")
        try:
            await websocket.close(code=1011)
        except Exception:
            logger.debug("WebSocket already closed before call setup abort")
        return

    logger.info(
        "SmartFlo call started: profile={} stream_id={} call_id={}",
        profile,
        call_data["stream_id"],
        call_data["call_id"],
    )

    serializer = SmartFloFrameSerializer(
        stream_sid=call_data["stream_id"],
        call_sid=call_data["call_id"],
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    rag_max_chars, rag_max_chars_detail, rag_max_chunks, rag_max_chunks_detail = _limits_for_profile(profile)

    rag = None
    rag_api_key = ""
    kb_text = ""
    rag_init_failed = False
    rag_init_lock = asyncio.Lock()

    def prepare_context(q: str, c: str) -> str:
        if profile == "apsagent":
            return agent_mod.prepare_rag_context(q, c, kb_text)
        return agent_mod.prepare_rag_context(q, c)

    async def ensure_rag_ready() -> bool:
        nonlocal rag, rag_api_key, kb_text, rag_init_failed

        if rag is not None:
            return True
        if rag_init_failed:
            return False

        async with rag_init_lock:
            if rag is not None:
                return True
            if rag_init_failed:
                return False

            try:
                if profile == "apsagent":
                    rag, rag_api_key, kb_text = agent_mod.create_knowledge_rag()
                else:
                    rag, rag_api_key = agent_mod.create_knowledge_rag()
                logger.info("SmartFlo RAG initialized for profile='{}'", profile)
                return True
            except Exception:
                rag_init_failed = True
                logger.exception("SmartFlo RAG initialization failed; continuing without RAG tool")
                return False

    llm = agent_mod.create_gemini_live_service(
        tools=[{"function_declarations": [agent_mod.RAG_TOOL_DECLARATION]}]
    )

    async def handle_retrieve_knowledge(params: FunctionCallParams):
        query = str(params.arguments.get("query", "")).strip()
        if not query:
            await params.result_callback(
                {
                    "context": (
                        "క్షమించండి, మీ ప్రశ్న స్పష్టంగా అర్థం కాలేదు. "
                        "దయచేసి పంట పేరు మరియు సమస్యను ఒక చిన్న వాక్యంలో చెప్పండి."
                    )
                }
            )
            return

        if not await ensure_rag_ready():
            await params.result_callback(
                {
                    "context": (
                        "క్షమించండి, ఇప్పుడు నాలెడ్జ్ బేస్ సేవ తాత్కాలికంగా అందుబాటులో లేదు. "
                        "దయచేసి మీ ప్రశ్నను సరళంగా మళ్లీ అడగండి."
                    )
                }
            )
            return

        try:
            retrieved = await rag.retrieve_async(query, api_key=rag_api_key)
        except Exception:
            logger.exception("SmartFlo RAG retrieval failed")
            await params.result_callback(
                {
                    "context": (
                        "క్షమించండి, ప్రస్తుతం నాలెడ్జ్ బేస్ నుంచి సమాచారం తెచ్చేటప్పుడు సమస్య వచ్చింది. "
                        "దయచేసి అదే ప్రశ్నను మరోసారి చెప్పగలరా?"
                    )
                }
            )
            return

        if not retrieved:
            retrieved = "No relevant information found in the knowledge base."

        retrieved = prepare_context(query, retrieved)

        detail_intent = agent_mod._is_detail_intent(query)
        max_chunks = rag_max_chunks_detail if detail_intent else rag_max_chunks
        max_chars = rag_max_chars_detail if detail_intent else rag_max_chars

        retrieved = agent_mod._limit_rag_chunks(retrieved, max_chunks)
        if len(retrieved) > max_chars:
            retrieved = retrieved[:max_chars]

        await params.result_callback({"context": retrieved})

    llm.register_function(
        "retrieve_knowledge",
        handle_retrieve_knowledge,
        cancel_on_interruption=False,
        timeout_secs=25,
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline_components = [
        transport.input(),
        user_aggregator,
        llm,
    ]

    pipeline_components.append(EscalationDetector(task=None, call_id=call_data["call_id"]))
    pipeline_components += [
        transport.output(),
        assistant_aggregator,
    ]

    pipeline = Pipeline(pipeline_components)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    for processor in pipeline_components:
        if isinstance(processor, EscalationDetector) and processor._task is None:
            processor._task = task

    conversation_started = False

    async def start_once():
        nonlocal conversation_started
        if conversation_started:
            return
        conversation_started = True
        context.add_message({"role": "user", "content": "Please introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("SmartFlo client connected")
        await start_once()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("SmartFlo client disconnected")
        await task.cancel()

    # SmartFlo connections are already accepted at FastAPI level, and depending on
    # transport timing the on_client_connected callback may not fire. Kick off the
    # first turn explicitly so the caller hears speech immediately.
    await start_once()

    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)
