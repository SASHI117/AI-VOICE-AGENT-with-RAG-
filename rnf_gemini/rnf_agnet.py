#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""rnf_agnet - Independent Gemini Live + Pipecat + RAG voice agent.

Run examples:

    uv run rnf_gemini/rnf_agnet.py --transport webrtc --port 7861

This agent is isolated from the default root bot via:
- dedicated KB default path
- dedicated system prompt default path
- dedicated RAG cache dir
- RNF_* environment variable overrides
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.genai.types import HttpOptions
from loguru import logger
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import DailyRunnerArguments, RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, GeminiModalities
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport


load_dotenv(override=False)


def _apply_rnf_rag_env_overrides() -> None:
    """Set safe RNF defaults for local RAG settings."""
    os.environ.setdefault("RNF_RAG_CHUNK_MODE", "qa")
    os.environ.setdefault("RNF_RAG_CACHE_DIR", ".rag_cache/rnf_agnet")


_apply_rnf_rag_env_overrides()

try:
    # Works when launched from project root: uv run rnf_gemini/rnf_agnet.py
    from rnf_gemini.rag import TieredRAG  # type: ignore[attr-defined]  # noqa: E402
except ModuleNotFoundError:
    # Works when launched inside rnf_gemini: uv run rnf_agnet.py
    from rag import TieredRAG  # type: ignore[no-redef]  # noqa: E402


RAG_TOOL_DECLARATION = {
    "name": "retrieve_knowledge",
    "description": (
        "Search the RNF agriculture knowledge base for factual answers. "
        "Use for crop, pest, disease, product usage, and company-related factual questions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "User question to search in the RNF knowledge base.",
            }
        },
        "required": ["query"],
    },
}


def _strip_inline_kb(prompt_text: str) -> str:
    start_token = "KNOWLEDGE BASE"
    end_token = "END OF KNOWLEDGE BASE"

    start_idx = prompt_text.find(start_token)
    if start_idx == -1:
        return prompt_text

    end_idx = prompt_text.find(end_token, start_idx)
    if end_idx == -1:
        logger.warning("Prompt has '{}' without '{}', keeping original.", start_token, end_token)
        return prompt_text

    return prompt_text[:start_idx].strip()


def _with_tool_guidance(system_instruction: str) -> str:
    guidance = (
        "Tool usage guidance:\n"
        "- For greetings/casual chat, answer directly without tools.\n"
        "- For agriculture/product/company factual queries, call retrieve_knowledge first.\n"
        "- Do not provide factual claims before retrieve_knowledge returns.\n"
        "- After retrieval, answer only from returned context and avoid hallucinations.\n"
        "- If context is insufficient, say clearly information is unavailable in knowledge base.\n"
        "- If user asks for detailed explanation, provide a more detailed grounded answer."
    )
    return f"{system_instruction}\n\n{guidance}".strip()


def _contains_any(text: str, terms: list[str]) -> bool:
    lower = text.lower()
    return any(t.lower() in lower for t in terms)


def _is_price_delivery_intent(query: str) -> bool:
    return _contains_any(
        query,
        [
            "price",
            "availability",
            "delivery",
            "ధర",
            "లభ్యత",
            "డెలివరీ",
            "dispatch",
            "ఎన్ని రూపాయలు",
        ],
    )


def _is_detail_intent(query: str) -> bool:
    return _contains_any(
        query,
        [
            "వివరంగా",
            "ఇంకా",
            "మరింత",
            "పూర్తిగా",
            "in detail",
            "detailed",
            "explain",
            "full",
            "inka",
            "baga",
        ],
    )


def _remove_representative_lines(context_text: str) -> str:
    patterns = [
        r"నేను\s*మీ\s*కాల్[^\n.]*ప్రతినిధి[^\n.]*[.।]?",
        r"ప్రతినిధితో\s*కాల్\s*కలుపుతాను[^\n.]*[.।]?",
        r"ప్రతినిధికి\s*కాల్\s*కలుప[^\n.]*[.।]?",
        r"మా\s*సంస్థ\s*ప్రతినిధి[^\n.]*[.।]?",
    ]
    out = context_text
    for p in patterns:
        out = re.sub(p, "", out, flags=re.IGNORECASE)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _limit_rag_chunks(context_text: str, max_chunks: int) -> str:
    if max_chunks <= 0 or not context_text.strip():
        return context_text
    parts = [p for p in context_text.split("\n\n---\n\n") if p.strip()]
    if len(parts) <= max_chunks:
        return context_text
    return "\n\n---\n\n".join(parts[:max_chunks])


def _count_rag_chunks(context_text: str) -> int:
    if not context_text.strip():
        return 0
    return context_text.count("\n\n---\n\n") + 1


def prepare_rag_context(query: str, retrieved: str) -> str:
    out = retrieved
    if not _is_price_delivery_intent(query):
        out = _remove_representative_lines(out)
    return out


def _load_knowledge_text() -> str:
    agent_root = Path(__file__).resolve().parent
    repo_root = agent_root.parent

    # Keep RNF isolated from old-agent RAG_KNOWLEDGE_BASE_FILE.
    # Support both:
    # - knowledge_base/rnf_agnet_kb.txt
    # - rnf_gemini/knowledge_base/rnf_agnet_kb.txt
    kb_file = os.getenv("RNF_RAG_KNOWLEDGE_BASE_FILE", "knowledge_base/rnf_agnet_kb.txt")
    kb_path = Path(kb_file)

    if not kb_path.is_absolute():
        candidates = [repo_root / kb_path, agent_root / kb_path]
        kb_path = next((p for p in candidates if p.exists()), candidates[0])

    kb_text = kb_path.read_text(encoding="utf-8").strip()
    if not kb_text:
        raise ValueError(f"RNF knowledge base file is empty: {kb_path}")

    logger.info("Loaded RNF KB from '{}' ({} chars)", kb_path, len(kb_text))
    return kb_text


def create_knowledge_rag() -> tuple[TieredRAG, str]:
    kb_text = _load_knowledge_text()
    embedding_api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    chunk_threshold = int(os.getenv("RNF_FAST_RAG_CHUNK_THRESHOLD", os.getenv("FAST_RAG_CHUNK_THRESHOLD", "10")))

    if not embedding_api_key:
        logger.warning("No embedding API key found. Falling back to FastRAG mode.")
        chunk_threshold = 10**9

    rag = TieredRAG(
        documents=kb_text,
        api_key=embedding_api_key,
        chunk_threshold=chunk_threshold,
    )
    logger.info("RNF RAG ready: strategy='{}', chunks={}", rag.strategy, rag.num_chunks)
    return rag, embedding_api_key


def load_system_instruction() -> str:
    env_prompt = os.getenv("RNF_GEMINI_SYSTEM_INSTRUCTION", "").strip()
    if env_prompt:
        return _with_tool_guidance(_strip_inline_kb(env_prompt))

    agent_root = Path(__file__).resolve().parent
    repo_root = agent_root.parent
    default_prompt_path = agent_root / "rnf_agnet_system_prompt.txt"
    prompt_file = os.getenv("RNF_GEMINI_SYSTEM_PROMPT_FILE") or str(default_prompt_path)
    prompt_path = Path(prompt_file)
    if not prompt_path.is_absolute():
        candidates = [repo_root / prompt_path, agent_root / prompt_path]
        prompt_path = next((p for p in candidates if p.exists()), candidates[0])

    try:
        prompt_text = prompt_path.read_text(encoding="utf-8").strip()
        if prompt_text:
            logger.info("Loaded RNF system prompt from '{}'", prompt_path)
            return _with_tool_guidance(_strip_inline_kb(prompt_text))
    except FileNotFoundError:
        logger.warning("RNF system prompt file '{}' not found; using fallback.", prompt_path)

    fallback = (
        "You are a helpful agriculture assistant in a voice conversation. "
        "Keep responses natural and grounded in tool output for factual answers."
    )
    return _with_tool_guidance(fallback)


def create_gemini_live_service(tools: list[dict[str, Any]] | None = None) -> GeminiLiveLLMService:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    requested_model = os.getenv("RNF_GEMINI_MODEL", os.getenv("GEMINI_MODEL", "models/gemini-3.1-flash-live-preview"))
    voice = os.getenv("RNF_GEMINI_VOICE", os.getenv("GEMINI_VOICE", "Charon"))
    api_version = os.getenv("RNF_GEMINI_API_VERSION", os.getenv("GEMINI_API_VERSION", "v1alpha"))
    ws_open_timeout = float(os.getenv("RNF_GEMINI_WS_OPEN_TIMEOUT_SECS", os.getenv("GEMINI_WS_OPEN_TIMEOUT_SECS", "30")))

    if requested_model in {"models/gemini-3.1-flash", "gemini-3.1-flash"}:
        effective_model = "models/gemini-3.1-flash-live-preview"
    else:
        effective_model = requested_model

    return GeminiLiveLLMService(
        api_key=google_api_key,
        tools=tools,
        settings=GeminiLiveLLMService.Settings(
            model=effective_model,
            voice=voice,
            modalities=GeminiModalities.AUDIO,
            system_instruction=load_system_instruction(),
        ),
        http_options=HttpOptions(
            api_version=api_version,
            async_client_args={"open_timeout": ws_open_timeout},
        ),
    )


async def run_bot(transport: BaseTransport):
    logger.info("Starting RNF Gemini bot")

    rag, rag_api_key = create_knowledge_rag()

    llm = create_gemini_live_service(
        tools=[{"function_declarations": [RAG_TOOL_DECLARATION]}]
    )

    rag_max_context_chars = int(os.getenv("RNF_RAG_MAX_CONTEXT_CHARS", "5000"))
    rag_max_context_chars_detail = int(os.getenv("RNF_RAG_MAX_CONTEXT_CHARS_DETAIL", "8000"))
    rag_max_chunks = int(os.getenv("RNF_RAG_MAX_CHUNKS", "4"))
    rag_max_chunks_detail = int(os.getenv("RNF_RAG_MAX_CHUNKS_DETAIL", "6"))

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

        logger.info("RNF RAG tool call [{}] query='{}'", params.tool_call_id, query)

        try:
            retrieved = await rag.retrieve_async(query, api_key=rag_api_key)
        except Exception:
            logger.exception("RNF RAG retrieval failed")
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

        retrieved = prepare_rag_context(query, retrieved)

        detail_intent = _is_detail_intent(query)
        target_max_chunks = rag_max_chunks_detail if detail_intent else rag_max_chunks
        target_max_chars = rag_max_context_chars_detail if detail_intent else rag_max_context_chars

        retrieved = _limit_rag_chunks(retrieved, target_max_chunks)

        chunks_before = _count_rag_chunks(retrieved)
        trimmed = False
        if len(retrieved) > target_max_chars:
            retrieved = retrieved[:target_max_chars]
            trimmed = True
        chunks_after = _count_rag_chunks(retrieved)

        logger.info(
            "RNF RAG result [{}]: detail_intent={}, max_chunks={}, max_chars={}, chunks_before={}, chunks_after={}, chars={}, trimmed={}",
            params.tool_call_id,
            detail_intent,
            target_max_chunks,
            target_max_chars,
            chunks_before,
            chunks_after,
            len(retrieved),
            trimmed,
        )

        await params.result_callback({"context": retrieved})

    llm.register_function(
        "retrieve_knowledge",
        handle_retrieve_knowledge,
        cancel_on_interruption=False,
        timeout_secs=25,
    )

    context = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(context)

    pipeline = Pipeline([
        transport.input(),
        user_agg,
        llm,
        transport.output(),
        assistant_agg,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[],
    )

    conversation_started = False

    async def start_conversation_once():
        nonlocal conversation_started
        if conversation_started:
            return
        conversation_started = True
        context.add_message({"role": "user", "content": "Please introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

    @task.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await start_conversation_once()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("RNF client connected")
        await start_conversation_once()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("RNF client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    transport = None

    match runner_args:
        case DailyRunnerArguments():
            try:
                from pipecat.transports.daily.transport import DailyParams, DailyTransport
            except ImportError:
                logger.error("Daily transport dependencies missing for RNF bot.")
                return

            transport = DailyTransport(
                runner_args.room_url,
                runner_args.token,
                "RNF Gemini Bot",
                params=DailyParams(audio_in_enabled=True, audio_out_enabled=True),
            )
        case SmallWebRTCRunnerArguments():
            webrtc_connection: SmallWebRTCConnection = runner_args.webrtc_connection
            transport = SmallWebRTCTransport(
                webrtc_connection=webrtc_connection,
                params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
            )
        case _:
            logger.error("Unsupported runner args type: {}", type(runner_args))
            return

    await run_bot(transport)


if __name__ == "__main__":
    import sys

    if "--port" not in sys.argv:
        sys.argv.extend(["--port", os.getenv("RNF_UI_PORT", "7861")])

    from pipecat.runner.run import main

    main()
