# RNF Agnet Detailed Guide

This document explains exactly how RNF agnet works in this repository, how to run it reliably, and how to troubleshoot common failures.

## What RNF Agnet Is

RNF agnet is an independent real-time voice bot built with:

- Pipecat transport and pipeline runtime
- Gemini Live speech-to-speech model service
- Function calling with retrieve_knowledge(query)
- RNF-local RAG implementation in rnf_gemini/rag

It is isolated from apsagent by:

- Different entry file
- Different default prompt file
- Different default KB path
- RNF_* env namespace and cache defaults
- Different UI port (7861 by default)

## Core Files

- Bot runtime: rnf_gemini/rnf_agnet.py
- Prompt file: rnf_gemini/rnf_agnet_system_prompt.txt
- Knowledge base: rnf_gemini/knowledge_base/rnf_agnet_kb.txt
- RNF env template: rnf_gemini/config/rnf_agnet.env.example
- Start script from server root: start_rnf_agnet.sh
- Start script from RNF folder: rnf_gemini/start_rnf_agnet.sh

## Functional Architecture

### Conversation pipeline

transport.input -> user aggregator -> GeminiLiveLLMService -> transport.output -> assistant aggregator

### Tool architecture

1. Model decides to call retrieve_knowledge.
2. handle_retrieve_knowledge executes async retrieval.
3. TieredRAG selects strategy:
	- FastRAG for speed routes or missing embedding route
	- SimpleRAG for embedding-based semantic retrieval
4. Retrieved context is cleaned and bounded.
5. Tool result is returned to model for grounded response.

### Context shaping behavior

- Non-price requests remove repetitive representative transfer lines.
- Detailed intent can use larger chunk/character budgets.
- Default requests use tighter budgets for latency.

## How RNF Agnet Loads Config

load_dotenv is non-overriding in RNF runtime, so shell vars can override file vars.

Resolution order highlights:

- Model: RNF_GEMINI_MODEL then GEMINI_MODEL
- Voice: RNF_GEMINI_VOICE then GEMINI_VOICE
- API version: RNF_GEMINI_API_VERSION then GEMINI_API_VERSION
- Prompt file: RNF_GEMINI_SYSTEM_PROMPT_FILE then default rnf_gemini/rnf_agnet_system_prompt.txt
- KB file: RNF_RAG_KNOWLEDGE_BASE_FILE then default knowledge_base/rnf_agnet_kb.txt

Prompt and KB path resolution supports both repo-relative and agent-relative paths.

## Main Commands

### Start from server root

cd /home/mesashivardhan4080/pipecat-quickstart/server

uv run rnf_gemini/rnf_agnet.py --transport webrtc --port 7861

or

bash start_rnf_agnet.sh

### Start from rnf_gemini folder

cd /home/mesashivardhan4080/pipecat-quickstart/server/rnf_gemini

uv run rnf_agnet.py --transport webrtc --port 7861

or

bash start_rnf_agnet.sh

### PowerShell-safe start

bash -lc 'cd /home/mesashivardhan4080/pipecat-quickstart/server; bash start_rnf_agnet.sh'

### Stop

cd /home/mesashivardhan4080/pipecat-quickstart/server

bash stop_rnf_agnet.sh

### Client URL

http://localhost:7861/client/

### API connectivity check

curl -sS -X POST http://localhost:7861/start -H "Content-Type: application/json" -d '{"createDailyRoom":false,"enableDefaultIceServers":true,"transport":"webrtc"}'

If healthy, response includes sessionId.

## Environment Keys (RNF)

### Required

- GOOGLE_API_KEY

### Recommended for semantic embeddings

- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_API_VERSION
- AZURE_OPENAI_EMBEDDING_DEPLOYMENT

### RNF tuning keys

- RNF_GEMINI_MODEL
- RNF_GEMINI_VOICE
- RNF_GEMINI_API_VERSION
- RNF_GEMINI_SYSTEM_PROMPT_FILE
- RNF_GEMINI_SYSTEM_INSTRUCTION
- RNF_RAG_KNOWLEDGE_BASE_FILE
- RNF_FAST_RAG_CHUNK_THRESHOLD
- RNF_CHUNK_SIZE
- RNF_TOP_K
- RNF_SIMILARITY_THRESHOLD
- RNF_MIN_CHUNKS
- RNF_RAG_CHUNK_MODE
- RNF_RAG_CACHE_DIR
- RNF_RAG_MAX_CONTEXT_CHARS
- RNF_RAG_MAX_CONTEXT_CHARS_DETAIL
- RNF_RAG_MAX_CHUNKS
- RNF_RAG_MAX_CHUNKS_DETAIL

Use rnf_gemini/config/rnf_agnet.env.example as the baseline.

## Common Errors and How to Fix

### 1) PowerShell path error for /home/...

Symptom:

cd /home/... fails in PowerShell.

Fix:

Use bash -lc wrapper.

### 2) uv command not found

Symptom:

start script fails with uv not found.

Fix:

Run inside bash/WSL where uv is installed and on PATH.

### 3) FileNotFoundError for RNF KB

Symptom:

Runtime fails while loading KB.

Fix:

Set RNF_RAG_KNOWLEDGE_BASE_FILE to rnf_gemini/knowledge_base/rnf_agnet_kb.txt or keep default.

### 4) RNF prompt fallback warning

Symptom:

Warning that RNF prompt file not found, using fallback.

Fix:

Set RNF_GEMINI_SYSTEM_PROMPT_FILE to rnf_gemini/rnf_agnet_system_prompt.txt and verify file exists.

### 5) Agent stuck on connecting

Symptom:

UI shows connecting and never becomes ready.

Fix:

Check terminal traceback first.
Then call /start endpoint manually to confirm session creation.

### 6) Port 7861 already in use

Symptom:

Bind error on startup.

Fix:

Run bash stop_rnf_agnet.sh and restart.

### 7) Low relevance answers or latency spikes

Symptom:

Poor grounding or slow answer generation.

Fix:

- Enable embedding keys for semantic route.
- Tune RNF_* chunk and context limits.
- Verify KB path points to intended file.

## Operational Best Practices

1. Run one agent at a time while troubleshooting.
2. Keep terminal logs open during first connect.
3. Validate /start before full browser testing.
4. Keep RNF prompt and KB paths explicit in .env for production stability.
