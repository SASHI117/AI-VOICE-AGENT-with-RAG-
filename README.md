# Voice Agents Runbook

This repository runs two independent real-time voice agents:

- apsagent: primary production assistant.
- rnf_agnet: RNF-specific assistant with isolated RAG defaults and files.

Both use Pipecat + Gemini Live + function-calling RAG, but they are isolated by port and file defaults.

## Architecture Overview

### Shared runtime architecture

1. Client connects to Small WebRTC endpoint exposed by Pipecat runner.
2. Pipeline receives user audio.
3. Gemini Live service handles realtime speech conversation.
4. Model can call retrieve_knowledge(query) tool for factual questions.
5. Tool handler queries TieredRAG and returns bounded context.
6. Gemini responds to user with grounded answer.

### Pipeline shape (both agents)

transport.input -> user context aggregator -> GeminiLiveLLMService -> transport.output -> assistant context aggregator

### RAG shape (both agents)

query -> preprocess -> TieredRAG strategy select

- FastRAG if no embedding key or chunk threshold route
- SimpleRAG embedding route if embedding keys are available

Then context shaping happens:

- remove repetitive representative lines for non-price intents
- cap chunks and chars for latency control
- return final context through retrieve_knowledge result callback

## Agent Functionality

### apsagent

- Entry file: apsagent.py
- Prompt default: apsagent_system_prompt.txt
- KB default: knowledge_base/apsagent_kb.txt
- Port default in scripts: 7860
- RAG implementation: rag_sharekit/rag

### rnf_agnet

- Entry file: rnf_gemini/rnf_agnet.py
- Prompt default: rnf_gemini/rnf_agnet_system_prompt.txt
- KB default: rnf_gemini/knowledge_base/rnf_agnet_kb.txt
- Port default in scripts: 7861
- RAG implementation: rnf_gemini/rag

## File Map

- apsagent.py
- apsagent_system_prompt.txt
- knowledge_base/apsagent_kb.txt
- start_apsagent.sh
- stop_apsagent.sh
- rnf_gemini/rnf_agnet.py
- rnf_gemini/rnf_agnet_system_prompt.txt
- rnf_gemini/knowledge_base/rnf_agnet_kb.txt
- start_rnf_agnet.sh
- stop_rnf_agnet.sh

## Setup

### Prerequisites

1. Python 3.10+.
2. uv installed.
3. .env file in repository root.

### Install dependencies

Run from repository root:

python -m venv .venv
.venv\\Scripts\\activate
pip install -e .

Or with uv:

uv sync

## Environment Variables

### Required for both agents

- GOOGLE_API_KEY

### Recommended for semantic embedding RAG

- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_API_VERSION
- AZURE_OPENAI_EMBEDDING_DEPLOYMENT

### Optional apsagent settings

- GEMINI_MODEL
- GEMINI_VOICE
- GEMINI_API_VERSION
- GEMINI_SYSTEM_PROMPT_FILE
- GEMINI_SYSTEM_INSTRUCTION
- RAG_KNOWLEDGE_BASE_FILE
- FAST_RAG_CHUNK_THRESHOLD
- CHUNK_SIZE
- TOP_K
- SIMILARITY_THRESHOLD
- MIN_CHUNKS
- RAG_MAX_CONTEXT_CHARS
- RAG_MAX_CONTEXT_CHARS_DETAIL
- RAG_MAX_CHUNKS
- RAG_MAX_CHUNKS_DETAIL

### Optional rnf_agnet settings

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

Reference examples:

- .env.example
- rnf_gemini/config/rnf_agnet.env.example

## Main Commands

### Linux Bash or WSL shell

cd /home/mesashivardhan4080/pipecat-quickstart/server

Start apsagent:

bash start_apsagent.sh

Start rnf_agnet:

bash start_rnf_agnet.sh

Stop apsagent:

bash stop_apsagent.sh

Stop rnf_agnet:

bash stop_rnf_agnet.sh

### Direct run (without helper scripts)

cd /home/mesashivardhan4080/pipecat-quickstart/server

uv run apsagent.py --transport webrtc --port 7860

uv run rnf_gemini/rnf_agnet.py --transport webrtc --port 7861

### PowerShell-safe versions

If you are in PowerShell, Linux paths and uv in bash scripts can fail unless invoked through bash -lc.

Use:

bash -lc 'cd /home/mesashivardhan4080/pipecat-quickstart/server; bash start_apsagent.sh'

bash -lc 'cd /home/mesashivardhan4080/pipecat-quickstart/server; bash start_rnf_agnet.sh'

bash -lc 'cd /home/mesashivardhan4080/pipecat-quickstart/server; bash stop_apsagent.sh; bash stop_rnf_agnet.sh'

## Client URLs

- apsagent: http://localhost:7860/client/
- rnf_agnet: http://localhost:7861/client/

## Connectivity Checks

A healthy running agent returns JSON with sessionId on start endpoint.

apsagent:

curl -sS -X POST http://localhost:7860/start -H "Content-Type: application/json" -d '{"createDailyRoom":false,"enableDefaultIceServers":true,"transport":"webrtc"}'

rnf_agnet:

curl -sS -X POST http://localhost:7861/start -H "Content-Type: application/json" -d '{"createDailyRoom":false,"enableDefaultIceServers":true,"transport":"webrtc"}'

## Typical One-by-One Validation Flow

1. Start apsagent.
2. Open apsagent client URL.
3. Verify response and tool calling.
4. Stop apsagent.
5. Start rnf_agnet.
6. Open rnf_agnet client URL.
7. Verify response and tool calling.
8. Stop rnf_agnet.

## Common Errors and Fixes

### 1) Path not found in PowerShell for /home/...

Symptom:

cd /home/... fails in PowerShell.

Cause:

Linux path used in PowerShell filesystem provider.

Fix:

Use bash -lc wrapper commands shown above.

### 2) uv command not found when running shell scripts

Symptom:

start script logs uv: command not found.

Cause:

Script executed in environment without uv on PATH.

Fix:

Run from bash with proper environment or use explicit venv/uv path.

### 3) Agent UI shows connecting but not connected

Symptom:

Client panel remains connecting.

Cause:

Server-side runtime exception during call start.

Fix:

Check terminal logs first.
Then test /start endpoint manually.

### 4) RNF knowledge base file not found

Symptom:

FileNotFoundError at RNF run start.

Cause:

Incorrect RNF_RAG_KNOWLEDGE_BASE_FILE or wrong relative base.

Fix:

Set RNF_RAG_KNOWLEDGE_BASE_FILE to rnf_gemini/knowledge_base/rnf_agnet_kb.txt in .env or leave default.

### 5) Prompt file fallback warning

Symptom:

Warning that RNF prompt file was not found and fallback prompt is used.

Cause:

RNF_GEMINI_SYSTEM_PROMPT_FILE points to wrong path.

Fix:

Set RNF_GEMINI_SYSTEM_PROMPT_FILE=rnf_gemini/rnf_agnet_system_prompt.txt.

### 6) Port already in use

Symptom:

Bind error on 7860 or 7861.

Cause:

Previous agent process still running.

Fix:

Run stop scripts, then restart.

### 7) RAG behaves poorly or slowly

Symptom:

Low relevance or high latency.

Cause:

FastRAG fallback only, or context limits too strict, or wrong thresholds.

Fix:

Provide embedding keys for semantic retrieval and tune thresholds/chunk limits in .env.

### 8) No audio in browser

Symptom:

Connected but no speech.

Cause:

Mic permissions/device routing issues.

Fix:

Allow browser microphone access and verify selected input/output devices in client panel.

## Operational Notes

- apsagent and rnf_agnet can run in parallel on separate ports.
- For debugging, always capture terminal output during startup and first client connection.
- Keep prompts and KB files versioned with agent-specific naming to avoid cross-agent path bleed.
