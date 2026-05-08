# Pipecat RAG Integration — Share Kit

A drop-in RAG (Retrieval-Augmented Generation) module for Pipecat voice AI bots.
Prevents hallucinations by making the LLM search a knowledge base before answering.

---

## What is inside

```
rag_sharekit/
├── rag/
│   ├── __init__.py        — package exports
│   ├── helpers.py         — SimpleRAG, FastRAG, TieredRAG classes
│   ├── rag_config.py      — configuration constants (all env-overridable)
│   └── preprocessing.py   — optional STT query cleanup
├── example_bot.py         — complete working Pipecat bot using TieredRAG
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# → fill in AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY
# → or use OPENAI_API_KEY for standard OpenAI

# 3. Drop in your knowledge base
# Create knowledge_base.txt with your domain content (any plain text)

# 4. Run the example bot
python example_bot.py
```

---

## Choosing a RAG class

| Class | When to use | Latency | Requires API |
|---|---|---|---|
| `FastRAG` | Small KB (< ~20 chunks, ~8 000 words) | < 5 ms | No |
| `SimpleRAG` | Large KB, needs semantic precision | ~1 s first call, then < 10 ms | Yes |
| **`TieredRAG`** | **Default — auto-picks above** | best of both | Conditional |

> Rule of thumb: use `TieredRAG` everywhere. It inspects your KB at startup and
> picks the right strategy automatically.

---

## Integrating into your own bot

### Step 1 — Initialise once at startup

```python
from rag import TieredRAG

with open("knowledge_base.txt") as f:
    kb = f.read()

rag = TieredRAG(documents=kb, api_key=os.getenv("AZURE_OPENAI_API_KEY"))
```

### Step 2 — Register a function tool on the LLM

```python
TOOLS = [{
    "type": "function",
    "function": {
        "name": "retrieve_knowledge",
        "description": "Search the knowledge base to answer a question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question to search for."}
            },
            "required": ["query"],
        },
    },
}]
```

### Step 3 — Handle the tool call

```python
async def handle_tool_call(function_name, tool_call_id, arguments, llm, context, result_callback):
    if function_name == "retrieve_knowledge":
        query = arguments.get("query", "")

        # ✅ retrieve() returns a string — assign it directly
        result_text = await rag.retrieve_async(query, api_key=OPENAI_API_KEY)

        # ❌ WRONG — join() iterates every character of the string
        # result_text = "\n\n".join(rag.retrieve(query))

        if not result_text:
            result_text = "No relevant information found."

        await result_callback({"context": result_text})

llm.register_function("retrieve_knowledge", handle_tool_call)
```

### Step 4 — Add to pipeline

```python
context = OpenAILLMContext(messages, TOOLS)
context_aggregator = llm.create_context_aggregator(context)

pipeline = Pipeline([
    transport.input(),
    context_aggregator.user(),
    llm,
    context_aggregator.assistant(),
    transport.output(),
])
```

See `example_bot.py` for the complete runnable version.

---

## Tuning

All settings can be overridden in `.env`:

| Variable | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` | `400` | Words per chunk when splitting the KB |
| `TOP_K` | `6` | Number of chunks returned per query |
| `SIMILARITY_THRESHOLD` | `0.3` | Cosine cutoff for SimpleRAG (0 = return everything) |
| `MIN_CHUNKS` | `5` | Always return at least this many chunks (STT noise fallback) |
| `RAG_CACHE_DIR` | `.rag_cache` | Where embedding vectors are cached on disk |
| `FAST_RAG_CHUNK_THRESHOLD` | `20` | KBs with ≤ this many chunks use FastRAG |

---

## STT query cleanup (optional)

If your STT engine makes domain-specific mistakes, add corrections to
`rag/preprocessing.py → STT_TRANSLITERATION_DICT`:

```python
STT_TRANSLITERATION_DICT = {
    "fertlizer": "fertilizer",
    "my product name": "MyProductName",
}
```

`preprocess_stt_query()` is called automatically inside every `retrieve()` call.

---

## How the cache works

`SimpleRAG` generates embeddings once and saves them to `.rag_cache/` as `.npy` files.
The file is keyed by an MD5 hash of your knowledge base content.

- First run: embeddings generated (~30 s for a large document), saved to disk.
- Subsequent runs: loaded from disk in milliseconds.
- If you update the KB file, the hash changes and embeddings are regenerated automatically.

To force a cache rebuild, delete `.rag_cache/`.

---

## Common mistakes

**Wrong — join() on a string**
```python
context = "\n\n".join(rag.retrieve(query))   # iterates every character!
```

**Correct**
```python
context = rag.retrieve(query)                # already a formatted string
```

---

## Azure OpenAI vs standard OpenAI

The embedding client (`_get_embed_client`) reads `AZURE_OPENAI_*` variables by default.

To use standard OpenAI instead, edit `rag/helpers.py → _get_embed_client()`:

```python
from openai import OpenAI
_embed_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_embed_deployment = "text-embedding-3-large"
```
