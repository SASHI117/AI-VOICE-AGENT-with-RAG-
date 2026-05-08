"""RAG Configuration Constants.

All values can be overridden via environment variables or .env file.
"""

import os

from dotenv import load_dotenv

load_dotenv(override=True)

# Embedding model name (Azure OpenAI deployment name or OpenAI model name)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

# Words per chunk when splitting the knowledge base
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))

# Number of chunks returned per query
TOP_K = int(os.getenv("TOP_K", "6"))

# Cosine similarity minimum — chunks below this are filtered out (0–1, lower = more permissive)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# If fewer chunks pass the threshold, still return at least this many (fallback for noisy queries)
MIN_CHUNKS = int(os.getenv("MIN_CHUNKS", "5"))

# Directory where embedding vectors are cached as .npy files
RAG_CACHE_DIR = os.getenv("RAG_CACHE_DIR", ".rag_cache")

# If the knowledge base produces ≤ this many chunks, TieredRAG uses FastRAG (TF-IDF, no API calls)
# Otherwise it falls back to SimpleRAG (OpenAI embeddings, higher quality but ~1s latency)
FAST_RAG_CHUNK_THRESHOLD = int(os.getenv("FAST_RAG_CHUNK_THRESHOLD", "20"))
