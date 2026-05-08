"""RNF RAG configuration constants.

These read RNF-prefixed environment variables first, then fall back to shared
keys for compatibility.
"""

from __future__ import annotations

import os

CHUNK_SIZE = int(os.getenv("RNF_CHUNK_SIZE", os.getenv("CHUNK_SIZE", "300")))
TOP_K = int(os.getenv("RNF_TOP_K", os.getenv("TOP_K", "6")))
SIMILARITY_THRESHOLD = float(
    os.getenv("RNF_SIMILARITY_THRESHOLD", os.getenv("SIMILARITY_THRESHOLD", "0.22"))
)
MIN_CHUNKS = int(os.getenv("RNF_MIN_CHUNKS", os.getenv("MIN_CHUNKS", "4")))
FAST_RAG_CHUNK_THRESHOLD = int(
    os.getenv("RNF_FAST_RAG_CHUNK_THRESHOLD", os.getenv("FAST_RAG_CHUNK_THRESHOLD", "10"))
)
RAG_CACHE_DIR = os.getenv("RNF_RAG_CACHE_DIR", os.getenv("RAG_CACHE_DIR", ".rag_cache/rnf_agnet"))
