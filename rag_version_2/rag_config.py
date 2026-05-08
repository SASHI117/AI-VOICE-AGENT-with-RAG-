"""RAG v2 configuration (BAAI BGE-M3 local retrieval).

All values can be overridden via environment variables or .env.
"""

import os

from dotenv import load_dotenv

load_dotenv(override=True)

RAG_V2_MODEL_NAME = os.getenv("RAG_V2_MODEL_NAME", "BAAI/bge-m3")
RAG_V2_DEVICE = os.getenv("RAG_V2_DEVICE", "cpu")
RAG_V2_USE_FP16 = os.getenv("RAG_V2_USE_FP16", "false").lower() in {"1", "true", "yes"}
RAG_V2_BATCH_SIZE = int(os.getenv("RAG_V2_BATCH_SIZE", "8"))
RAG_V2_MAX_LENGTH = int(os.getenv("RAG_V2_MAX_LENGTH", "512"))
RAG_V2_QUERY_MAX_LENGTH = int(os.getenv("RAG_V2_QUERY_MAX_LENGTH", "256"))

RAG_V2_TOP_K = int(os.getenv("RAG_V2_TOP_K", "4"))
RAG_V2_MIN_CHUNKS = int(os.getenv("RAG_V2_MIN_CHUNKS", "3"))
RAG_V2_SIMILARITY_THRESHOLD = float(os.getenv("RAG_V2_SIMILARITY_THRESHOLD", "0.10"))

RAG_V2_CACHE_DIR = os.getenv("RAG_V2_CACHE_DIR", ".rag_cache/rag_v2")
RAG_V2_HYBRID_ALPHA = float(os.getenv("RAG_V2_HYBRID_ALPHA", "0.45"))
RAG_V2_USE_SPARSE = os.getenv("RAG_V2_USE_SPARSE", "true").lower() in {"1", "true", "yes"}

RAG_V2_CHUNK_SIZE = int(os.getenv("RAG_V2_CHUNK_SIZE", "200"))
RAG_V2_CHUNK_MODE = os.getenv("RAG_V2_CHUNK_MODE", "qa_v2").strip().lower()

# ============ ENHANCEMENT 1: TELUGU PREPROCESSING ============
RAG_V2_ENABLE_TELUGU_PREPROCESSING = os.getenv("RAG_V2_ENABLE_TELUGU_PREPROCESSING", "true").lower() in {"1", "true", "yes"}
RAG_V2_TELUGU_REMOVE_STOP_WORDS = os.getenv("RAG_V2_TELUGU_REMOVE_STOP_WORDS", "false").lower() in {"1", "true", "yes"}
RAG_V2_TELUGU_NORMALIZE_VARIANTS = os.getenv("RAG_V2_TELUGU_NORMALIZE_VARIANTS", "true").lower() in {"1", "true", "yes"}

# ============ ENHANCEMENT 2: BGE-RERANKER ============
RAG_V2_ENABLE_RERANKING = os.getenv("RAG_V2_ENABLE_RERANKING", "false").lower() in {"1", "true", "yes"}
RAG_V2_RERANKER_MODEL = os.getenv("RAG_V2_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RAG_V2_RERANKING_WEIGHT = float(os.getenv("RAG_V2_RERANKING_WEIGHT", "0.6"))  # Weight for reranker vs initial score blend
RAG_V2_RERANKING_THRESHOLD = float(os.getenv("RAG_V2_RERANKING_THRESHOLD", "0.0"))
RAG_V2_RERANKING_BATCH_SIZE = int(os.getenv("RAG_V2_RERANKING_BATCH_SIZE", "32"))

# ============ ENHANCEMENT 3: PER-DOMAIN ALPHA TUNING ============
RAG_V2_ENABLE_DOMAIN_TUNING = os.getenv("RAG_V2_ENABLE_DOMAIN_TUNING", "false").lower() in {"1", "true", "yes"}
RAG_V2_DOMAIN_ALPHA_CONFIDENCE_THRESHOLD = float(os.getenv("RAG_V2_DOMAIN_ALPHA_CONFIDENCE_THRESHOLD", "0.5"))
RAG_V2_DOMAIN_ALPHA_BASE = float(os.getenv("RAG_V2_DOMAIN_ALPHA_BASE", "0.5"))  # Base alpha when no domain match
