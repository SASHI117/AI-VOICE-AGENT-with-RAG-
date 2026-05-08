"""RAG v2 package (BAAI BGE-M3 local retrieval with enhancements)."""

from .bge_m3_rag import BGEM3RAG
from .rag_config import (
    RAG_V2_BATCH_SIZE,
    RAG_V2_CACHE_DIR,
    RAG_V2_CHUNK_MODE,
    RAG_V2_CHUNK_SIZE,
    RAG_V2_HYBRID_ALPHA,
    RAG_V2_MAX_LENGTH,
    RAG_V2_MIN_CHUNKS,
    RAG_V2_MODEL_NAME,
    RAG_V2_QUERY_MAX_LENGTH,
    RAG_V2_SIMILARITY_THRESHOLD,
    RAG_V2_TOP_K,
    RAG_V2_USE_SPARSE,
    # Enhancement configs
    RAG_V2_ENABLE_TELUGU_PREPROCESSING,
    RAG_V2_TELUGU_NORMALIZE_VARIANTS,
    RAG_V2_ENABLE_RERANKING,
    RAG_V2_RERANKER_MODEL,
    RAG_V2_RERANKING_WEIGHT,
    RAG_V2_ENABLE_DOMAIN_TUNING,
)

# Optional enhancement imports (only import if available)
try:
    from .telugu_preprocessing import TeluguPreprocessor
except ImportError:
    TeluguPreprocessor = None

try:
    from .reranker import BGEReranker
except ImportError:
    BGEReranker = None

try:
    from .domain_tuner import DomainAlphaTuner
except ImportError:
    DomainAlphaTuner = None

__all__ = [
    "BGEM3RAG",
    "RAG_V2_MODEL_NAME",
    "RAG_V2_BATCH_SIZE",
    "RAG_V2_MAX_LENGTH",
    "RAG_V2_QUERY_MAX_LENGTH",
    "RAG_V2_CACHE_DIR",
    "RAG_V2_CHUNK_SIZE",
    "RAG_V2_CHUNK_MODE",
    "RAG_V2_TOP_K",
    "RAG_V2_MIN_CHUNKS",
    "RAG_V2_SIMILARITY_THRESHOLD",
    "RAG_V2_HYBRID_ALPHA",
    "RAG_V2_USE_SPARSE",
    # Enhancement exports
    "RAG_V2_ENABLE_TELUGU_PREPROCESSING",
    "RAG_V2_TELUGU_NORMALIZE_VARIANTS",
    "RAG_V2_ENABLE_RERANKING",
    "RAG_V2_RERANKER_MODEL",
    "RAG_V2_RERANKING_WEIGHT",
    "RAG_V2_ENABLE_DOMAIN_TUNING",
    "TeluguPreprocessor",
    "BGEReranker",
    "DomainAlphaTuner",
]
