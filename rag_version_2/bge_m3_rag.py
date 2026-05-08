"""BAAI BGE-M3 local RAG implementation with enhancements.

Drop-in interface compatible with SimpleRAG/TieredRAG:
  retrieve(query, api_key, k, threshold) -> str
  retrieve_async(query, api_key, k, threshold) -> str

Enhancements:
  1. Telugu diacritics preprocessing
  2. BGE-Reranker-v2-m3 second-stage ranking
  3. Per-domain alpha tuning for hybrid blend
"""

import asyncio
import hashlib
import os
import pickle
import re
import time

import numpy as np
from loguru import logger
from FlagEmbedding import BGEM3FlagModel

from rag_sharekit.rag.preprocessing import preprocess_stt_query

from .chunking import chunk_text_v2
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
    RAG_V2_USE_FP16,
    RAG_V2_USE_SPARSE,
    RAG_V2_DEVICE,
    # Enhancement configs
    RAG_V2_ENABLE_TELUGU_PREPROCESSING,
    RAG_V2_TELUGU_NORMALIZE_VARIANTS,
    RAG_V2_ENABLE_RERANKING,
    RAG_V2_RERANKING_WEIGHT,
    RAG_V2_RERANKING_BATCH_SIZE,
    RAG_V2_ENABLE_DOMAIN_TUNING,
    RAG_V2_DOMAIN_ALPHA_CONFIDENCE_THRESHOLD,
)

# Optional imports for enhancements
_TELUGU_PROCESSOR = None
_RERANKER = None
_DOMAIN_TUNER = None

def _get_telugu_processor():
    global _TELUGU_PROCESSOR
    if _TELUGU_PROCESSOR is None:
        try:
            from .telugu_preprocessing import TeluguPreprocessor
            _TELUGU_PROCESSOR = TeluguPreprocessor
            logger.info("Loaded Telugu preprocessor for diacritics normalization")
        except ImportError:
            logger.warning("Telugu preprocessing module not available")
    return _TELUGU_PROCESSOR

def _get_reranker():
    global _RERANKER
    if _RERANKER is None and RAG_V2_ENABLE_RERANKING:
        try:
            from .reranker import BGEReranker
            from .rag_config import RAG_V2_RERANKER_MODEL
            _RERANKER = BGEReranker(RAG_V2_RERANKER_MODEL, RAG_V2_DEVICE)
            logger.info("Loaded BGE reranker for second-stage ranking")
        except Exception as e:
            logger.warning("BGE reranker initialization failed: {}", e)
    return _RERANKER

def _get_domain_tuner():
    global _DOMAIN_TUNER
    if _DOMAIN_TUNER is None and RAG_V2_ENABLE_DOMAIN_TUNING:
        try:
            from .domain_tuner import DomainAlphaTuner
            _DOMAIN_TUNER = DomainAlphaTuner
            logger.info("Loaded domain alpha tuner for per-domain optimization")
        except ImportError:
            logger.warning("Domain tuner module not available")
    return _DOMAIN_TUNER

_BGE_MODEL: BGEM3FlagModel | None = None


def _get_bge_model() -> BGEM3FlagModel:
    global _BGE_MODEL
    if _BGE_MODEL is None:
        import torch
        # CPU optimization for sub-second latency
        # Reverted back to 4 threads as single thread increased latency on this specific CPU
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
        
        device = RAG_V2_DEVICE.strip().lower() or "cpu"
        use_fp16 = bool(RAG_V2_USE_FP16) and device != "cpu"
        logger.info("Loading BGE-M3 model '{}' (device={}, fp16={})", RAG_V2_MODEL_NAME, device, use_fp16)
        _BGE_MODEL = BGEM3FlagModel(
            RAG_V2_MODEL_NAME,
            use_fp16=use_fp16,
            device=device,
        )
    return _BGE_MODEL


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vec)
    if denom == 0:
        return vec
    return vec / denom


def _sparse_dot(q: dict[int, float], d: dict[int, float]) -> float:
    if not q or not d:
        return 0.0
    if len(q) > len(d):
        q, d = d, q
    return float(sum(float(w) * float(d.get(tok, 0.0)) for tok, w in q.items()))


def _is_telugu(text: str) -> bool:
    """Detect whether a query contains Telugu script characters."""
    return any("\u0C00" <= char <= "\u0C7F" for char in text)


class BGEM3RAG:
    """Local hybrid RAG using BAAI/bge-m3 (dense + sparse) with enhancements.

    Parameters
    ----------
    documents : str
        Full knowledge base text.
    cache_dir : str | None
        Directory to store embedding cache files.
    """

    def __init__(self, documents: str, cache_dir: str | None = None):
        self.strategy = "bge_m3"
        self.use_sparse = bool(RAG_V2_USE_SPARSE)
        self.hybrid_alpha = min(max(float(RAG_V2_HYBRID_ALPHA), 0.0), 1.0)
        self.domain_alpha_confidence_threshold = float(RAG_V2_DOMAIN_ALPHA_CONFIDENCE_THRESHOLD)
        self.enable_telugu_preprocessing = bool(RAG_V2_ENABLE_TELUGU_PREPROCESSING)
        
        # Enhancement: Initialize reranker if enabled
        self.reranker = _get_reranker() if RAG_V2_ENABLE_RERANKING else None
        self.enable_reranking = RAG_V2_ENABLE_RERANKING and self.reranker is not None
        self.reranking_weight = RAG_V2_RERANKING_WEIGHT if self.enable_reranking else 0.0
        
        # Enhancement: Initialize domain tuner if enabled
        self.domain_tuner = _get_domain_tuner() if RAG_V2_ENABLE_DOMAIN_TUNING else None
        self.enable_domain_tuning = RAG_V2_ENABLE_DOMAIN_TUNING and self.domain_tuner is not None

        if cache_dir is None:
            cache_dir = RAG_V2_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        self.chunks = chunk_text_v2(documents, RAG_V2_CHUNK_SIZE, RAG_V2_CHUNK_MODE)
        self.num_chunks = len(self.chunks)
        logger.info("BGEM3RAG: {} chunks created", self.num_chunks)

        cache_key_src = (
            f"chunk_size={RAG_V2_CHUNK_SIZE}|chunk_mode={RAG_V2_CHUNK_MODE}"
            f"|model={RAG_V2_MODEL_NAME}|docs={documents}"
        )
        doc_hash = hashlib.md5(cache_key_src.encode()).hexdigest()
        dense_path = os.path.join(cache_dir, f"bge_m3_{doc_hash}_dense.npy")
        sparse_path = os.path.join(cache_dir, f"bge_m3_{doc_hash}_sparse.pkl")

        self.doc_dense: np.ndarray
        self.doc_sparse: list[dict[int, float]] | None = None

        if os.path.exists(dense_path):
            self.doc_dense = np.load(dense_path)
            if self.doc_dense.shape[0] != self.num_chunks:
                logger.warning("BGE-M3 cache mismatch; rebuilding embeddings")
                self._build_and_cache(dense_path, sparse_path)
            else:
                if self.use_sparse and os.path.exists(sparse_path):
                    with open(sparse_path, "rb") as f:
                        self.doc_sparse = pickle.load(f)
                logger.info("Loaded BGE-M3 embeddings from cache")
        else:
            self._build_and_cache(dense_path, sparse_path)

        # Warm the model once at startup so the first real query does not pay load cost.
        model = _get_bge_model()
        model.encode(
            ["warmup"],
            batch_size=1,
            max_length=16,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        logger.info("BGE-M3 model warmed up and ready")

    def _build_and_cache(self, dense_path: str, sparse_path: str) -> None:
        model = _get_bge_model()
        t0 = time.time()
        output = model.encode(
            self.chunks,
            batch_size=RAG_V2_BATCH_SIZE,
            max_length=RAG_V2_MAX_LENGTH,
            return_dense=True,
            return_sparse=self.use_sparse,
            return_colbert_vecs=False,
        )
        dense = np.array(output["dense_vecs"], dtype=np.float32)
        dense = _l2_normalize_rows(dense)
        self.doc_dense = dense

        if self.use_sparse:
            self.doc_sparse = output.get("lexical_weights")
            with open(sparse_path, "wb") as f:
                pickle.dump(self.doc_sparse, f)

        np.save(dense_path, self.doc_dense)
        logger.info("BGE-M3 embeddings built in {:.1f}s and cached", time.time() - t0)

    def _prepare_query(self, query: str) -> str:
        processed_query = query

        if RAG_V2_ENABLE_TELUGU_PREPROCESSING:
            processor = _get_telugu_processor()
            if processor:
                preprocessed, _ = processor.preprocess_query(query)
                if preprocessed != query:
                    logger.debug("Telugu preprocessing: '{}' -> '{}'", query, preprocessed)
                processed_query = preprocessed

        query_lower = processed_query.lower()
        is_telugu_query = _is_telugu(processed_query)

        bilingual_pairs = [
            ("mango", "మామిడి"),
            ("coconut", "కొబ్బరి"),
            ("fruit fly", "పండు ఈగ"),
            ("horn beetle", "కొమ్ము పురుగు"),
            ("hornbeetle", "కొమ్ము పురుగు"),
            ("rhinoceros beetle", "కొమ్ము పురుగు"),
            ("red palm weevil", "ఎర్రముక్కు పురుగు"),
            ("red weevil", "ఎర్రముక్కు పురుగు"),
            ("sticky trap", "స్టిక్కీ ట్రాప్"),
            ("sticky traps", "స్టిక్కీ ట్రాప్స్"),
            ("sticky roll", "స్టిక్కీ రోల్"),
            ("sticky rolls", "స్టిక్కీ రోల్స్"),
            ("bucket trap", "బకెట్ ట్రాప్"),
            ("funnel trap", "ఫన్నెల్ ట్రాప్"),
            ("pheromone", "ఫెరమోన్"),
            ("lure", "లూర్"),
            ("price", "ధర"),
            ("cost", "ఖరీదు"),
            ("field life", "ఫీల్డ్ లైఫ్"),
            ("per acre", "ఎకరానికి"),
            ("how many", "ఎన్ని"),
            ("distance", "దూరం"),
            ("dimensions", "పరిమాణం"),
            ("address", "చిరునామా"),
            ("contact", "కాంటాక్ట్"),
            ("pincode", "పిన్ కోడ్"),
            ("solar light trap", "సోలార్ లైట్ ట్రాప్"),
            ("okra", "బెండ"),
            ("cotton", "ప్రత్తి"),
            ("rice", "వరి"),
            ("stem borer", "కాండం తొలిచే పురుగు"),
            ("pink bollworm", "గులాబీ కాయతొలుచు పురుగు"),
            ("neem cake", "వేప పిండి"),
            ("neemcake", "వేప పిండి"),
            ("poly house", "పాలి హౌస్"),
            ("polyhouses", "పాలి హౌస్‌లలో"),
            ("greenhouse", "గ్రీన్ హౌస్"),
            ("vegetables", "కూరగాయలు"),
            ("vegetable", "కూరగాయలు"),
            ("length", "పొడవు"),
            ("width", "వెడల్పు"),
            ("centimeters", "సెంటీమీటర్లు"),
            ("gum", "గమ్"),
            ("days", "రోజులు"),
            ("cherlapalli", "చెర్లపల్లి"),
            ("hyderabad", "హైదరాబాద్"),
            ("postal code", "పిన్ కోడ్"),
            ("plot", "ప్లాట్"),
            ("company address", "సంస్థను ఎలా సంప్రదించాలి అడ్రస్ కాంటాక్ట్ వివరాలు"),
            ("founder", "ఫౌండర్స్"),
            ("founders", "ఫౌండర్స్"),
            ("ceo", "ఫౌండర్స్"),
            ("company", "కంపెనీ"),
            ("experience", "అనుభవం"),
            ("background", "అనుభవం"),
            ("establish", "సంవత్సరాలుగా"),
            ("established", "సంవత్సరాలుగా"),
            ("manufacturing", "మ్యానుఫ్యాక్చరింగ్"),
            ("unit", "యూనిట్"),
            ("years", "సంవత్సరాలుగా"),
            ("agri ferro", "అగ్రి ఫెరో అగ్రి ఫెర్రో"),
            ("solutions", "సొల్యూషన్స్"),
            ("who founded", "ఫౌండర్స్ విష్ణు కవిత నితిన్"),
            ("how many years", "12 సంవత్సరాలుగా"),
        ]

        expansions: list[str] = []
        for english_term, telugu_term in bilingual_pairs:
            if english_term in query_lower or telugu_term in processed_query:
                if is_telugu_query:
                    expansions.append(f"{english_term} {telugu_term}")
                else:
                    expansions.append(f"{telugu_term} {english_term}")

        if any(term in query_lower for term in ("spray", "spraying", "method", "apply", "application", "how to", "use")):
            expansions.append("పిచికారీ చేయడం పద్ధతి ఎలా ఉపయోగించాలి విధానం")

        if any(term in query_lower for term in ("lifecycle", "life cycle", "process", "complete process", "control methods", "all the control methods")):
            expansions.append("జీవ చక్రం నియంత్రణ పద్ధతులు నివారణ")

        if any(term in query_lower for term in ("setup", "setup guide", "maintaining", "maintenance", "schedule")):
            expansions.append("ఎలా అమర్చుకోవాలి నిర్వహణ షెడ్యూల్")

        if any(term in query_lower for term in ("differences", "compare", "which works better", "effectiveness")):
            expansions.append("తేడాలు పోలిక ఏది మంచిది")

        if any(term in query_lower for term in ("complete", "exact", "minimum", "maximum", "distance", "count", "quantity")):
            expansions.append("ఖచ్చితమైన వివరాలు ఎన్ని ఎంత దూరం")

        if expansions:
            processed_query = f"{processed_query} {' '.join(expansions)}"

        return " ".join(processed_query.split())

    def _encode_query(self, processed_query: str) -> tuple[np.ndarray, dict[int, float] | None]:
        model = _get_bge_model()
        output = model.encode(
            [processed_query],
            batch_size=1,
            max_length=RAG_V2_QUERY_MAX_LENGTH,
            return_dense=True,
            return_sparse=self.use_sparse,
            return_colbert_vecs=False,
        )
        q_dense = _l2_normalize(np.array(output["dense_vecs"][0], dtype=np.float32))
        q_sparse = output.get("lexical_weights", [None])[0] if self.use_sparse else None
        return q_dense, q_sparse

    def _score(self, q_dense: np.ndarray, q_sparse: dict[int, float] | None, query: str = "") -> np.ndarray:
        dense_scores = self.doc_dense @ q_dense

        # Use pure dense for cross-lingual queries (English -> Telugu KB), hybrid for Telugu
        alpha = self.hybrid_alpha if _is_telugu(query) else 1.0

        if self.enable_domain_tuning and self.domain_tuner:
            alpha = self.domain_tuner.get_optimal_alpha(query, alpha, self.domain_alpha_confidence_threshold)

        if self.use_sparse and self.doc_sparse and q_sparse and alpha < 1.0:
            sparse_scores = np.array([
                _sparse_dot(q_sparse, doc_sparse) for doc_sparse in self.doc_sparse
            ])
            return alpha * dense_scores + (1.0 - alpha) * sparse_scores
            
        return dense_scores

    def _select(self, scores: np.ndarray, k: int, threshold: float, chunks_indices: list[int] | None = None) -> str:
        if self.num_chunks == 0:
            return ""

        scored = list(zip(scores.tolist(), self.chunks))
        scored.sort(reverse=True, key=lambda x: x[0])
        filtered = [(s, c) for s, c in scored if s >= threshold]

        if len(filtered) >= RAG_V2_MIN_CHUNKS:
            results = filtered[:k]
        else:
            if not filtered:
                logger.warning("No chunks above threshold {:.2f}; using top results", threshold)
            else:
                logger.warning("Only {} chunks above threshold; padding to min", len(filtered))
            results = scored[:max(k, RAG_V2_MIN_CHUNKS)]

        return "\n\n---\n\n".join(f"[Relevance: {s:.2f}]\n{c}" for s, c in results)
    
    def _select_with_reranking(self, scores: np.ndarray, k: int, threshold: float, query: str) -> str:
        """Select and optionally rerank results using BGE-reranker."""
        if self.num_chunks == 0:
            return ""
        
        # First pass: get top-k based on hybrid scores
        scored = list(zip(scores.tolist(), self.chunks, range(len(self.chunks))))
        scored.sort(reverse=True, key=lambda x: x[0])
        
        # Filter by threshold
        filtered = [(s, c, i) for s, c, i in scored if s >= threshold]
        if len(filtered) < RAG_V2_MIN_CHUNKS:
            filtered = scored[:RAG_V2_MIN_CHUNKS]
        
        # Second pass: rerank with BGE-reranker if enabled
        if self.enable_reranking and self.reranker:
            try:
                chunk_texts = [c for _, c, _ in filtered]
                initial_scores = [s for s, _, _ in filtered]
                
                # Rerank using combined scores
                reranked = self.reranker.rerank_with_scores(
                    query,
                    chunk_texts,
                    initial_scores,
                    top_k=k,
                    reranking_weight=self.reranking_weight
                )
                
                logger.debug("BGE reranking: {} -> {} results", len(filtered), len(reranked))
                return "\n\n---\n\n".join(
                    f"[Relevance: {score:.2f}]\n{chunk}" 
                    for _, chunk, score in reranked
                )
            except Exception as e:
                logger.warning("Reranking failed: {}; using original results", e)
                return self._select(scores, k, threshold)
        
        # No reranking: use original results
        return "\n\n---\n\n".join(f"[Relevance: {s:.2f}]\n{c}" for s, c, _ in filtered[:k])

    def retrieve(
        self,
        query: str,
        api_key: str | None = None,
        k: int = RAG_V2_TOP_K,
        threshold: float = RAG_V2_SIMILARITY_THRESHOLD,
    ) -> str:
        _ = api_key
        # Fully prepare the query with all Telugu expansions
        processed = self._prepare_query(preprocess_stt_query(query))
        
        t0 = time.time()
        q_dense, q_sparse = self._encode_query(processed)
        logger.info("BGE-M3 embed: {:.0f} ms", (time.time() - t0) * 1000)
        
        # Pass fully processed query to _score so hybrid_alpha triggers if Telugu expansions exist
        scores = self._score(q_dense, q_sparse, processed)
        
        # Use reranking-aware selection
        if self.enable_reranking:
            return self._select_with_reranking(scores, k, threshold, processed)
        else:
            return self._select(scores, k, threshold)

    async def retrieve_async(
        self,
        query: str,
        api_key: str | None = None,
        k: int = RAG_V2_TOP_K,
        threshold: float = RAG_V2_SIMILARITY_THRESHOLD,
    ) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.retrieve, query, api_key, k, threshold)
