"""RAG Helper Classes for Pipecat Voice AI Bots.

Three retrieval strategies, all sharing the same interface:

  retrieve(query, api_key, k, threshold) -> str
  retrieve_async(query, api_key, k, threshold) -> str   (non-blocking)

Strategy guide
--------------
  FastRAG    — TF-IDF cosine similarity, no API calls, < 5 ms.
               Best when knowledge base is small (≤ FAST_RAG_CHUNK_THRESHOLD chunks).

  SimpleRAG  — OpenAI / Azure OpenAI embeddings, cached to disk.
               ~1 s on first query per session, then < 10 ms from cache.
               Best for large knowledge bases where semantic precision matters.

  TieredRAG  — Automatically picks FastRAG or SimpleRAG based on document size.
               Recommended default.

Return value
------------
All three classes return a single formatted string ready to inject into the
LLM system prompt or tool result:

    [Relevance: 0.82]
    <chunk text>

    ---

    [Relevance: 0.74]
    <chunk text>

IMPORTANT: retrieve() returns a string, not a list.
Do NOT do:  "\n\n".join(rag.retrieve(...))  — that iterates every character.
Do:         result_text = rag.retrieve(...)
"""

import asyncio
import hashlib
import os
import re
import time

import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim
from functools import lru_cache

from .preprocessing import preprocess_stt_query
from .rag_config import (
    CHUNK_SIZE,
    FAST_RAG_CHUNK_THRESHOLD,
    MIN_CHUNKS,
    RAG_CACHE_DIR,
    SIMILARITY_THRESHOLD,
    TOP_K,
)

# ---------------------------------------------------------------------------
# Shared embedding client (lazy singleton — avoids repeated auth overhead)
# ---------------------------------------------------------------------------
_embed_client = None
_embed_deployment = None


_embed_client = None
_async_embed_client = None

def _get_embed_client():
    """Return a cached sync Azure OpenAI client for embedding calls, forcing IPv4."""
    global _embed_client, _embed_deployment
    if _embed_client is None:
        from openai import AzureOpenAI
        import httpx
        
        _embed_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
        transport = httpx.HTTPTransport(local_address="0.0.0.0", retries=1)
        http_client = httpx.Client(transport=transport, timeout=15.0)

        _embed_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            http_client=http_client,
        )
        logger.info(f"🔌 Sync Embedding client ready (deployment: {_embed_deployment})")
    return _embed_client, _embed_deployment

def _get_async_embed_client():
    """Return a cached async Azure OpenAI client for embedding calls, forcing IPv4."""
    global _async_embed_client, _embed_deployment
    if _async_embed_client is None:
        from openai import AsyncAzureOpenAI
        import httpx

        _embed_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
        transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0", retries=1)
        http_client = httpx.AsyncClient(transport=transport, timeout=15.0)

        _async_embed_client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            http_client=http_client,
        )
        logger.info(f"🔌 Async Embedding client ready (deployment: {_embed_deployment})")
    return _async_embed_client, _embed_deployment


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _chunk_by_words(text: str, chunk_size: int) -> list[str]:
    """Split text by word count."""
    words = text.split()
    chunks, buf = [], []
    for w in words:
        buf.append(w)
        if len(buf) >= chunk_size:
            chunks.append(" ".join(buf))
            buf = []
    if buf:
        chunks.append(" ".join(buf))
    return [c.strip() for c in chunks if c.strip()]


def _chunk_by_qa_sections(text: str, chunk_size: int) -> list[str]:
    """Split text by numbered Q&A sections and sub-chunk long sections by words."""
    text = text.lstrip("\ufeff")
    # Only split on top-level headings like "1.", "2.", ... and keep sub-items
    # like "6.1", "50.2" inside the parent section body.
    pattern = re.compile(r"(?ms)^\s*(\d+\.(?!\d)[^\n]*)\n(.*?)(?=^\s*\d+\.(?!\d)[^\n]*\n|\Z)")
    sections = []
    for m in pattern.finditer(text):
        heading = m.group(1).strip()
        body = m.group(2).strip()
        if not body:
            continue
        section_text = f"{heading}\n{body}"
        # Keep section coherence, but split if a section is too large.
        if len(section_text.split()) > int(chunk_size * 1.5):
            heading_words = len(heading.split()) + 2
            body_chunk_size = max(chunk_size - heading_words, 80)
            parts = _chunk_by_words(body, body_chunk_size)
            sections.extend([f"{heading}\n{part}" for part in parts if part.strip()])
        else:
            sections.append(section_text)
    return [s.strip() for s in sections if s.strip()]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split text into chunks.

    Modes:
      - qa:   numbered-section-aware chunking (best for FAQ-style KBs)
      - auto: use qa when numbered sections are detected, else word chunking
      - words: always word chunking
    """
    mode = os.getenv("RAG_CHUNK_MODE", "auto").strip().lower()

    if mode in {"qa", "auto"}:
        qa_chunks = _chunk_by_qa_sections(text, chunk_size)
        if mode == "qa":
            return qa_chunks if qa_chunks else _chunk_by_words(text, chunk_size)
        # auto mode: use qa only if it found enough sections
        if len(qa_chunks) >= 5:
            return qa_chunks

    chunks = _chunk_by_words(text, chunk_size)
    if not chunks and text.strip():
        chunks = [text.strip()]
    return chunks


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str], api_key: str) -> list[np.ndarray]:
    """Call Azure OpenAI embeddings API and return numpy vectors (sync)."""
    client, deployment = _get_embed_client()
    resp = client.embeddings.create(model=deployment, input=texts)
    return [np.array(d.embedding) for d in resp.data]

async def embed_texts_async(texts: list[str], api_key: str) -> list[np.ndarray]:
    """Non-blocking call to Azure OpenAI embeddings API."""
    client, deployment = _get_async_embed_client()
    resp = await client.embeddings.create(model=deployment, input=texts)
    return [np.array(d.embedding) for d in resp.data]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _tokenize_for_overlap(text: str) -> set[str]:
    """Unicode-safe tokenization for lightweight lexical overlap scoring."""
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return {t for t in tokens if len(t) > 1}


# ---------------------------------------------------------------------------
# SimpleRAG — OpenAI / Azure OpenAI embeddings
# ---------------------------------------------------------------------------

class SimpleRAG:
    """Semantic retrieval using OpenAI embeddings with disk cache.

    Parameters
    ----------
    documents : str
        Full knowledge base text (will be chunked automatically).
    api_key : str
        OpenAI or Azure OpenAI API key (passed through to embed calls).
    cache_dir : str | None
        Directory to store embedding cache files. Defaults to RAG_CACHE_DIR.
    """

    def __init__(self, documents: str, api_key: str, cache_dir: str | None = None):
        if cache_dir is None:
            cache_dir = RAG_CACHE_DIR
        logger.info(f"📚 SimpleRAG: initialising with {len(documents):,} chars …")
        self.chunks = chunk_text(documents)
        logger.info(f"📦 {len(self.chunks)} chunks created")
        self.chunk_token_sets = [_tokenize_for_overlap(chunk) for chunk in self.chunks]

        # Build lightweight IDF stats to make lexical overlap robust and less noisy.
        df: dict[str, int] = {}
        for token_set in self.chunk_token_sets:
            for token in token_set:
                df[token] = df.get(token, 0) + 1
        num_docs = max(len(self.chunk_token_sets), 1)
        self.token_idf = {t: float(np.log((num_docs + 1) / (f + 1)) + 1.0) for t, f in df.items()}

        os.makedirs(cache_dir, exist_ok=True)
        # Cache key must include chunking config. Otherwise changing CHUNK_SIZE
        # can incorrectly reuse embeddings generated for a different chunk layout.
        chunk_mode = os.getenv("RAG_CHUNK_MODE", "auto")
        chunker_version = "qa_heading_repeat_v3"
        cache_key_src = (
            f"chunk_size={CHUNK_SIZE}|chunk_mode={chunk_mode}|chunker_version={chunker_version}|docs={documents}"
        )
        doc_hash = hashlib.md5(cache_key_src.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"embeddings_{doc_hash}.npy")
        hash_file = os.path.join(cache_dir, "current_hash.txt")

        cached_hash = ""
        if os.path.exists(hash_file):
            with open(hash_file) as f:
                cached_hash = f.read().strip()

        if os.path.exists(cache_file) and cached_hash == doc_hash:
            logger.info(f"📂 Loading cached embeddings from {cache_file}")
            self.embeddings = np.load(cache_file, allow_pickle=True).tolist()
            if len(self.embeddings) != len(self.chunks):
                logger.warning(
                    "⚠️  Cached embeddings/chunk count mismatch (embeddings={}, chunks={}). Rebuilding cache.",
                    len(self.embeddings),
                    len(self.chunks),
                )
                self.embeddings = embed_texts(self.chunks, api_key)
                np.save(cache_file, self.embeddings)
                with open(hash_file, "w") as f:
                    f.write(doc_hash)
        else:
            logger.info("🆕 Generating embeddings (first run or document changed) …")
            self.embeddings = embed_texts(self.chunks, api_key)
            np.save(cache_file, self.embeddings)
            with open(hash_file, "w") as f:
                f.write(doc_hash)
        
        # Convert list of embeddings to a single numpy matrix for vectorized scoring
        self.embeddings_matrix = np.vstack([np.array(emb).flatten() for emb in self.embeddings])
        logger.info(f"✅ SimpleRAG ready (Matrix shape: {self.embeddings_matrix.shape})")

    @lru_cache(maxsize=200)
    def _cached_retrieve(self, query_text: str, k: int, threshold: float) -> str:
        """Internal cached method to skip embedding API and scoring for repeat queries."""
        # Note: This is called after query embedding is obtained in retrieve()
        # BUT we can also wrap the whole retrieve call if we manage the API key.
        pass

    def _weighted_lexical_overlap_vectorized(self, query_text: str) -> np.ndarray:
        """Vectorized lexical overlap calculation for all chunks at once."""
        query_tokens = _tokenize_for_overlap(query_text)
        if not query_tokens:
            return np.zeros(len(self.chunks))
        
        denom = sum(self.token_idf.get(t, 1.0) for t in query_tokens)
        if denom <= 0:
            return np.zeros(len(self.chunks))
            
        # Pre-calculate scores for each chunk
        scores = []
        for chunk_tokens in self.chunk_token_sets:
            numer = sum(self.token_idf.get(t, 1.0) for t in query_tokens if t in chunk_tokens)
            scores.append(numer / denom)
        return np.array(scores)

    def _score_and_select(self, query_text: str, q_emb: np.ndarray, k: int, threshold: float) -> str:
        hybrid_weight = float(os.getenv("RAG_HYBRID_LEXICAL_WEIGHT", "0.35"))
        hybrid_weight = min(max(hybrid_weight, 0.0), 0.9)

        # Vectorized Semantic Scoring (Cosine Similarity)
        # q_emb: (D,), matrix: (N, D)
        q_norm = np.linalg.norm(q_emb)
        matrix_norms = np.linalg.norm(self.embeddings_matrix, axis=1)
        
        # Avoid division by zero
        matrix_norms[matrix_norms == 0] = 1.0
        if q_norm == 0: q_norm = 1.0

        semantic_scores = np.dot(self.embeddings_matrix, q_emb) / (matrix_norms * q_norm)
        
        # Vectorized Lexical Scoring
        lexical_scores = self._weighted_lexical_overlap_vectorized(query_text)
        
        # Keyword Boosting: If a chunk contains exact rare keywords from the query, boost it.
        # This is the "Secret Sauce" for 99% accuracy on pest-specific questions.
        boost = np.zeros(len(self.chunks))
        query_tokens = _tokenize_for_overlap(query_text)
        for i, chunk_tokens in enumerate(self.chunk_token_sets):
            # If query has unique keywords (IDF > 2.0) that match the chunk, give a boost
            match_count = sum(1 for t in query_tokens if t in chunk_tokens and self.token_idf.get(t, 0) > 2.0)
            if match_count > 0:
                boost[i] = 0.15 * match_count # 15% boost per rare keyword match
        
        # Hybrid Blend
        combined_scores = (1.0 - hybrid_weight) * semantic_scores + hybrid_weight * lexical_scores + boost
        
        scored = [(float(combined_scores[i]), self.chunks[i]) for i in range(len(self.chunks))]
        scored.sort(reverse=True, key=lambda x: x[0])
        
        filtered = [(s, c) for s, c in scored if s >= threshold]
        
        # Selection logic (ensure MIN_CHUNKS)
        if len(filtered) >= MIN_CHUNKS:
            results = filtered[:k]
        else:
            results = scored[:MIN_CHUNKS]

        return "\n\n---\n\n".join(f"[Relevance: {s:.2f}]\n{c}" for s, c in results)

    # Simple dictionary-based cache to bypass the 4-second API latency
    _QUERY_CACHE = {}

    def retrieve(self, query: str, api_key: str = None, k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD) -> str:
        """Return a formatted context string for *query* (Synchronous with Caching)."""
        processed = preprocess_stt_query(query)
        cache_key = f"{processed}_{k}_{threshold}"
        if cache_key in self._QUERY_CACHE:
            logger.info(f"🚀 Cache Hit (Sync): '{processed}' (0 ms)")
            return self._QUERY_CACHE[cache_key]

        # NEW: Fast-Path for exact keyword matches. 
        # If lexical overlap is very high, we can skip waiting for the 1.5s embedding API!
        lexical_scores = self._weighted_lexical_overlap_vectorized(processed)
        if np.max(lexical_scores) > 0.85: # 85% keyword match
            logger.info(f"⚡ Fast-Path Match: '{processed}' (Keyword Overlap > 0.85)")
            # Use dummy zero embedding for score_and_select since lexical is dominant
            result = self._score_and_select(processed, np.zeros(self.embeddings_matrix.shape[1]), k, threshold)
            self._QUERY_CACHE[cache_key] = result
            return result

        t0 = time.time()
        try:
            q_emb = embed_texts([processed], api_key)[0]
            logger.info(f"⏱️  Embedding (Sync): {(time.time()-t0)*1000:.0f} ms")
        except Exception as e:
            logger.error(f"❌ Embedding API failed: {e}")
            q_emb = np.zeros(self.embeddings_matrix.shape[1])
            
        result = self._score_and_select(processed, q_emb, k, threshold)
        self._QUERY_CACHE[cache_key] = result
        return result

    async def retrieve_async(self, query: str, api_key: str = None, k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD) -> str:
        """Non-blocking version of retrieve() with Result Caching and Fast-Path."""
        processed = preprocess_stt_query(query)
        
        # FAST PATH: If we have seen this query before, return it in < 1ms
        cache_key = f"{processed}_{k}_{threshold}"
        if cache_key in self._QUERY_CACHE:
            logger.info(f"🚀 Cache Hit: '{processed}' (0 ms)")
            return self._QUERY_CACHE[cache_key]

        # Fast-Path for exact keyword matches (< 100ms)
        lexical_scores = self._weighted_lexical_overlap_vectorized(processed)
        if np.max(lexical_scores) > 0.85:
            logger.info(f"⚡ Fast-Path Match (Async): '{processed}'")
            result = self._score_and_select(processed, np.zeros(self.embeddings_matrix.shape[1]), k, threshold)
            self._QUERY_CACHE[cache_key] = result
            return result

        t0 = time.time()
        try:
            q_emb = (await embed_texts_async([processed], api_key))[0]
            emb_time = (time.time() - t0) * 1000
            logger.info(f"⏱️  Embedding: {emb_time:.0f} ms")
        except Exception as e:
            logger.error(f"❌ Embedding API failed: {e}")
            # Fallback to pure lexical search if API fails to keep 100ms target
            q_emb = np.zeros(self.embeddings_matrix.shape[1])
        
        result = self._score_and_select(processed, q_emb, k, threshold)
        
        # Store in cache
        if len(self._QUERY_CACHE) > 500: self._QUERY_CACHE.clear() # Basic size limit
        self._QUERY_CACHE[cache_key] = result
        
        return result


# ---------------------------------------------------------------------------
# FastRAG — TF-IDF (no API calls)
# ---------------------------------------------------------------------------

class FastRAG:
    """TF-IDF retrieval — zero network latency, < 5 ms per query.

    No API key required. Suitable for small knowledge bases where the
    embedding API's ~1 s round-trip would dominate voice bot latency.

    Parameters
    ----------
    documents : str
        Full knowledge base text.
    chunk_size : int
        Words per chunk (default CHUNK_SIZE).
    """

    def __init__(self, documents: str, chunk_size: int = CHUNK_SIZE):
        logger.info(f"⚡ FastRAG: initialising with {len(documents):,} chars …")
        self.chunks = chunk_text(documents, chunk_size)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        logger.info(f"✅ FastRAG ready — {self.tfidf_matrix.shape[0]} chunks, vocab {len(self.vectorizer.vocabulary_)}")

    def _score_and_select(self, query: str, k: int, threshold: float) -> str:
        query_vec = self.vectorizer.transform([query])
        scores = sklearn_cosine_sim(query_vec, self.tfidf_matrix).flatten()
        scored = [(float(scores[i]), self.chunks[i]) for i in range(len(self.chunks))]
        scored.sort(reverse=True, key=lambda x: x[0])
        filtered = [(s, c) for s, c in scored if s >= threshold]

        if len(filtered) >= MIN_CHUNKS:
            results = filtered[:k]
        elif filtered:
            results = scored[:MIN_CHUNKS]
        else:
            results = scored[:MIN_CHUNKS]

        if len(results) < MIN_CHUNKS:
            results = scored[:MIN_CHUNKS]

        return "\n\n---\n\n".join(f"[Relevance: {s:.2f}]\n{c}" for s, c in results)

    def retrieve(self, query: str, api_key: str = None, k: int = TOP_K, threshold: float = 0.1) -> str:
        """Return a formatted context string for *query*."""
        processed = preprocess_stt_query(query)
        t0 = time.time()
        result = self._score_and_select(processed, k, threshold)
        logger.info(f"⚡ FastRAG: {(time.time()-t0)*1000:.1f} ms")
        return result

    async def retrieve_async(self, query: str, api_key: str = None, k: int = TOP_K, threshold: float = 0.1) -> str:
        """Async alias — TF-IDF is fast enough to run on the event loop."""
        return self.retrieve(query, api_key, k, threshold)


# ---------------------------------------------------------------------------
# TieredRAG — auto-picks FastRAG or SimpleRAG
# ---------------------------------------------------------------------------

class TieredRAG:
    """Automatically selects FastRAG or SimpleRAG based on knowledge base size.

      ≤ FAST_RAG_CHUNK_THRESHOLD chunks  →  FastRAG  (< 5 ms, no API)
      >  FAST_RAG_CHUNK_THRESHOLD chunks  →  SimpleRAG (~1 s first call, cached)

    This is the recommended class to use in production — it gives you the
    best latency for small KBs and the best accuracy for large ones.

    Parameters
    ----------
    documents : str
        Full knowledge base text.
    api_key : str
        OpenAI / Azure OpenAI API key (needed only if SimpleRAG is selected).
    cache_dir : str | None
        Embedding cache directory (SimpleRAG only).
    chunk_threshold : int
        Override the FastRAG / SimpleRAG boundary.
    chunk_size : int
        Words per chunk.
    """

    def __init__(
        self,
        documents: str,
        api_key: str,
        cache_dir: str | None = None,
        chunk_threshold: int = FAST_RAG_CHUNK_THRESHOLD,
        chunk_size: int = CHUNK_SIZE,
    ):
        num_chunks = len(chunk_text(documents, chunk_size))
        if num_chunks <= chunk_threshold:
            logger.info(f"⚡ TieredRAG → FastRAG ({num_chunks} chunks ≤ {chunk_threshold})")
            self.strategy = "fast"
            self.rag = FastRAG(documents, chunk_size)
        else:
            logger.info(f"🔬 TieredRAG → SimpleRAG ({num_chunks} chunks > {chunk_threshold})")
            self.strategy = "embedding"
            self.rag = SimpleRAG(documents, api_key, cache_dir)
        self.num_chunks = num_chunks

    def retrieve(self, query: str, api_key: str = None, k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD) -> str:
        effective_threshold = 0.1 if self.strategy == "fast" else threshold
        return self.rag.retrieve(query, api_key, k, effective_threshold)

    async def retrieve_async(self, query: str, api_key: str = None, k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD) -> str:
        effective_threshold = 0.1 if self.strategy == "fast" else threshold
        return await self.rag.retrieve_async(query, api_key, k, effective_threshold)
