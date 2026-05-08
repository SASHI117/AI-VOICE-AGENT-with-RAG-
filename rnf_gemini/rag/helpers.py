"""RNF-local RAG helper classes for Pipecat Voice AI bots."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time

import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim

from .preprocessing import preprocess_stt_query
from .rag_config import (
    CHUNK_SIZE,
    FAST_RAG_CHUNK_THRESHOLD,
    MIN_CHUNKS,
    RAG_CACHE_DIR,
    SIMILARITY_THRESHOLD,
    TOP_K,
)

_embed_client = None
_embed_deployment = None


def _get_embed_client():
    global _embed_client, _embed_deployment
    if _embed_client is None:
        from openai import AzureOpenAI

        _embed_deployment = (
            os.getenv("RNF_AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            or "text-embedding-3-large"
        )
        _embed_client = AzureOpenAI(
            api_key=os.getenv("RNF_AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=(
                os.getenv("RNF_AZURE_OPENAI_API_VERSION")
                or os.getenv("AZURE_OPENAI_API_VERSION")
                or "2024-12-01-preview"
            ),
            azure_endpoint=os.getenv("RNF_AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        logger.info("Embedding client ready (deployment: {})", _embed_deployment)
    return _embed_client, _embed_deployment


def _chunk_by_words(text: str, chunk_size: int) -> list[str]:
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
    text = text.lstrip("\ufeff")
    pattern = re.compile(r"(?ms)^\s*(\d+\.(?!\d)[^\n]*)\n(.*?)(?=^\s*\d+\.(?!\d)[^\n]*\n|\Z)")
    sections = []
    for m in pattern.finditer(text):
        heading = m.group(1).strip()
        body = m.group(2).strip()
        if not body:
            continue
        section_text = f"{heading}\n{body}"
        if len(section_text.split()) > int(chunk_size * 1.5):
            heading_words = len(heading.split()) + 2
            body_chunk_size = max(chunk_size - heading_words, 80)
            parts = _chunk_by_words(body, body_chunk_size)
            sections.extend([f"{heading}\n{part}" for part in parts if part.strip()])
        else:
            sections.append(section_text)
    return [s.strip() for s in sections if s.strip()]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    mode = (os.getenv("RNF_RAG_CHUNK_MODE") or os.getenv("RAG_CHUNK_MODE") or "auto").strip().lower()

    if mode in {"qa", "auto"}:
        qa_chunks = _chunk_by_qa_sections(text, chunk_size)
        if mode == "qa":
            return qa_chunks if qa_chunks else _chunk_by_words(text, chunk_size)
        if len(qa_chunks) >= 5:
            return qa_chunks

    chunks = _chunk_by_words(text, chunk_size)
    if not chunks and text.strip():
        chunks = [text.strip()]
    return chunks


def embed_texts(texts: list[str], api_key: str) -> list[np.ndarray]:
    client, deployment = _get_embed_client()
    resp = client.embeddings.create(model=deployment, input=texts)
    return [np.array(d.embedding) for d in resp.data]


async def embed_texts_async(texts: list[str], api_key: str) -> list[np.ndarray]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_texts, texts, api_key)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _tokenize_for_overlap(text: str) -> set[str]:
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return {t for t in tokens if len(t) > 1}


class SimpleRAG:
    def __init__(self, documents: str, api_key: str, cache_dir: str | None = None):
        if cache_dir is None:
            cache_dir = RAG_CACHE_DIR
        logger.info("SimpleRAG: initialising with {:,} chars", len(documents))
        self.chunks = chunk_text(documents)
        logger.info("{} chunks created", len(self.chunks))
        self.chunk_token_sets = [_tokenize_for_overlap(chunk) for chunk in self.chunks]

        df: dict[str, int] = {}
        for token_set in self.chunk_token_sets:
            for token in token_set:
                df[token] = df.get(token, 0) + 1
        num_docs = max(len(self.chunk_token_sets), 1)
        self.token_idf = {t: float(np.log((num_docs + 1) / (f + 1)) + 1.0) for t, f in df.items()}

        os.makedirs(cache_dir, exist_ok=True)
        chunk_mode = (os.getenv("RNF_RAG_CHUNK_MODE") or os.getenv("RAG_CHUNK_MODE") or "auto")
        chunker_version = "rnf_qa_heading_repeat_v1"
        cache_key_src = (
            f"chunk_size={CHUNK_SIZE}|chunk_mode={chunk_mode}|chunker_version={chunker_version}|docs={documents}"
        )
        doc_hash = hashlib.md5(cache_key_src.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"embeddings_{doc_hash}.npy")
        hash_file = os.path.join(cache_dir, "current_hash.txt")

        cached_hash = ""
        if os.path.exists(hash_file):
            with open(hash_file, encoding="utf-8") as f:
                cached_hash = f.read().strip()

        if os.path.exists(cache_file) and cached_hash == doc_hash:
            logger.info("Loading cached embeddings from {}", cache_file)
            self.embeddings = np.load(cache_file, allow_pickle=True).tolist()
            if len(self.embeddings) != len(self.chunks):
                logger.warning(
                    "Cached embeddings/chunk mismatch (embeddings={}, chunks={}). Rebuilding cache.",
                    len(self.embeddings),
                    len(self.chunks),
                )
                self.embeddings = embed_texts(self.chunks, api_key)
                np.save(cache_file, self.embeddings)
                with open(hash_file, "w", encoding="utf-8") as f:
                    f.write(doc_hash)
        else:
            logger.info("Generating embeddings (first run or document changed)")
            self.embeddings = embed_texts(self.chunks, api_key)
            np.save(cache_file, self.embeddings)
            with open(hash_file, "w", encoding="utf-8") as f:
                f.write(doc_hash)
        logger.info("SimpleRAG ready")

    def _weighted_lexical_overlap(self, query_text: str, chunk_tokens: set[str]) -> float:
        query_tokens = _tokenize_for_overlap(query_text)
        if not query_tokens:
            return 0.0
        denom = sum(self.token_idf.get(t, 1.0) for t in query_tokens)
        if denom <= 0:
            return 0.0
        numer = sum(self.token_idf.get(t, 1.0) for t in query_tokens if t in chunk_tokens)
        return numer / denom

    def _score_and_select(self, query_text: str, q_emb: np.ndarray, k: int, threshold: float) -> str:
        hybrid_weight = float(
            os.getenv("RNF_RAG_HYBRID_LEXICAL_WEIGHT")
            or os.getenv("RAG_HYBRID_LEXICAL_WEIGHT")
            or "0.35"
        )
        hybrid_weight = min(max(hybrid_weight, 0.0), 0.9)

        scored = []
        for emb, chunk, chunk_tokens in zip(self.embeddings, self.chunks, self.chunk_token_sets):
            semantic = cosine_sim(q_emb, emb)
            lexical = self._weighted_lexical_overlap(query_text, chunk_tokens)
            score = (1.0 - hybrid_weight) * semantic + hybrid_weight * lexical
            scored.append((score, chunk))

        scored.sort(reverse=True, key=lambda x: x[0])
        filtered = [(s, c) for s, c in scored if s >= threshold]

        if len(filtered) >= MIN_CHUNKS:
            results = filtered[:k]
        elif filtered:
            logger.warning("Only {} chunks above threshold; padding to {}", len(filtered), MIN_CHUNKS)
            results = scored[:MIN_CHUNKS]
        else:
            logger.warning("No chunks above threshold {:.2f}; using top {}", threshold, MIN_CHUNKS)
            results = scored[:MIN_CHUNKS]

        if len(results) < MIN_CHUNKS:
            results = scored[:MIN_CHUNKS]

        return "\n\n---\n\n".join(f"[Relevance: {s:.2f}]\n{c}" for s, c in results)

    def retrieve(self, query: str, api_key: str = None, k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD) -> str:
        processed = preprocess_stt_query(query)
        t0 = time.time()
        q_emb = embed_texts([processed], api_key)[0]
        logger.info("Embedding: {:.0f} ms", (time.time() - t0) * 1000)
        return self._score_and_select(processed, q_emb, k, threshold)

    async def retrieve_async(self, query: str, api_key: str = None, k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD) -> str:
        processed = preprocess_stt_query(query)
        t0 = time.time()
        q_emb = (await embed_texts_async([processed], api_key))[0]
        logger.info("Embedding: {:.0f} ms", (time.time() - t0) * 1000)
        return self._score_and_select(processed, q_emb, k, threshold)


class FastRAG:
    def __init__(self, documents: str, chunk_size: int = CHUNK_SIZE):
        logger.info("FastRAG: initialising with {:,} chars", len(documents))
        self.chunks = chunk_text(documents, chunk_size)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        logger.info(
            "FastRAG ready: {} chunks, vocab {}",
            self.tfidf_matrix.shape[0],
            len(self.vectorizer.vocabulary_),
        )

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
        processed = preprocess_stt_query(query)
        t0 = time.time()
        result = self._score_and_select(processed, k, threshold)
        logger.info("FastRAG: {:.1f} ms", (time.time() - t0) * 1000)
        return result

    async def retrieve_async(self, query: str, api_key: str = None, k: int = TOP_K, threshold: float = 0.1) -> str:
        return self.retrieve(query, api_key, k, threshold)


class TieredRAG:
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
            logger.info("TieredRAG -> FastRAG ({} chunks <= {})", num_chunks, chunk_threshold)
            self.strategy = "fast"
            self.rag = FastRAG(documents, chunk_size)
        else:
            logger.info("TieredRAG -> SimpleRAG ({} chunks > {})", num_chunks, chunk_threshold)
            self.strategy = "embedding"
            self.rag = SimpleRAG(documents, api_key, cache_dir)
        self.num_chunks = num_chunks

    def retrieve(self, query: str, api_key: str = None, k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD) -> str:
        effective_threshold = 0.1 if self.strategy == "fast" else threshold
        return self.rag.retrieve(query, api_key, k, effective_threshold)

    async def retrieve_async(self, query: str, api_key: str = None, k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD) -> str:
        effective_threshold = 0.1 if self.strategy == "fast" else threshold
        return await self.rag.retrieve_async(query, api_key, k, effective_threshold)
