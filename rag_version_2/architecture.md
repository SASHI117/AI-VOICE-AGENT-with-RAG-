# RAG Version 2 (BGE-M3 Hybrid) Architecture

This document provides a comprehensive overview of the Retrieval-Augmented Generation (RAG) architecture implemented in the `rag_version_2` directory. This system is designed for high-accuracy, cross-lingual (English/Telugu) retrieval using hybrid dense-sparse embeddings, with specific optimizations for agricultural domain queries.

## 1. System Overview

The core of the RAG v2 system is built around the **BAAI/bge-m3** model, which is a state-of-the-art multilingual model that natively supports both Dense (semantic meaning) and Sparse (lexical/keyword) embeddings. The architecture employs a **Hybrid Search** strategy, meaning it calculates similarity based on both the conceptual meaning of the query and exact keyword matches, blending them together using an alpha parameter.

### Core Components
- **`bge_m3_rag.py`**: The main execution engine. Handles model loading, query preparation, embedding generation, scoring, and hybrid retrieval.
- **`rag_config.py`**: Centralized configuration management using environment variables.
- **`chunking.py`**: Advanced subsection-aware text chunking.
- **`telugu_preprocessing.py`**: Diacritics normalization and Telugu text variant handling.
- **`domain_tuner.py`**: Dynamic adjustment of the hybrid search blend based on the topic of the query.
- **`reranker.py`**: Optional second-stage reranking using a cross-encoder model.

---

## 2. Chunking Techniques (`chunking.py`)

Chunking is crucial for RAG performance. Traditional chunking splits text arbitrarily by word count, which can sever the connection between a heading and its content. 

This system uses a **QA Subsection-Aware Chunker** (`_chunk_by_qa_sections_v2`):
1. **Regex Section Matching**: It scans for numbered sections like `1.`, `2.3`, etc., using the regex `(?ms)^[ \t]*(\d+(?:\.\d+)*\.?[ \t]*[^\n]{3,})\n`. (Note: The regex handles both `1. Topic` and `1.Topic` without spaces).
2. **Context Preservation**: If a section is small enough (under `chunk_size * 1.5`, roughly 300 words), it keeps the heading and the body together as a single chunk.
3. **Sub-chunking**: If a section is too large, it splits the body using word-count chunking (`_chunk_by_words`), but **prepends the parent heading to every split sub-chunk**. This guarantees that a chunk containing paragraph 3 of a topic still contains the title of the topic, providing vital context to the dense vector.

---

## 3. Query Preparation & Cross-Lingual Search (`bge_m3_rag.py`)

Because users often ask queries in English (e.g., "What are the founders of Agri Ferro Solutions?"), while the Knowledge Base is entirely in Telugu, standard semantic search often fails on specific proper nouns.

**The Solution (`_prepare_query`):**
The system uses a hardcoded mapping (`bilingual_pairs`) of English domain terms to their exact Telugu equivalents in the KB (e.g., `"founders" -> "ఫౌండర్స్"`, `"agri ferro" -> "అగ్రి ఫెరో"`). 
When an English query is received:
1. The system scans the query for known English terms.
2. It translates them and appends the Telugu terms to the end of the query string.
3. This forces the **Sparse (Lexical) Search** engine to trigger, hunting down the exact Telugu words in the KB, completely solving the cross-lingual proper-noun gap.

---

## 4. Retrieval Architecture, Similarities, and Indexing

Unlike massive-scale RAG systems that require complex vector databases (like Milvus or Pinecone), this system leverages an **In-Memory Exhaustive Search Architecture**. 

Because the knowledge base is highly specific and relatively small (around 131 chunks), setting up an HNSW (Hierarchical Navigable Small World) index or IVF (Inverted File) index would add unnecessary overhead without any real speed benefit. Instead, the system loads all chunk embeddings directly into NumPy arrays in RAM and performs a brute-force matrix multiplication. This is extremely fast (sub-second) for small datasets and guarantees 100% recall of the top vectors without the approximation errors common in HNSW.

### Similarity Calculations (`_score` function)

The retrieval process calculates two different similarity scores and blends them.

1. **Dense Score (`q_dense @ doc_dense`)**: 
   - Uses **Cosine Similarity**. The BGE-M3 model outputs 1024-dimensional dense vectors. Before scoring, these vectors are L2-normalized. Computing the dot product (`@`) of two L2-normalized vectors is mathematically identical to calculating their Cosine Similarity.
   - Excellent for understanding concepts (e.g., knowing that "cost" and "price" mean the same thing).
2. **Sparse Score (`_sparse_dot(q_sparse, doc_sparse)`)**: 
   - Uses **Lexical Weights (similar to BM25/TF-IDF)**. BGE-M3 generates sparse vectors representing term importance, similar to a learned TF-IDF but trained by a neural network. 
   - The similarity is calculated via an unnormalized dot product between the query's sparse weights and the document's sparse weights.
   - Excellent for exact keyword matching, names, numbers, and specific terminology.

**The Hybrid Blend**:
```python
final_score = (alpha * dense_scores) + ((1.0 - alpha) * sparse_scores)
```
- The `alpha` value determines the weight. If `alpha = 0.45`, the system relies 45% on semantic meaning and 55% on exact keyword matching.

---

## 5. Configuration Parameters (`rag_config.py`)

Key parameters controlling the RAG behavior:

*   **`RAG_V2_MODEL_NAME`** (`BAAI/bge-m3`): The HuggingFace model used for embeddings.
*   **`RAG_V2_DEVICE`** (`cpu`): Computation device. (Optimized to run on `cpu` for Azure Container Apps).
*   **`RAG_V2_HYBRID_ALPHA`** (`0.45`): The blend weight. Lower values prioritize exact keyword matches; higher values prioritize conceptual similarity.
*   **`RAG_V2_CHUNK_SIZE`** (`200`): The target word limit for document chunks.
*   **`RAG_V2_MIN_CHUNKS`** (`3`): The minimum number of chunks to return to the LLM, regardless of threshold.
*   **`RAG_V2_SIMILARITY_THRESHOLD`** (`0.10`): Minimum score required for a chunk to be considered relevant.

---

## 6. Advanced Enhancements

### A. Domain Tuner (`domain_tuner.py`)
*(Configurable via `RAG_V2_ENABLE_DOMAIN_TUNING`)*
Not all queries need the same Hybrid Alpha. A query about "Pest Control" might need exact matching (`alpha=0.40`) for insect names, while a query about "Seasons" might need semantic reasoning (`alpha=0.55`). The Domain Tuner scans the query for trigger words and dynamically adjusts the alpha value on the fly.

### B. Telugu Preprocessing (`telugu_preprocessing.py`)
*(Configurable via `RAG_V2_ENABLE_TELUGU_PREPROCESSING`)*
Handles diacritic normalization (e.g., standardizing different typing styles of Chandrabindu) and phonetic spelling variations in Telugu (e.g., mapping `ఫిరమోన్` and `ఫెర్మోన్` to `ఫెరమోన్`), ensuring lexical searches don't fail due to minor typos in the KB.

### C. Reranker (`reranker.py`)
*(Configurable via `RAG_V2_ENABLE_RERANKING`)*

The system **does** have full support for a second-stage reranker (`BAAI/bge-reranker-v2-m3`), but it is **turned off by default** (`RAG_V2_ENABLE_RERANKING = False`) to maintain sub-second latency targets on CPU instances.

**How Reranking works when enabled:**
Standard vector search (Dense/Sparse) uses a "bi-encoder" architecture: the query and chunks are encoded separately, and their vectors are compared. While fast, this misses complex relationships. 
The Reranker uses a "cross-encoder" architecture. It takes the top `k` chunks retrieved by the initial hybrid search and feeds both the `query` and the `chunk text` into the neural network *simultaneously* (`inputs = self.tokenizer([[query, doc1], ...])`). The model outputs a raw probability (Sigmoid score) of how relevant the document is to the query.
- It then blends the cross-encoder score with the initial retrieval score using `RAG_V2_RERANKING_WEIGHT` (e.g., 60% reranker, 40% initial score).
- While this guarantees a massive boost in precision, it adds significant latency (often 2-3 seconds on a CPU) because running a cross-encoder requires full neural network inference for every chunk evaluated.

---
## Summary of Search Flow
1. User asks question.
2. `bge_m3_rag.py` pre-processes the text, appending Telugu keyword translations.
3. Query is passed to `BAAI/bge-m3` model to generate a Dense Vector and a Sparse Vector dictionary.
4. Dot-products are calculated against the cached KB arrays (`.npy` and `.pkl`).
5. Dense and Sparse arrays are blended using the Hybrid Alpha (`0.45`).
6. The top `k` scoring chunks are retrieved, formatted, and returned to the LLM context window.
