"""
BGE-Reranker-v2-m3 integration for second-stage ranking of retrieval results.
Helps filter and re-rank retrieved chunks for better relevance.
"""

from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BGEReranker:
    """
    Second-stage reranking using BAAI/bge-reranker-v2-m3.
    Filters and re-ranks initial retrieval results for higher precision.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"):
        """
        Initialize the BGE reranker.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the reranker model from HuggingFace."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            logger.info(f"Loading BGE reranker from {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"BGE reranker loaded on {self.device}")
        except ImportError as e:
            logger.error(f"Failed to load reranker: {e}. Ensure transformers is installed.")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 6,
        threshold: float = 0.0,
        batch_size: int = 32
    ) -> List[Tuple[int, str, float]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            documents: List of document chunks to rerank
            top_k: Return top k results (default: 6)
            threshold: Minimum relevance score to include (default: 0.0)
            batch_size: Batch size for inference (default: 32)
            
        Returns:
            List of (original_index, document, score) tuples, sorted by score descending
        """
        if not documents:
            return []
        
        import torch
        
        results = []
        
        # Process in batches for efficiency
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            
            # Prepare inputs: [query, document1], [query, document2], ...
            pairs = [[query, doc] for doc in batch_docs]
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                ).to(self.device)
                
                outputs = self.model(**inputs)
                scores = outputs.logits.view(-1).float()
                
                # Convert scores to sigmoid probabilities (normalized 0-1)
                scores = torch.sigmoid(scores).cpu().tolist()
            
            # Store results with original indices
            for j, (doc, score) in enumerate(zip(batch_docs, scores)):
                original_idx = i + j
                if score >= threshold:
                    results.append((original_idx, doc, score))
        
        # Sort by score descending and return top_k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
    
    def rerank_with_scores(
        self,
        query: str,
        documents: List[str],
        initial_scores: List[float],
        top_k: int = 6,
        reranking_weight: float = 0.6
    ) -> List[Tuple[int, str, float]]:
        """
        Rerank documents using both initial scores and reranker scores.
        Combines initial retrieval scores with reranker scores for better results.
        
        Args:
            query: Search query
            documents: List of document chunks
            initial_scores: Initial retrieval scores for documents
            top_k: Return top k results
            reranking_weight: Weight for reranker score (0.0-1.0), rest goes to initial score
            
        Returns:
            List of (original_index, document, combined_score) tuples
        """
        if not documents:
            return []
        
        import torch
        
        # Get reranker scores
        pairs = [[query, doc] for doc in documents]
        
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(self.device)
            
            outputs = self.model(**inputs)
            reranker_scores = torch.sigmoid(outputs.logits.view(-1)).cpu().tolist()
        
        # Normalize initial scores to 0-1 range
        if initial_scores:
            max_init = max(initial_scores) if max(initial_scores) > 0 else 1
            min_init = min(initial_scores) if len(initial_scores) > 1 else 0
            range_init = max_init - min_init if max_init > min_init else 1
            
            normalized_initial = [
                (s - min_init) / range_init if range_init > 0 else s
                for s in initial_scores
            ]
        else:
            normalized_initial = [0.5] * len(documents)
        
        # Combine scores
        results = []
        for idx, (doc, init_score, rerank_score) in enumerate(
            zip(documents, normalized_initial, reranker_scores)
        ):
            combined_score = (
                reranking_weight * rerank_score +
                (1 - reranking_weight) * init_score
            )
            results.append((idx, doc, combined_score))
        
        # Sort by combined score and return top_k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
    
    def __del__(self):
        """Cleanup: move model to CPU before deletion."""
        if self.model is not None:
            import torch
            self.model.to('cpu')
            torch.cuda.empty_cache()
