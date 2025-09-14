"""Hybrid retriever combining BM25 and dense embeddings for window-based retrieval.

This module implements the end-to-end retrieval workflow:
1. HybridRetriever.search() - retrieves parent windows using BM25 + FAISS
2. expand_windows() - expands to child paragraphs with deduplication

The system uses window-based indexing where:
- Parent windows contain multiple paragraphs with overlap
- Child paragraphs are the actual content units
- Hybrid retrieval combines lexical (BM25) and semantic (FAISS) search
"""

import json
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

import numpy as np
from nltk.stem import WordNetLemmatizer

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except ImportError:
    BM25Okapi = None

def light_lemmatize(query: str, lemmatizer: WordNetLemmatizer) -> List[str]:
    """Lightweight lemmatization for BM25 search - ~10x faster than POS-aware."""
    # Convert to lowercase and split into words
    text = query.lower()
    words = re.findall(r'\b\w+\b', text)
    # Simple lemmatization without POS tagging
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return lemmatized_words

def calculate_adaptive_alpha(query: str) -> float:
    """Calculate adaptive α based on query characteristics.
    
    Parameters
    ----------
    query : str
        The user query
        
    Returns
    -------
    float
        Adaptive α value between 0 and 1
    """
    query = query.strip()
    
    # Short queries (≤3 words): rely more on dense search
    if len(query.split()) <= 3:
        return 0.35
    
    # Question queries: slightly more dense
    if "?" in query:
        return 0.45
    
    # Default: balanced approach
    return 0.55

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None


class HybridRetriever:
    """Window-based hybrid retriever combining BM25 and dense embeddings."""

    def __init__(self, bm25_path: str, embeddings_path: str, metadata_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the hybrid retriever with window-based indices.
        
        Parameters
        ----------
        bm25_path : str
            Path to BM25 index pickle file for windows
        embeddings_path : str
            Path to window embeddings numpy file
        metadata_path : str
            Path to window metadata JSON file
        model_name : str
            Name of the embedding model used
        """
        # Load BM25 index for windows
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
        
        # Load window metadata
        with open(metadata_path, 'r') as f:
            self.window_metadata = json.load(f)
        
        # Store metadata path for loading window texts
        self.metadata_path = metadata_path
        
        # Load window text data
        self.window_texts = self._load_window_texts()
        
        # Load paragraph map for expansion
        self.para_map = self._load_paragraph_map()
        
        # Initialize lightweight lemmatizer (no POS tagging)
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize embedding model
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for dense retrieval.")
        
        # Use trust_remote_code only for nomic models
        if "nomic" in model_name.lower():
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        else:
            self.model = SentenceTransformer(model_name)
        
        # Warm up the model to initialize JIT and tokenizer
        self._warm_up_model()
        
        # Load or create FAISS index for dense search
        self.faiss_index = self._load_or_create_faiss_index(embeddings_path)
    
    def _warm_up_model(self):
        """Warm up the Sentence-Transformer model to initialize JIT and tokenizer."""
        print("Warming up Sentence-Transformer model...")
        try:
            # Encode a dummy text to initialize JIT compilation and tokenizer
            dummy_text = "This is a warm-up query to initialize the model."
            _ = self.model.encode([dummy_text])
            print("✓ Sentence-Transformer model warmed up successfully")
        except Exception as e:
            print(f"Warning: Failed to warm up model: {e}")
    
    def _load_window_texts(self) -> Dict[str, str]:
        """Load window text data from windows.jsonl file."""
        window_texts = {}
        windows_file = os.path.join(os.path.dirname(self.metadata_path), "windows.jsonl")
        
        try:
            with open(windows_file, 'r') as f:
                for line in f:
                    window_data = json.loads(line.strip())
                    window_texts[window_data["window_id"]] = window_data["text"]
        except FileNotFoundError:
            print(f"Warning: {windows_file} not found. Text data will not be available.")
        except Exception as e:
            print(f"Warning: Error loading window texts: {e}")
        
        return window_texts

    def _load_paragraph_map(self) -> Dict[str, Dict]:
        """Load paragraph map from the original sections file."""
        para_map = {}
        try:
            with open("data/tc4-02.1_sections.jsonl", 'r') as f:
                for line in f:
                    para = json.loads(line)
                    para_map[para["id"]] = para
        except FileNotFoundError:
            print("Warning: Could not load paragraph map. Window expansion may not work properly.")
        return para_map

    def _load_or_create_faiss_index(self, embeddings_path: str):
        """Load existing FAISS index or create and save new one with memory mapping."""
        if faiss is None:
            raise ImportError("faiss-cpu is required for dense retrieval.")
        
        # Try optimized index first, then fallback to standard index
        optimized_path = embeddings_path.replace('.npy', '_optimized.index')
        standard_path = embeddings_path.replace('.npy', '.index')
        
        # Try to load optimized FAISS index with memory mapping
        for faiss_path in [optimized_path, standard_path]:
            if os.path.exists(faiss_path):
                try:
                    print(f"Loading FAISS index from {faiss_path}")
                    # Use memory mapping for faster loading
                    index = faiss.read_index(faiss_path, faiss.IO_FLAG_MMAP)
                    print(f"✓ FAISS index loaded successfully ({index.ntotal} vectors)")
                    return index
                except Exception as e:
                    print(f"Warning: Failed to load FAISS index from {faiss_path}: {e}")
                    continue
        
        # Create new FAISS index from embeddings
        print(f"Creating FAISS index from {embeddings_path}")
        embeddings = np.load(embeddings_path)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity (in-place, memory efficient)
        # Use asarray with copy=False to avoid doubling RAM usage
        embeddings_normalized = np.asarray(embeddings, dtype=np.float32, order='C', copy=False)
        faiss.normalize_L2(embeddings_normalized)
        index.add(embeddings_normalized)
        
        # Save FAISS index for future use
        try:
            faiss.write_index(index, standard_path)
            print(f"✓ FAISS index saved to {standard_path}")
        except Exception as e:
            print(f"Warning: Failed to save FAISS index: {e}")
        
        return index

    def _dense_search(self, query: str, top_n: int) -> Dict[int, float]:
        """Perform dense retrieval using FAISS."""
        # Encode query
        query_embedding = self.model.encode([query]).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(query_embedding, top_n)
        
        # Convert to mapping
        dense_scores = {}
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            dense_scores[int(idx)] = float(score)
        
        return dense_scores

    def _bm25_search(self, query: str, top_n: int) -> Dict[int, float]:
        """Perform BM25 retrieval on windows with lightweight lemmatization."""
        if BM25Okapi is None:
            raise ImportError("rank_bm25 is not installed, cannot use BM25 retrieval.")
        
        # Lightweight lemmatization for BM25 search
        query_words = light_lemmatize(query, self.lemmatizer)
        scores = self.bm25.get_scores(query_words)
        
        # Get top_n highest scoring indices
        top_idxs = np.argsort(-scores)[:top_n]
        return {int(idx): float(scores[idx]) for idx in top_idxs}
    
    def _bm25_search_with_ranks(self, query: str, top_n: int) -> List[Tuple[int, float, int]]:
        """Perform BM25 retrieval and return (idx, score, rank) tuples."""
        if BM25Okapi is None:
            raise ImportError("rank-bm25 is not installed, cannot use BM25 retrieval.")
        
        # Lightweight lemmatization for BM25 search
        query_words = light_lemmatize(query, self.lemmatizer)
        scores = self.bm25.get_scores(query_words)
        
        # Get top_n highest scoring indices with their ranks
        sorted_indices = np.argsort(-scores)[:top_n]
        results = []
        for rank, idx in enumerate(sorted_indices):
            results.append((int(idx), float(scores[idx]), rank))
        
        return results
    
    def _calculate_bm25_zscore_threshold(self, query: str, top_n: int = 100) -> float:
        """Calculate z-score based threshold for BM25 scores.
        
        Parameters
        ----------
        query : str
            The search query
        top_n : int
            Number of top results to consider for threshold calculation
            
        Returns
        -------
        float
            Z-score based threshold (max - mean) / std
        """
        if BM25Okapi is None:
            return 0.005  # Fallback to static threshold
        
        # Lightweight lemmatization for BM25 search
        query_words = light_lemmatize(query, self.lemmatizer)
        scores = self.bm25.get_scores(query_words)
        
        # Get top_n scores for threshold calculation
        top_scores = np.sort(scores)[-top_n:]
        
        if len(top_scores) < 2:
            return 0.005  # Fallback if not enough scores
        
        # Calculate z-score threshold: (max - mean) / std
        max_score = np.max(top_scores)
        mean_score = np.mean(top_scores)
        std_score = np.std(top_scores)
        
        if std_score == 0 or std_score < 1e-10:
            # If no variance, use a more aggressive threshold based on score range
            score_range = max_score - np.min(top_scores)
            return max(0.001, min(0.01, score_range * 0.1))
        
        zscore_threshold = (max_score - mean_score) / std_score
        
        # Normalize to a reasonable range (0.001 to 0.01)
        # Higher z-score means more discriminative, so lower threshold
        # Use a more aggressive scaling factor
        normalized_threshold = max(0.001, min(0.01, zscore_threshold * 0.01))
        
        return normalized_threshold
    
    def _dense_search_with_ranks(self, query: str, top_n: int) -> List[Tuple[int, float, int]]:
        """Perform dense retrieval and return (idx, score, rank) tuples."""
        # Encode query
        query_embedding = self.model.encode([query]).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(query_embedding, top_n)
        
        # Convert to (idx, score, rank) tuples
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            results.append((int(idx), float(score), rank))
        
        return results

    def search(self, query: str, state_hint: Optional[str] = None, k: int = 6, bm25_n: int = 50, faiss_n: int = 50, alpha: Optional[float] = None, bm25_query: Optional[str] = None, dense_query: Optional[str] = None) -> Tuple[List[Dict], float]:
        """Return the top `k` parent windows for a given query using hybrid retrieval.

        Parameters
        ----------
        query: str
            The user query (e.g., "needle decompression site for tension pneumothorax").
        state_hint: Optional[str]
            A keyword describing the current MARCH‑PAWS state (e.g., "respiration OR chest OR pneumothorax").
        k: int
            Number of parent windows to return.
        bm25_n: int
            Number of BM25 candidates to consider.
        faiss_n: int
            Number of dense candidates to consider.
        alpha: Optional[float]
            Weighting between BM25 and dense scores in the fusion. If None, uses adaptive α based on query characteristics.

        Returns
        -------
        Tuple[List[Dict], float]
            List of parent window dictionaries with hybrid scores, sorted by descending score,
            and the calculated z-score based threshold.
        """
        # Calculate adaptive α if not provided
        if alpha is None:
            alpha = calculate_adaptive_alpha(query)
        
        # Use separate queries for BM25 and dense retrieval
        if bm25_query is not None:
            bm25_query_final = bm25_query
        else:
            # Augment the query with the state hint for BM25 only (lexical matching)
            if state_hint:
                bm25_query_final = f"{query} {state_hint}"
            else:
                bm25_query_final = query
        
        if dense_query is not None:
            dense_query_final = dense_query
        else:
            # Keep dense query unchanged (semantic matching)
            dense_query_final = query

        # Get ranked results from both retrieval methods
        bm25_results = self._bm25_search_with_ranks(bm25_query_final, int(bm25_n))
        dense_results = self._dense_search_with_ranks(dense_query_final, int(faiss_n))

        # Reciprocal Rank Fusion (RRF) using original ranks
        fused: Dict[int, float] = defaultdict(float)
        
        # RRF from BM25: α / (rank + 60)
        for idx, score, rank in bm25_results:
            fused[idx] += alpha / (rank + 60)
        
        # RRF from dense: (1-α) / (rank + 60) - using ranks, not raw scores
        for idx, score, rank in dense_results:
            fused[idx] += (1 - alpha) / (rank + 60)

        # Get top-k parent windows
        sorted_windows = sorted(fused.items(), key=lambda x: -x[1])[:k]
        
        # Calculate z-score based threshold for this query
        zscore_threshold = self._calculate_bm25_zscore_threshold(bm25_query_final, bm25_n)
        
        results = []
        for idx, score in sorted_windows:
            window_meta = self.window_metadata[idx]
            window_result = {
                "window_id": window_meta["window_id"],
                "stride_idx": window_meta["stride_idx"],
                "paragraph_ids": window_meta["paragraph_ids"],
                "chapter": window_meta["chapter"],
                "page_start": window_meta["page_start"],
                "page_end": window_meta["page_end"],
                "headings": window_meta["headings"],
                "section_paths": window_meta["section_paths"],
                "text": self.window_texts.get(window_meta["window_id"], ""),  # Get text from windows data
                "score": float(score)
            }
            results.append(window_result)
        
        return results, zscore_threshold

    def expand_windows(self, win_hits: List[Dict], max_paras: int = 10) -> List[Dict]:
        """Expand parent windows to child paragraphs with deduplication using round-robin.
        
        This method ensures we get paragraphs from multiple windows rather than
        exhausting the first window before moving to the next.
        
        Parameters
        ----------
        win_hits : List[Dict]
            List of parent windows from search()
        max_paras : int
            Maximum number of child paragraphs to return
            
        Returns
        -------
        List[Dict]
            List of child paragraph dictionaries, deduplicated and ordered
        """
        if not win_hits:
            return []
        
        paras = []
        seen: Set[str] = set()
        
        # Create iterators for each window's paragraph_ids
        window_iterators = []
        for window in win_hits:
            para_ids = [pid for pid in window["paragraph_ids"] if pid in self.para_map]
            if para_ids:
                window_iterators.append(iter(para_ids))
        
        # Round-robin through windows to get paragraphs
        while len(paras) < max_paras and window_iterators:
            # Track which iterators are exhausted
            exhausted_iterators = []
            
            for i, para_iter in enumerate(window_iterators):
                if len(paras) >= max_paras:
                    break
                
                try:
                    para_id = next(para_iter)
                    if para_id not in seen:
                        seen.add(para_id)
                        paras.append(self.para_map[para_id])
                except StopIteration:
                    # This iterator is exhausted
                    exhausted_iterators.append(i)
            
            # Remove exhausted iterators
            for i in reversed(exhausted_iterators):
                window_iterators.pop(i)
        
        return paras
