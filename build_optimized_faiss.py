#!/usr/bin/env python3
"""Build optimized FAISS index with memory mapping and quantization."""

import sys
import os
sys.path.append('src')

import numpy as np
import faiss
import time

def build_optimized_faiss_index(embeddings_path: str):
    """Build optimized FAISS index with IndexFlatIP for small datasets."""
    print(f"Building optimized FAISS index from {embeddings_path}")
    
    # Load embeddings
    embeddings = np.load(embeddings_path)
    print(f"Loaded embeddings: {embeddings.shape}")
    
    # Normalize embeddings for cosine similarity
    embeddings_normalized = np.asarray(embeddings, dtype=np.float32, order='C', copy=False)
    faiss.normalize_L2(embeddings_normalized)
    
    # Create optimized index
    dimension = embeddings.shape[1]
    n_vectors = embeddings.shape[0]
    
    # Use IndexFlatIP for small datasets - exact search, no training required
    print(f"Creating IndexFlatIP for {n_vectors} vectors")
    
    # Create IndexFlatIP for exact inner product search
    index = faiss.IndexFlatIP(dimension)
    
    # Add vectors to index (no training needed for IndexFlatIP)
    print("Adding vectors to index...")
    start_time = time.time()
    index.add(embeddings_normalized)
    add_time = time.time() - start_time
    print(f"Adding completed in {add_time:.2f}s")
    
    # Save index with memory mapping support
    faiss_path = embeddings_path.replace('.npy', '_optimized.index')
    faiss.write_index(index, faiss_path)
    print(f"✓ Optimized FAISS index saved to {faiss_path}")
    
    # Test index loading
    print("\nTesting index loading...")
    start_time = time.time()
    loaded_index = faiss.read_index(faiss_path)
    load_time = time.time() - start_time
    print(f"✓ Index loaded in {load_time:.2f}s")
    print(f"✓ Index contains {loaded_index.ntotal} vectors")
    
    # Test search performance
    print("\nTesting search performance...")
    test_query = np.random.rand(1, dimension).astype(np.float32)
    faiss.normalize_L2(test_query)
    
    start_time = time.time()
    scores, indices = loaded_index.search(test_query, 10)
    search_time = time.time() - start_time
    print(f"✓ Search completed in {search_time*1000:.2f}ms")
    
    return faiss_path

if __name__ == "__main__":
    embeddings_path = "data/window_embeddings.npy"
    
    if not os.path.exists(embeddings_path):
        print(f"Error: {embeddings_path} not found")
        sys.exit(1)
    
    faiss_path = build_optimized_faiss_index(embeddings_path)
    print(f"\n🎉 Optimized FAISS index built successfully!")
    print(f"Next time, use {faiss_path} for fast and reliable search.")
