#!/usr/bin/env python3
"""Embed windows using a stable model that works reliably."""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import time
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

# Configuration
WINDOWS_FILE = "data/windows.jsonl"
EMBEDDINGS_FILE = "data/window_embeddings.npy"
METADATA_FILE = "data/window_metadata.json"
INFO_FILE = "data/embedding_info.json"
MODEL_NAME = "all-MiniLM-L6-v2"  # Stable, reliable model
BATCH_SIZE = 32

def load_windows(file_path):
    """Load windows from JSONL file."""
    with open(file_path) as f:
        return [json.loads(line) for line in f]

def embed_batch(texts_batch, model_name=MODEL_NAME):
    """Embed a batch of texts using the specified model."""
    model = SentenceTransformer(model_name)
    return model.encode(texts_batch, convert_to_numpy=True)

def embed_windows_parallel(windows, model_name=MODEL_NAME, batch_size=BATCH_SIZE, n_jobs=None):
    """Embed windows using parallel processing for faster ingestion."""
    if n_jobs is None:
        n_jobs = min(multiprocessing.cpu_count(), 4)  # Limit to 4 to avoid memory issues
    
    print(f"Using {n_jobs} parallel workers for embedding")
    
    # Extract texts for embedding
    texts = [window["text"] for window in windows]
    print(f"Embedding {len(texts)} windows in parallel...")
    
    # Split texts into batches for parallel processing
    text_batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    start_time = time.time()
    
    # Process batches in parallel
    embedding_batches = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(embed_batch)(batch, model_name) for batch in tqdm(text_batches, desc="Embedding batches")
    )
    
    # Concatenate all embeddings
    embeddings = np.vstack(embedding_batches)
    
    end_time = time.time()
    
    print(f"Parallel embedding completed in {end_time - start_time:.2f} seconds")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Average time per window: {(end_time - start_time) / len(texts):.3f} seconds")
    
    # Load model for return (needed for testing)
    model = SentenceTransformer(model_name)
    return embeddings, model

def embed_windows(windows, model_name=MODEL_NAME, batch_size=BATCH_SIZE, use_parallel=True):
    """Embed windows using the specified model with optional parallel processing."""
    if use_parallel:
        return embed_windows_parallel(windows, model_name, batch_size)
    else:
        # Original sequential method
        print(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        
        # Extract texts for embedding
        texts = [window["text"] for window in windows]
        print(f"Embedding {len(texts)} windows...")
        
        start_time = time.time()
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        end_time = time.time()
        
        print(f"Embedding completed in {end_time - start_time:.2f} seconds")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Average time per window: {(end_time - start_time) / len(texts):.3f} seconds")
        
        return embeddings, model

def save_embeddings(embeddings, windows, model_info):
    """Save embeddings, metadata, and model info."""
    np.save(EMBEDDINGS_FILE, embeddings)
    
    # Create metadata file from windows
    metadata = []
    for window in windows:
        meta_entry = {k: v for k, v in window.items() if k != "text"}  # Exclude raw text from metadata
        metadata.append(meta_entry)
    
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    
    with open(INFO_FILE, "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Embeddings saved to {EMBEDDINGS_FILE}")
    print(f"Metadata saved to {METADATA_FILE}")
    print(f"Model info saved to {INFO_FILE}")

def test_retrieval(query, embeddings, metadata, model, k=5):
    """Test retrieval with a sample query."""
    print(f"\nTesting retrieval with query: '{query}'")
    query_embedding = model.encode(query, convert_to_numpy=True)
    
    # Calculate cosine similarity
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    
    # Get top k results
    top_indices = np.argsort(similarities)[::-1][:k]
    
    for i, idx in enumerate(top_indices):
        window = metadata[idx]
        print(f"{i+1}. Window {window['window_id']} (similarity: {similarities[idx]:.4f})")
        print(f"   Chapter: {window['chapter']}, Pages: {window['page_start']}-{window['page_end']}")
        print(f"   Headings: {', '.join(window['headings'])}")
        print(f"   Section: {', '.join(window['section_paths'])}")

if __name__ == "__main__":
    windows = load_windows(WINDOWS_FILE)
    print(f"Loaded {len(windows)} windows")

    # Basic text length stats
    text_lengths = [len(w["text"].split()) for w in windows]
    print(f"Text length stats: min={min(text_lengths)}, max={max(text_lengths)}, avg={sum(text_lengths)/len(text_lengths):.1f}")

    embeddings, model = embed_windows(windows, use_parallel=True)
    
    # Prepare model info
    model_info = {
        "model_name": MODEL_NAME,
        "embedding_dim": embeddings.shape[1],
        "num_windows": len(windows),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    }
    
    save_embeddings(embeddings, windows, model_info)
    
    # Load metadata for testing
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)

    # Test retrieval with sample queries
    sample_queries = [
        "massive hemorrhage control",
        "airway obstruction treatment",
        "chest wound bleeding",
        "shock management"
    ]
    for query in sample_queries:
        test_retrieval(query, embeddings, metadata, model)
