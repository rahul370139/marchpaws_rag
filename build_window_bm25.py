#!/usr/bin/env python3
"""Build BM25 index for windows."""

import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def load_windows(file_path):
    """Load windows from JSONL file."""
    with open(file_path) as f:
        return [json.loads(line) for line in f]

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_text(text, lemmatizer):
    """Preprocess text for BM25 indexing with lemmatization."""
    # Convert to lowercase and split into words
    text = text.lower()
    # Remove extra whitespace and split
    words = re.findall(r'\b\w+\b', text)
    # Lemmatize each word
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
    return lemmatized_words

def build_bm25_index(windows):
    """Build BM25 index from windows with lemmatization."""
    print(f"Building BM25 index for {len(windows)} windows...")
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Extract and preprocess texts with headings for BM25
    texts = []
    for window in windows:
        # Combine headings with text for BM25 (but not for embeddings)
        headings = window.get("headings", [])
        heading_text = " ".join(headings) if headings else ""
        
        # Create full text: headings + " " + original text
        full_text = f"{heading_text} {window['text']}".strip()
        
        # Preprocess the combined text
        processed_text = preprocess_text(full_text, lemmatizer)
        texts.append(processed_text)
    
    # Build BM25 index
    bm25 = BM25Okapi(texts)
    
    print(f"BM25 index built successfully!")
    print(f"Corpus size: {bm25.corpus_size}")
    print(f"Vocabulary size: {len(bm25.idf)}")
    
    return bm25

def test_bm25_retrieval(bm25, windows, query, top_k=5):
    """Test BM25 retrieval with a sample query."""
    print(f"\nTesting BM25 retrieval with query: '{query}'")
    
    # Initialize lemmatizer for query
    lemmatizer = WordNetLemmatizer()
    
    # Preprocess query
    query_words = preprocess_text(query, lemmatizer)
    
    # Get scores
    scores = bm25.get_scores(query_words)
    
    # Get top-k results
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    print(f"Top {top_k} BM25 results:")
    for i, idx in enumerate(top_indices):
        window = windows[idx]
        score = scores[idx]
        print(f"{i+1}. {window['window_id']} (BM25 score: {score:.4f})")
        print(f"   Ch{window['chapter']} p{window['page_start']}-{window['page_end']}")
        print(f"   Headings: {', '.join(window['headings'])}")
        if window['section_paths']:
            print(f"   Section: {window['section_paths'][0]}")
        print(f"   Text preview: {window['text'][:100]}...")
        
        # Show the enhanced BM25 text (headings + text)
        headings = window.get("headings", [])
        heading_text = " ".join(headings) if headings else ""
        enhanced_text = f"{heading_text} {window['text']}".strip()
        print(f"   BM25 enhanced text: {enhanced_text[:150]}...")
        print()

def download_nltk_data():
    """Download required NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        print("NLTK data downloaded successfully!")

def main():
    # Download NLTK data if needed
    download_nltk_data()
    
    # Load windows
    windows = load_windows("data/windows.jsonl")
    print(f"Loaded {len(windows)} windows")
    
    # Build BM25 index
    bm25 = build_bm25_index(windows)
    
    # Save BM25 index
    output_path = "data/window_bm25_index.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(bm25, f)
    print(f"BM25 index saved to {output_path}")
    
    # Test retrieval with sample queries
    test_queries = [
        "massive hemorrhage control",
        "airway obstruction treatment", 
        "chest wound bleeding",
        "shock management",
        "fracture immobilization"
    ]
    
    for query in test_queries:
        test_bm25_retrieval(bm25, windows, query)
    
    # Save window texts for hybrid retrieval
    window_texts = [window["text"] for window in windows]
    texts_path = "data/window_texts.json"
    with open(texts_path, 'w') as f:
        json.dump(window_texts, f, indent=2)
    print(f"Window texts saved to {texts_path}")
    
    print(f"\nBM25 indexing completed successfully!")
    print(f"Files created:")
    print(f"  - {output_path}")
    print(f"  - {texts_path}")

if __name__ == "__main__":
    main()
