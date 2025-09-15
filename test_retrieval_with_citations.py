#!/usr/bin/env python3
"""
Simple scenario-based document retrieval test with smart paragraph selection.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.orchestrator_async import AsyncOrchestrator
from src.utils import get_smart_paragraphs_from_windows, format_excerpts


async def test_scenario_retrieval():
    """Test document retrieval based on user-provided scenario."""
    
    print("🔍 Scenario-Based Document Retrieval Test")
    print("=" * 50)
    
    # Get scenario from user
    scenario = input("Enter a medical scenario: ").strip()
    if not scenario:
        print("❌ No scenario provided. Exiting.")
        return
    
    print(f"\n📋 Scenario: {scenario}")
    print("-" * 50)
    
    # Initialize orchestrator
    try:
        orchestrator = AsyncOrchestrator(
            bm25_path='data/window_bm25_index.pkl',
            embeddings_path='data/window_embeddings.npy',
            metadata_path='data/window_metadata.json'
        )
        print("✅ Orchestrator initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")
        return
    
    try:
        # Step 1: Retrieve windows based on scenario
        print(f"\n🔍 Step 1: Retrieving documents for scenario...")
        win_hits, dynamic_threshold = orchestrator.retriever.search(
            scenario,
            state_hint=None,  # No specific state, just general retrieval
            k=10,
            bm25_n=20,
            faiss_n=20
        )
        
        print(f"Found {len(win_hits)} windows (threshold: {dynamic_threshold:.3f})")
        
        if win_hits:
            # Display main window retrieval results
            print(f"\n🪟 Main Window Retrieval Results:")
            print("-" * 50)
            
            for i, window in enumerate(win_hits):
                window_id = window.get("window_id", "Unknown")
                score = window.get("score", 0.0)
                chapter = window.get("chapter", "?")
                page_start = window.get("page_start", "?")
                page_end = window.get("page_end", "?")
                para_count = len(window.get("paragraph_ids", []))
                text = window.get("text", "")[:150] + "..." if len(window.get("text", "")) > 150 else window.get("text", "")
                
                print(f"{i+1}. Window: {window_id} - Score: {score:.4f}")
                print(f"   Ch{chapter}, Pages {page_start}-{page_end}, {para_count} paragraphs")
                print(f"   Text: {text}")
                print()
            
            # Step 2: Cross-encoder reranking
            print(f"\n🎯 Step 2: Cross-encoder reranking...")
            reranked_hits = orchestrator.reranker.rerank(scenario, win_hits[:10])
            print(f"Reranked {len(reranked_hits)} windows")
            
            # Display reranked results
            print(f"\n🔄 Reranked Window Results:")
            print("-" * 40)
            
            for i, window in enumerate(reranked_hits):
                window_id = window.get("window_id", "Unknown")
                score_ce = window.get("score_ce", 0.0)
                original_score = window.get("score", 0.0)
                chapter = window.get("chapter", "?")
                page_start = window.get("page_start", "?")
                page_end = window.get("page_end", "?")
                para_count = len(window.get("paragraph_ids", []))
                text = window.get("text", "")[:150] + "..." if len(window.get("text", "")) > 150 else window.get("text", "")
                
                print(f"{i+1}. Window: {window_id} - CE Score: {score_ce:.4f} (Original: {original_score:.4f})")
                print(f"   Ch{chapter}, Pages {page_start}-{page_end}, {para_count} paragraphs")
                print(f"   Text: {text}")
                print()
            
            # Step 3: Smart paragraph selection
            print(f"\n🧠 Step 3: Smart paragraph selection...")
            smart_paragraphs = get_smart_paragraphs_from_windows(
                reranked_hits[:5],  # Use top 5 windows
                scenario,
                orchestrator.reranker,
                orchestrator.retriever,
                max_paragraphs=8
            )
            
            print(f"Selected {len(smart_paragraphs)} most relevant paragraphs")
            
            # Display smart paragraph results
            print(f"\n📊 Smart Paragraph Selection Results:")
            print("-" * 45)
            
            for i, para in enumerate(smart_paragraphs):
                chapter = para.get("chapter", "?")
                para_id = para.get("para", "?")
                version = para.get("version", "Base")
                score = para.get("score_ce", 0.0)
                para_id_full = para.get("id", "Unknown")
                text = para.get("text", "")[:200] + "..." if len(para.get("text", "")) > 200 else para.get("text", "")
                
                print(f"{i+1}. Ch{chapter} §{para_id} ({version}) - Relevance: {score:.4f}")
                print(f"   ID: {para_id_full}")
                print(f"   Text: {text}")
                print()
                
        else:
            print("❌ No relevant documents found for this scenario")
            
    except Exception as e:
        print(f"❌ Error during retrieval: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎯 Retrieval test complete!")


if __name__ == "__main__":
    asyncio.run(test_scenario_retrieval())