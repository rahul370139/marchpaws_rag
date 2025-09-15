#!/usr/bin/env python3
"""MARCH-PAWS Medical Assistant - Streamlit App

A RAG-powered medical assistant following the MARCH-PAWS protocol for tactical combat casualty care.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append('src')

from src.orchestrator_async import AsyncOrchestrator
from quality_evaluator import QualityEvaluator
import asyncio
import json
from src.utils import map_citations_to_database, get_smart_paragraphs_from_windows

# Page configuration
st.set_page_config(
    page_title="MARCH-PAWS Medical Assistant - Enhanced",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force light theme
st.markdown("""
<script>
    // Force Streamlit to use light theme
    const theme = {
        "base": "light",
        "primaryColor": "#1f77b4",
        "backgroundColor": "#ffffff",
        "secondaryBackgroundColor": "#f0f2f6",
        "textColor": "#000000"
    };
    
    // Apply theme immediately
    const root = document.querySelector('.stApp');
    if (root) {
        root.setAttribute('data-theme', 'light');
    }
</script>
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>
    /* Force light theme and bright colors */
    .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Strong override for assessment history expander background */
    .stExpander, .stExpander > div, .stExpanderContent, .stExpanderContent div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Make all text bright and visible */
    .stMarkdown, .stText, .stTextInput, .stTextArea, .stSelectbox, .stButton, 
    .stExpander, .stExpander > div, .stExpanderContent, .element-container,
    .stMarkdown p, .stMarkdown div, .stMarkdown span, .stMarkdown strong,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #000000 !important;
        background-color: transparent !important;
    }
    
    /* Force sidebar to be bright */
    .stApp [data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        color: #000000 !important;
    }
    
    .stApp [data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    /* Override Streamlit's dark theme */
    .stApp [data-testid="stSidebar"] .stMarkdown,
    .stApp [data-testid="stSidebar"] .stText,
    .stApp [data-testid="stSidebar"] p,
    .stApp [data-testid="stSidebar"] div,
    .stApp [data-testid="stSidebar"] span {
        color: #000000 !important;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Text input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #ff4444 !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stage-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .checklist-item {
        background-color: #f0f8ff !important;
        color: #000000 !important;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        border-radius: 4px;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .citation {
        font-size: 0.9rem;
        color: #000000 !important;
        font-style: italic;
        background-color: #f8f9fa !important;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-left: 3px solid #28a745;
        border-radius: 3px;
        font-weight: 500;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 4px;
    }
    .quality-score {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 4px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-weight: bold;
        text-align: center;
    }
    .quality-excellent { background-color: #d4edda; border-color: #28a745; }
    .quality-good { background-color: #d1ecf1; border-color: #17a2b8; }
    .quality-fair { background-color: #fff3cd; border-color: #ffc107; }
    .quality-poor { background-color: #f8d7da; border-color: #dc3545; }
    .transparency-panel {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .system-info {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .performance-metric {
        display: inline-block;
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 0.5rem;
        margin: 0.25rem;
        text-align: center;
        min-width: 80px;
    }
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Ensure all text in expanders is black and background is light */
    .streamlit-expanderContent {
        color: #000000 !important;
        background-color: #ffffff !important;
        background: #ffffff !important;
    }
    
    /* Fix expander styling to prevent dark backgrounds */
    .streamlit-expander {
        background-color: #ffffff !important;
        background: #ffffff !important;
        border: 1px solid #dee2e6 !important;
    }
    
    /* Ensure all markdown content in expanders is visible */
    .streamlit-expanderContent .markdown-text-container,
    .streamlit-expanderContent p,
    .streamlit-expanderContent strong,
    .streamlit-expanderContent ul,
    .streamlit-expanderContent li,
    .streamlit-expanderContent h1,
    .streamlit-expanderContent h2,
    .streamlit-expanderContent h3,
    .streamlit-expanderContent h4,
    .streamlit-expanderContent h5,
    .streamlit-expanderContent h6,
    .streamlit-expanderContent div,
    .streamlit-expanderContent span {
        color: #000000 !important;
        background-color: transparent !important;
    }
    
    /* Fix assessment history expanders specifically */
    .streamlit-expanderContent .scenario-text,
    .streamlit-expanderContent .question-text,
    .streamlit-expanderContent .checklist-item,
    .streamlit-expanderContent .citation {
        color: #000000 !important;
        background-color: transparent !important;
    }
    
    /* Fix refusal scenario text visibility */
    .stAlert,
    .stAlert p,
    .stAlert div,
    .stAlert span {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Specific fix for error alerts (red boxes) */
    .stAlert[data-testid="error"] {
        background-color: #f8d7da !important;
        border: 1px solid #f5c6cb !important;
    }
    
    .stAlert[data-testid="error"] p,
    .stAlert[data-testid="error"] div,
    .stAlert[data-testid="error"] span {
        color: #000000 !important;
        font-weight: bold !important;
    }
    
    /* Fix info alerts (blue boxes) */
    .stAlert[data-testid="info"] {
        background-color: #d1ecf1 !important;
        border: 1px solid #bee5eb !important;
    }
    
    .stAlert[data-testid="info"] p,
    .stAlert[data-testid="info"] div,
    .stAlert[data-testid="info"] span {
        color: #000000 !important;
    }
    
    /* Fix warning alerts (yellow boxes) */
    .stAlert[data-testid="warning"] {
        background-color: #fff3cd !important;
        border: 1px solid #ffeaa7 !important;
    }
    
    .stAlert[data-testid="warning"] p,
    .stAlert[data-testid="warning"] div,
    .stAlert[data-testid="warning"] span {
        color: #000000 !important;
    }
    
    /* ULTRA AGGRESSIVE: Override ALL possible expander header styling */
    .streamlit-expanderHeader,
    .streamlit-expanderHeader:hover,
    .streamlit-expanderHeader:focus,
    .streamlit-expanderHeader:active,
    .streamlit-expanderHeader:visited,
    .streamlit-expanderHeader:link {
        color: #000000 !important;
        background-color: #f8f9fa !important;
        background: #f8f9fa !important;
        background-image: none !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 4px !important;
        box-shadow: none !important;
    }
    
    /* Ensure expander header text is always visible */
    .streamlit-expanderHeader p,
    .streamlit-expanderHeader div,
    .streamlit-expanderHeader span {
        color: #000000 !important;
        background-color: transparent !important;
    }
    
    /* Target ALL possible child elements */
    .streamlit-expanderHeader *,
    .streamlit-expanderHeader p,
    .streamlit-expanderHeader div,
    .streamlit-expanderHeader span,
    .streamlit-expanderHeader button,
    .streamlit-expanderHeader h1,
    .streamlit-expanderHeader h2,
    .streamlit-expanderHeader h3,
    .streamlit-expanderHeader h4,
    .streamlit-expanderHeader h5,
    .streamlit-expanderHeader h6 {
        color: #000000 !important;
        font-weight: 700 !important;
        background-color: transparent !important;
        background: transparent !important;
    }
    
    /* Override Streamlit's specific classes */
    [data-testid="stExpander"] .streamlit-expanderHeader,
    [data-testid="stExpander"] .streamlit-expanderHeader:hover,
    [data-testid="stExpander"] .streamlit-expanderHeader:focus {
        background-color: #ffffff !important;
        color: #000000 !important;
        background: #ffffff !important;
    }
    
    /* Target the specific expander in assessment history */
    .element-container .streamlit-expanderHeader,
    .element-container .streamlit-expanderHeader:hover,
    .element-container .streamlit-expanderHeader:focus {
        background-color: #ffffff !important;
        color: #000000 !important;
        background: #ffffff !important;
        border: 2px solid #28a745 !important;
    }
    
    /* Make sure the chevron icon is also visible */
    .streamlit-expanderHeader svg,
    .streamlit-expanderHeader path,
    .streamlit-expanderHeader g {
        color: #000000 !important;
        fill: #000000 !important;
        stroke: #000000 !important;
    }
    
    /* Override any dark theme classes */
    .stApp .streamlit-expanderHeader,
    .stApp .streamlit-expanderHeader:hover {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Nuclear option: target by attribute */
    [class*="expanderHeader"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* CRITICAL FIX: Override expander content background */
    .streamlit-expanderContent {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #dee2e6 !important;
        border-top: none !important;
    }
    
    /* Fix the dark expander content that appears when expanded */
    .streamlit-expanderContent > div,
    .streamlit-expanderContent > .element-container,
    .streamlit-expanderContent > .stContainer,
    .streamlit-expanderContent > .block-container {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Target specific Streamlit components inside expanders */
    .streamlit-expanderContent .stContainer > div,
    .streamlit-expanderContent .element-container > div,
    .streamlit-expanderContent .block-container > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Fix any nested containers */
    .streamlit-expanderContent div[data-testid="stContainer"],
    .streamlit-expanderContent div[data-testid="element-container"],
    .streamlit-expanderContent div[data-testid="block-container"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Nuclear option for expander content */
    [class*="expanderContent"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* AGGRESSIVE FIX for dynamically generated assessment history expanders */
    .streamlit-expanderContent[style*="background"],
    .streamlit-expanderContent[style*="color"] {
        background-color: #ffffff !important;
        background: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Target expanders that are expanded by default */
    .streamlit-expanderContent[aria-expanded="true"],
    .streamlit-expanderContent[data-expanded="true"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Force override any inline styles */
    .streamlit-expanderContent[style] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Target the specific assessment history section */
    div[data-testid="stExpander"] .streamlit-expanderContent {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Ultra aggressive: target by parent context */
    .element-container:has(.streamlit-expanderContent) .streamlit-expanderContent,
    .stContainer:has(.streamlit-expanderContent) .streamlit-expanderContent {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Force override for all possible Streamlit expander variations */
    [data-testid="stExpander"] .streamlit-expanderContent,
    [data-testid="stExpander"] > div > div,
    [data-testid="stExpander"] .streamlit-expanderContent > div,
    [data-testid="stExpander"] .streamlit-expanderContent .element-container,
    [data-testid="stExpander"] .streamlit-expanderContent .stContainer {
        background-color: #ffffff !important;
        background: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Make question text bright and clear */
    .question-text {
        font-size: 1.2rem;
        font-weight: 700;
        color: #000000 !important;
        background-color: #fff3cd !important;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        border-radius: 4px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Make scenario text bright and clear */
    .scenario-text {
        font-size: 1.1rem;
        font-weight: 600;
        color: #000000 !important;
        background-color: #e3f2fd !important;
        padding: 0.8rem;
        border-left: 4px solid #2196f3;
        border-radius: 4px;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Native <details>/<summary> fix for Streamlit expander */
    details,
    details[open],
    details > summary,
    details[open] > summary,
    details > div,
    details[open] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
</style>

<script>
// Force fix expander content styling after page load
function fixExpanderStyling() {
    const expanderContents = document.querySelectorAll('.streamlit-expanderContent');
    expanderContents.forEach(content => {
        content.style.backgroundColor = '#ffffff';
        content.style.color = '#000000';
        content.style.background = '#ffffff';
        
        // Fix nested divs
        const nestedDivs = content.querySelectorAll('div');
        nestedDivs.forEach(div => {
            div.style.backgroundColor = '#ffffff';
            div.style.color = '#000000';
            div.style.background = '#ffffff';
        });
    });
}

// Run immediately
fixExpanderStyling();

// Run after DOM changes (for dynamically generated content)
const observer = new MutationObserver(fixExpanderStyling);
observer.observe(document.body, { childList: true, subtree: true });

// Also run periodically as a fallback
setInterval(fixExpanderStyling, 1000);
</script>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    """Load the enhanced MARCH-PAWS system with async orchestrator and quality evaluator."""
    try:
        # Load quality evaluator
        evaluator = QualityEvaluator()
        
        # Load async orchestrator
        orchestrator = AsyncOrchestrator(
            bm25_path='data/window_bm25_index.pkl',
            embeddings_path='data/window_embeddings.npy',
            metadata_path='data/window_metadata.json'
        )
        
        return {
            'orchestrator': orchestrator,
            'evaluator': evaluator,
            'status': 'ready'
        }
    except Exception as e:
        st.error(f"Failed to load system: {e}")
        return None

async def initialize_orchestrator(orchestrator):
    """Initialize the async orchestrator."""
    try:
        await orchestrator.__aenter__()
        return True
    except Exception as e:
        st.error(f"Failed to initialize orchestrator: {e}")
        return False

def display_quality_metrics(evaluation_result):
    """Display quality evaluation metrics with visual indicators."""
    if not evaluation_result:
        return
    
    overall_score = evaluation_result.get('overall_score', 0)
    
    # Determine quality level and color
    if overall_score >= 0.9:
        quality_level = "Excellent"
        quality_class = "quality-excellent"
    elif overall_score >= 0.7:
        quality_level = "Good"
        quality_class = "quality-good"
    elif overall_score >= 0.5:
        quality_level = "Fair"
        quality_class = "quality-fair"
    else:
        quality_level = "Needs Improvement"
        quality_class = "quality-poor"
    
    st.markdown(f'''
    <div class="quality-score {quality_class}">
        🎯 Quality Score: {overall_score:.3f} ({quality_level})
    </div>
    ''', unsafe_allow_html=True)
    
    # Display detailed metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Question Quality", f"{evaluation_result.get('question_score', 0):.3f}")
    with col2:
        st.metric("Citation Accuracy", f"{evaluation_result.get('citation_score', 0):.3f}")
    with col3:
        st.metric("Checklist Quality", f"{evaluation_result.get('checklist_score', 0):.3f}")
    with col4:
        st.metric("Response Time", f"{evaluation_result.get('response_time', 0):.1f}s")
    
    # Display recommendations and issues
    recommendations = evaluation_result.get('recommendations', {})
    
    # Citation recommendations (new feature)
    if recommendations.get('citations'):
        with st.expander("📋 Citation Quality Feedback", expanded=False):
            citation_recs = recommendations['citations']
            if citation_recs:
                st.markdown("**Issues Found:**")
                for rec in citation_recs:
                    st.markdown(f"• {rec}")
            else:
                st.success("✅ All citations are valid and unique!")
    
    # Question recommendations
    if recommendations.get('question'):
        with st.expander("❓ Question Quality Feedback", expanded=False):
            question_recs = recommendations['question']
            if question_recs:
                st.markdown("**Suggestions:**")
                for rec in question_recs:
                    st.markdown(f"• {rec}")
            else:
                st.success("✅ Question quality is excellent!")
    
    # Checklist recommendations
    if recommendations.get('checklist'):
        with st.expander("✅ Checklist Quality Feedback", expanded=False):
            checklist_recs = recommendations['checklist']
            if checklist_recs:
                st.markdown("**Suggestions:**")
                for rec in checklist_recs:
                    st.markdown(f"• {rec}")
            else:
                st.success("✅ Checklist quality is excellent!")

def test_retrieval_for_ui(orchestrator, query):
    """Test retrieval for UI display - synchronous version of the terminal test."""
    try:
        # Step 1: Retrieve windows
        win_hits, dynamic_threshold = orchestrator.retriever.search(
            query,
            state_hint=None,
            k=10,
            bm25_n=20,
            faiss_n=20
        )
        
        if not win_hits:
            return {
                "error": "No relevant documents found for this scenario",
                "windows": [],
                "reranked": [],
                "paragraphs": []
            }
        
        # Step 2: Cross-encoder reranking
        reranked_hits = orchestrator.reranker.rerank(query, win_hits[:10])
        
        # Step 3: Smart paragraph selection
        smart_paragraphs = get_smart_paragraphs_from_windows(
            reranked_hits[:5],  # Use top 5 windows
            query,
            orchestrator.reranker,
            orchestrator.retriever,
            max_paragraphs=8
        )
        
        return {
            "query": query,
            "threshold": dynamic_threshold,
            "windows": win_hits,
            "reranked": reranked_hits,
            "paragraphs": smart_paragraphs
        }
        
    except Exception as e:
        return {
            "error": f"Retrieval test failed: {str(e)}",
            "windows": [],
            "reranked": [],
            "paragraphs": []
        }


def display_retrieval_results_main_page(results):
    """Display retrieval results on the main page."""
    if "error" in results:
        st.error(results["error"])
        return
    
    st.markdown(f"**Query:** {results['query']}")
    st.markdown(f"**Threshold:** {results['threshold']:.3f}")
    
    # Main window results
    with st.expander("🪟 Main Window Results", expanded=False):
        for i, window in enumerate(results['windows'][:5]):  # Show top 5
            window_id = window.get("window_id", "Unknown")
            score = window.get("score", 0.0)
            chapter = window.get("chapter", "?")
            page_start = window.get("page_start", "?")
            page_end = window.get("page_end", "?")
            para_count = len(window.get("paragraph_ids", []))
            text = window.get("text", "")[:100] + "..." if len(window.get("text", "")) > 100 else window.get("text", "")
            
            st.markdown(f"**{i+1}. {window_id}** - Score: {score:.4f}")
            st.markdown(f"Ch{chapter}, Pages {page_start}-{page_end}, {para_count} paragraphs")
            
            # Show all sections/paragraphs in this window
            paragraph_ids = window.get("paragraph_ids", [])
            if paragraph_ids:
                st.markdown("**Sections in this window:**")
                for para_id in paragraph_ids[:8]:  # Show up to 8 sections
                    # Try to get section info from the orchestrator's retriever
                    if hasattr(st.session_state.system['orchestrator'], 'retriever') and hasattr(st.session_state.system['orchestrator'].retriever, 'para_map'):
                        if para_id in st.session_state.system['orchestrator'].retriever.para_map:
                            para_data = st.session_state.system['orchestrator'].retriever.para_map[para_id]
                            para_num = para_data.get("para", "?")
                            version = para_data.get("version", "Base")
                            st.markdown(f"  • §{para_num} ({version})")
                        else:
                            st.markdown(f"  • {para_id}")
                    else:
                        st.markdown(f"  • {para_id}")
                
                if len(paragraph_ids) > 8:
                    st.markdown(f"  • ... and {len(paragraph_ids) - 8} more sections")
            
            st.markdown(f"*{text}*")
            st.markdown("---")
    
    # Reranked results
    with st.expander("🔄 Reranked Results", expanded=False):
        for i, window in enumerate(results['reranked'][:3]):  # Show top 3
            window_id = window.get("window_id", "Unknown")
            score_ce = window.get("score_ce", 0.0)
            original_score = window.get("score", 0.0)
            chapter = window.get("chapter", "?")
            page_start = window.get("page_start", "?")
            page_end = window.get("page_end", "?")
            para_count = len(window.get("paragraph_ids", []))
            text = window.get("text", "")[:100] + "..." if len(window.get("text", "")) > 100 else window.get("text", "")
            
            st.markdown(f"**{i+1}. {window_id}** - CE: {score_ce:.4f} (Orig: {original_score:.4f})")
            st.markdown(f"Ch{chapter}, Pages {page_start}-{page_end}, {para_count} paragraphs")
            
            # Show all sections/paragraphs in this window
            paragraph_ids = window.get("paragraph_ids", [])
            if paragraph_ids:
                st.markdown("**Sections in this window:**")
                for para_id in paragraph_ids[:6]:  # Show up to 6 sections for reranked
                    # Try to get section info from the orchestrator's retriever
                    if hasattr(st.session_state.system['orchestrator'], 'retriever') and hasattr(st.session_state.system['orchestrator'].retriever, 'para_map'):
                        if para_id in st.session_state.system['orchestrator'].retriever.para_map:
                            para_data = st.session_state.system['orchestrator'].retriever.para_map[para_id]
                            para_num = para_data.get("para", "?")
                            version = para_data.get("version", "Base")
                            st.markdown(f"  • §{para_num} ({version})")
                        else:
                            st.markdown(f"  • {para_id}")
                    else:
                        st.markdown(f"  • {para_id}")
                
                if len(paragraph_ids) > 6:
                    st.markdown(f"  • ... and {len(paragraph_ids) - 6} more sections")
            
            st.markdown(f"*{text}*")
            st.markdown("---")
    
    # Smart paragraph results
    with st.expander("📊 Smart Paragraphs", expanded=True):
        for i, para in enumerate(results['paragraphs']):
            chapter = para.get("chapter", "?")
            para_id = para.get("para", "?")
            version = para.get("version", "Base")
            score = para.get("score_ce", 0.0)
            para_id_full = para.get("id", "Unknown")
            text = para.get("text", "")[:150] + "..." if len(para.get("text", "")) > 150 else para.get("text", "")
            
            st.markdown(f"**{i+1}. Ch{chapter} §{para_id} ({version})** - Relevance: {score:.4f}")
            st.markdown(f"*ID: {para_id_full}*")
            st.markdown(f"*{text}*")
            st.markdown("---")


def display_transparency_info():
    """Display system transparency information."""
    st.markdown("### 🔍 System Transparency")
    
    with st.expander("How the System Works", expanded=False):
        st.markdown("""
        **Enhanced MARCH-PAWS Medical Assistant**
        
        This system uses advanced AI techniques to provide medical guidance:
        
        **🧠 Context Engineering**: Uses 45 carefully curated examples (5 per stage) to generate scenario-specific questions
        **📚 Hybrid Retrieval**: Combines keyword search (BM25) and semantic search (FAISS) for comprehensive content retrieval
        **🎯 Cross-Encoder Reranking**: Uses transformer models to ensure only relevant medical content is used
        **✅ Quality Evaluation**: Real-time assessment of question quality, citation accuracy, and checklist relevance
        **🚫 Smart Refusal**: Automatically detects and refuses non-medical queries
        
        **Quality Metrics Explained:**
        - **Question Quality**: Measures how well the question matches the medical scenario and stage requirements
        - **Citation Accuracy**: Validates that all citations exist in the medical database
        - **Checklist Quality**: Evaluates how actionable and relevant the recommendations are
        - **Overall Score**: Weighted combination of all quality factors
        """)
    
    with st.expander("Technical Details", expanded=False):
        st.markdown("""
        **System Architecture:**
        - **Async Orchestrator**: Handles multiple operations in parallel for faster responses
        - **WordNet + SpaCy**: Advanced natural language processing for verb detection and semantic analysis
        - **Sigmoid Normalization**: Ensures consistent scoring across different evaluation criteria
        - **LRU Caching**: Optimizes performance by caching frequent operations
        - **Database Mapping**: Converts citations to standardized medical reference format
        
        **Performance Optimizations:**
        - Parallel embedding generation during ingestion
        - Optimized FAISS index with memory mapping
        - Lightweight lemmatization for faster BM25 search
        - Background pre-fetching of next state content
        """)

def display_stage_info(stage):
    """Display information about the current MARCH-PAWS stage."""
    stage_info = {
        "M": {
            "name": "Massive Hemorrhage",
            "description": "Locate and control life-threatening bleeding",
            "color": "#dc3545"
        },
        "A": {
            "name": "Airway", 
            "description": "Ensure airway is patent and unobstructed",
            "color": "#fd7e14"
        },
        "R": {
            "name": "Respiration",
            "description": "Assess breathing and chest injuries",
            "color": "#ffc107"
        },
        "C": {
            "name": "Circulation",
            "description": "Assess perfusion and shock",
            "color": "#20c997"
        },
        "H": {
            "name": "Hypothermia/Head Injury",
            "description": "Prevent heat loss and assess head injury",
            "color": "#6f42c1"
        },
        "P": {
            "name": "Pain",
            "description": "Assess and manage pain",
            "color": "#e83e8c"
        },
        "A2": {
            "name": "Antibiotics",
            "description": "Determine need for antibiotics",
            "color": "#6c757d"
        },
        "W": {
            "name": "Wounds",
            "description": "Re-inspect for missed injuries",
            "color": "#17a2b8"
        },
        "S": {
            "name": "Splinting",
            "description": "Immobilize fractures and dislocations",
            "color": "#28a745"
        }
    }
    
    info = stage_info.get(stage, {"name": "Unknown", "description": "", "color": "#6c757d"})
    return info

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 MARCH-PAWS Medical Assistant - Enhanced</h1>', unsafe_allow_html=True)
    
    # Display transparency information
    display_transparency_info()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## About MARCH-PAWS")
        st.markdown("""
        **MARCH-PAWS** is a systematic approach to tactical combat casualty care:
        
        - **M** - Massive Hemorrhage
        - **A** - Airway  
        - **R** - Respiration
        - **C** - Circulation
        - **H** - Hypothermia/Head Injury
        - **P** - Pain
        - **A2** - Antibiotics
        - **W** - Wounds
        - **S** - Splinting
        """)
        
        st.markdown("## Instructions")
        st.markdown("""
        1. Enter a medical scenario
        2. Answer questions for each stage
        3. Follow the systematic assessment
        4. Get evidence-based recommendations
        """)
        
        # System status
        st.markdown("## System Status")
        if 'system' in st.session_state and st.session_state.system:
            st.success("✅ Enhanced System Ready")
            st.info("🚀 Async Orchestrator Active")
            st.info("📊 Quality Evaluation Enabled")
        else:
            st.error("❌ System Loading...")
        
    
    # Load enhanced system
    if 'system' not in st.session_state:
        with st.spinner("Loading enhanced MARCH-PAWS system..."):
            st.session_state.system = load_system()
    
    if not st.session_state.system:
        st.error("Failed to load enhanced MARCH-PAWS system. Please check the system configuration.")
        return
    
    # Initialize orchestrator if needed
    if not hasattr(st.session_state.system['orchestrator'], '_initialized'):
        with st.spinner("Initializing async orchestrator..."):
            asyncio.run(initialize_orchestrator(st.session_state.system['orchestrator']))
    
    # Initialize session state
    if 'scenario' not in st.session_state:
        st.session_state.scenario = ""
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = "M"
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'assessment_complete' not in st.session_state:
        st.session_state.assessment_complete = False
    if 'quality_metrics' not in st.session_state:
        st.session_state.quality_metrics = []
    
    # Main interface
    if not st.session_state.scenario:
        # Initial scenario input
        st.markdown("## Enter Medical Scenario")
        scenario = st.text_area(
            "Describe the medical situation:",
            placeholder="e.g., Soldier has a gunshot wound to the chest with visible bleeding",
            height=100
        )
        
        if st.button("Start MARCH-PAWS Assessment", type="primary"):
            if scenario.strip():
                st.session_state.scenario = scenario.strip()
                st.session_state.conversation_history = []
                st.session_state.assessment_complete = False
                # Ensure state machine starts from M on every new assessment
                if 'system' in st.session_state and st.session_state.system:
                    try:
                        st.session_state.system['orchestrator'].sm.reset()
                        st.session_state.current_stage = "M"
                    except Exception:
                        pass
                st.rerun()
            else:
                st.warning("Please enter a medical scenario.")
    
    else:
        # Display current scenario
        st.markdown(f'<div class="scenario-text"><strong>Scenario:</strong> {st.session_state.scenario}</div>', unsafe_allow_html=True)
        
        # Reset button
        if st.button("🔄 Start New Assessment"):
            st.session_state.scenario = ""
            st.session_state.conversation_history = []
            st.session_state.assessment_complete = False
            # Reset the orchestrator FSM as well
            if 'system' in st.session_state and st.session_state.system:
                try:
                    st.session_state.system['orchestrator'].sm.reset()
                except Exception:
                    pass
            st.rerun()
        
        # Process current stage
        if not st.session_state.assessment_complete:
            orc = st.session_state.system['orchestrator']
            evaluator = st.session_state.system['evaluator']
            
            # Get current stage info
            stage_info = display_stage_info(st.session_state.current_stage)
            
            # Display current stage
            st.markdown(f'<div class="stage-header" style="color: {stage_info["color"]};">{stage_info["name"]} Assessment</div>', unsafe_allow_html=True)
            st.markdown(f"*{stage_info['description']}*")
            
            # Process the stage
            if not st.session_state.conversation_history:
                # First interaction - get question
                with st.spinner("Analyzing scenario..."):
                    result = asyncio.run(orc.run_step(st.session_state.scenario))
                
                if result.get('refusal'):
                    st.error(f"❌ {result.get('message', 'Unable to process this scenario.')}")
                    st.info("Please try a different scenario or rephrase your description.")
                else:
                    question = result.get('question', '')
                    if question:
                        st.session_state.conversation_history.append({
                            'stage': st.session_state.current_stage,
                            'question': question,
                            'user_answer': None,
                            'checklist': [],
                            'citations': []
                        })
                        st.rerun()
            else:
                # Display current question
                current_entry = st.session_state.conversation_history[-1]
                if not current_entry['user_answer']:
                    st.markdown("### Current Question")
                    st.markdown(f'<div class="question-text">{current_entry["question"]}</div>', unsafe_allow_html=True)
                    
                    # User answer input
                    user_answer = st.text_area(
                        "Your answer:",
                        placeholder="Enter your assessment or response...",
                        height=100
                    )
                    
                    if st.button("Submit Answer", type="primary"):
                        if user_answer.strip():
                            # Process user answer
                            with st.spinner("Processing your answer..."):
                                result = asyncio.run(orc.run_step(st.session_state.scenario, user_answer.strip()))
                            
                            if result.get('refusal'):
                                st.error(f"❌ {result.get('message', 'Unable to process your answer.')}")
                            else:
                                # Update current entry
                                current_entry['user_answer'] = user_answer.strip()
                                current_entry['checklist'] = result.get('checklist', [])
                                current_entry['citations'] = result.get('citations', [])
                                
                                # Quality evaluation
                                mapped_citations = map_citations_to_database(
                                    current_entry['citations'], 
                                    evaluator.citation_data
                                )
                                
                                quality_eval = evaluator.evaluate_complete_response(
                                    current_entry['question'],
                                    current_entry['checklist'],
                                    mapped_citations,
                                    current_entry['user_answer'],
                                    current_entry['stage'],
                                    st.session_state.scenario
                                )
                                
                                current_entry['quality_evaluation'] = quality_eval
                                st.session_state.quality_metrics.append(quality_eval)
                                
                                # Check if there's a next question
                                next_question = result.get('question', '')
                                next_stage = result.get('question_state', '')
                                
                                if next_question and next_stage:
                                    # Add next stage entry
                                    st.session_state.conversation_history.append({
                                        'stage': next_stage,
                                        'question': next_question,
                                        'user_answer': None,
                                        'checklist': [],
                                        'citations': []
                                    })
                                    st.session_state.current_stage = next_stage
                                else:
                                    # Assessment complete
                                    st.session_state.assessment_complete = True
                                
                                st.rerun()
                        else:
                            st.warning("Please provide an answer.")
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.markdown("---")
            st.markdown("## Assessment History")
            
            for i, entry in enumerate(st.session_state.conversation_history):
                stage_info = display_stage_info(entry['stage'])
                
                with st.expander(f"Stage {i+1}: {stage_info['name']} ({entry['stage']})", expanded=True):
                    # Question
                    st.markdown(f'<div class="question-text"><strong>Question:</strong> {entry["question"]}</div>', unsafe_allow_html=True)
                    
                    # User answer
                    if entry['user_answer']:
                        st.markdown(f'<div class="scenario-text"><strong>Your Answer:</strong> {entry["user_answer"]}</div>', unsafe_allow_html=True)
                        
                        # Checklist
                        if entry['checklist']:
                            st.markdown("**Recommendations:**")
                            for item in entry['checklist']:
                                st.markdown(f'<div class="checklist-item">• {item}</div>', unsafe_allow_html=True)
                        
                        # Citations
                        if entry['citations']:
                            st.markdown("**References:**")
                            for citation in entry['citations']:
                                st.markdown(f'<div class="citation">• {citation}</div>', unsafe_allow_html=True)
                        
                        # Quality metrics
                        if 'quality_evaluation' in entry:
                            st.markdown("**Quality Assessment:**")
                            display_quality_metrics(entry['quality_evaluation'])
                    else:
                        st.info("Waiting for your answer...")
        
        # Assessment complete message
        if st.session_state.assessment_complete:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("## ✅ MARCH-PAWS Assessment Complete")
            st.markdown("All critical systems have been evaluated. Review the recommendations above and follow appropriate medical protocols.")
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Retrieval Testing Section (at bottom of page)
    st.markdown("---")
    st.markdown("## 🔍 Document Retrieval Testing")
    st.markdown("Test the document retrieval system with any medical scenario:")
    
    # Create two columns for the testing interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Input for retrieval testing
        retrieval_query = st.text_area(
            "Enter a medical scenario:",
            placeholder="e.g., chest wound with active bleeding\nsevere leg fracture due to accident\npatient with breathing difficulties",
            height=100,
            key="retrieval_query"
        )
        
        if st.button("🔍 Test Retrieval", key="test_retrieval_btn", type="primary"):
            if retrieval_query and 'system' in st.session_state and st.session_state.system:
                with st.spinner("Testing retrieval..."):
                    try:
                        # Test retrieval
                        retrieval_results = test_retrieval_for_ui(
                            st.session_state.system['orchestrator'], 
                            retrieval_query
                        )
                        st.session_state.retrieval_results = retrieval_results
                    except Exception as e:
                        st.error(f"Retrieval test failed: {str(e)}")
            else:
                st.warning("Please enter a medical scenario to test retrieval.")
    
    with col2:
        # Display retrieval results
        if 'retrieval_results' in st.session_state and st.session_state.retrieval_results:
            display_retrieval_results_main_page(st.session_state.retrieval_results)
        else:
            st.info("👆 Enter a medical scenario and click 'Test Retrieval' to see results here.")

if __name__ == "__main__":
    main()