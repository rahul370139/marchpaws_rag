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

from orchestrator import Orchestrator

# Page configuration
st.set_page_config(
    page_title="MARCH-PAWS Medical Assistant",
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
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Ensure all text in expanders is black */
    .streamlit-expanderContent {
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
        background-color: #ffffff !important;
        background: #ffffff !important;
        background-image: none !important;
        border: 2px solid #28a745 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
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
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_orchestrator():
    """Load the MARCH-PAWS orchestrator with caching."""
    try:
        orc = Orchestrator(
            bm25_path='data/window_bm25_index.pkl',
            embeddings_path='data/window_embeddings.npy',
            metadata_path='data/window_metadata.json'
        )
        return orc
    except Exception as e:
        st.error(f"Failed to load MARCH-PAWS system: {e}")
        return None

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
    st.markdown('<h1 class="main-header">🏥 MARCH-PAWS Medical Assistant</h1>', unsafe_allow_html=True)
    
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
        if 'orchestrator' in st.session_state and st.session_state.orchestrator:
            st.success("✅ System Ready")
        else:
            st.error("❌ System Loading...")
    
    # Load orchestrator
    if 'orchestrator' not in st.session_state:
        with st.spinner("Loading MARCH-PAWS system..."):
            st.session_state.orchestrator = load_orchestrator()
    
    if not st.session_state.orchestrator:
        st.error("Failed to load MARCH-PAWS system. Please check the system configuration.")
        return
    
    # Initialize session state
    if 'scenario' not in st.session_state:
        st.session_state.scenario = ""
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = "M"
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'assessment_complete' not in st.session_state:
        st.session_state.assessment_complete = False
    
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
                if 'orchestrator' in st.session_state and st.session_state.orchestrator:
                    try:
                        st.session_state.orchestrator.sm.reset()
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
            if 'orchestrator' in st.session_state and st.session_state.orchestrator:
                try:
                    st.session_state.orchestrator.sm.reset()
                except Exception:
                    pass
            st.rerun()
        
        # Process current stage
        if not st.session_state.assessment_complete:
            orc = st.session_state.orchestrator
            
            # Get current stage info
            stage_info = display_stage_info(st.session_state.current_stage)
            
            # Display current stage
            st.markdown(f'<div class="stage-header" style="color: {stage_info["color"]};">{stage_info["name"]} Assessment</div>', unsafe_allow_html=True)
            st.markdown(f"*{stage_info['description']}*")
            
            # Process the stage
            if not st.session_state.conversation_history:
                # First interaction - get question
                with st.spinner("Analyzing scenario..."):
                    result = orc.run_step(st.session_state.scenario)
                
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
                                result = orc.run_step(st.session_state.scenario, user_answer.strip())
                            
                            if result.get('refusal'):
                                st.error(f"❌ {result.get('message', 'Unable to process your answer.')}")
                            else:
                                # Update current entry
                                current_entry['user_answer'] = user_answer.strip()
                                current_entry['checklist'] = result.get('checklist', [])
                                current_entry['citations'] = result.get('citations', [])
                                
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
                    else:
                        st.info("Waiting for your answer...")
        
        # Assessment complete message
        if st.session_state.assessment_complete:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("## ✅ MARCH-PAWS Assessment Complete")
            st.markdown("All critical systems have been evaluated. Review the recommendations above and follow appropriate medical protocols.")
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()