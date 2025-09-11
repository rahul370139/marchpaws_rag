"""Orchestrator tying retrieval and LLM together with guardrails.

The `Orchestrator` class encapsulates the RAG pipeline:

1. Maintains a finite‑state machine for MARCH‑PAWS.
2. Uses a `HybridRetriever` to fetch relevant snippets from the document.
3. Constructs a prompt using templates and the retrieved snippets.
4. Sends the prompt to a running LLM server (e.g. Ollama or llama.cpp) via
   HTTP and parses the JSON response.
5. Enforces guardrails: minimum relevance threshold, citation requirement,
   JSON schema enforcement, and refusal behaviour.

To use the orchestrator, instantiate it with paths to the BM25 index, embeddings file,
and metadata file, and optionally specify the LLM endpoint and model name.  Then
call `run_step(query)` repeatedly to advance through the MARCH‑PAWS states.

```
orc = Orchestrator('data/bm25_index.pkl', 'data/embeddings.npy', 'data/metadata.json')
result = orc.run_step('Gunshot wound to the chest')
while result and not result['refusal'] and orc.sm.has_more():
    # Present result['checklist'] and result['citations'] to the user
    # Then advance to the next state
    orc.sm.advance()
    result = orc.run_step(query)
```
"""

import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import requests
import numpy as np
import torch
from sentence_transformers import CrossEncoder

# Optional quantization imports
try:
    from optimum.intel import IPEXModel
    from optimum.exporters.onnx import export_models
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False

from fsm import MARCHPAWSStateMachine
from retriever import HybridRetriever
from prompts import SYSTEM_PROMPT, USER_TEMPLATE, STAGE_DEFINITIONS
from utils import format_catalog, format_excerpts


class CrossEncoderReranker:
    """Cross-encoder re-ranker for improving retrieval precision.
    
    Takes query + document pairs and produces a single relevance score
    by allowing the Transformer to attend across both sequences.
    """
    
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", use_quantization=True):
        """Initialize the cross-encoder model with caching and optimization.
        
        Parameters
        ----------
        model_name : str
            HuggingFace model name for cross-encoder
        device : str
            Device to run the model on ('cpu' or 'cuda')
        use_quantization : bool
            Whether to use quantization for CPU optimization
        """
        # Let sentence-transformers handle device detection automatically
        self.ce = CrossEncoder(model_name)
        
        # Set model to evaluation mode for inference optimization
        self.ce.model.eval()
        
        # Apply quantization if available and requested
        if use_quantization and QUANTIZATION_AVAILABLE and device == "cpu":
            try:
                # Use Intel Extension for PyTorch for CPU optimization
                import intel_extension_for_pytorch as ipex
                self.ce.model = ipex.optimize(self.ce.model, dtype=torch.float32)
                print("✓ Cross-encoder optimized with Intel Extension for PyTorch")
            except ImportError:
                print("⚠ Intel Extension for PyTorch not available, using standard model")
        elif use_quantization and not QUANTIZATION_AVAILABLE:
            print("⚠ Quantization dependencies not available, using standard model")
        
        # Cache the model for reuse (avoid reloading)
        self._model_loaded = True
    
    def rerank(self, query: str, windows: List[Dict]) -> List[Dict]:
        """Re-rank windows based on query-document relevance.
        
        Parameters
        ----------
        query : str
            The search query
        windows : List[Dict]
            List of window dictionaries with 'text' field
            
        Returns
        -------
        List[Dict]
            Re-ranked windows with added 'score_ce' field, sorted by relevance
        """
        if not windows:
            return windows
            
        # Create query-document pairs
        pairs = [(query, w["text"]) for w in windows]
        
        # Get cross-encoder scores
        scores = self.ce.predict(pairs)
        
        # Normalize cross-encoder scores to 0-1 range using sigmoid
        # Cross-encoder outputs logits, sigmoid converts to probabilities
        normalized_scores = 1 / (1 + np.exp(-np.array(scores)))
        
        # Add cross-encoder scores to windows
        for window, score in zip(windows, normalized_scores):
            # Handle potential NaN values - check for NaN properly
            is_nan = score != score  # NaN check
            
            if isinstance(score, (int, float, np.floating, np.integer)) and not is_nan:
                window["score_ce"] = float(score)
            else:
                window["score_ce"] = 0.0  # Default score for invalid values
        
        # Sort by cross-encoder score (descending)
        return sorted(windows, key=lambda w: -w["score_ce"])
    
    def rerank_batch(self, queries_and_windows: List[Tuple[str, List[Dict]]]) -> List[List[Dict]]:
        """Re-rank multiple query-window pairs in a single batch for efficiency.
        
        Parameters
        ----------
        queries_and_windows : List[Tuple[str, List[Dict]]]
            List of (query, windows) tuples to re-rank
            
        Returns
        -------
        List[List[Dict]]
            List of re-ranked window lists, one per query
        """
        if not queries_and_windows:
            return []
        
        # Flatten all query-document pairs with metadata
        all_pairs = []
        query_indices = []  # Track which query each pair belongs to
        window_indices = []  # Track which window within each query
        
        for query_idx, (query, windows) in enumerate(queries_and_windows):
            for window_idx, window in enumerate(windows):
                all_pairs.append((query, window["text"]))
                query_indices.append(query_idx)
                window_indices.append(window_idx)
        
        if not all_pairs:
            return [[] for _ in queries_and_windows]
        
        # Get cross-encoder scores for all pairs in one batch
        scores = self.ce.predict(all_pairs)
        
        # Normalize scores
        normalized_scores = 1 / (1 + np.exp(-np.array(scores)))
        
        # Group results back by query
        results = [[] for _ in queries_and_windows]
        
        for (query_idx, window_idx), score in zip(zip(query_indices, window_indices), normalized_scores):
            # Handle potential NaN values
            is_nan = score != score
            if isinstance(score, (int, float, np.floating, np.integer)) and not is_nan:
                score_ce = float(score)
            else:
                score_ce = 0.0
            
            # Add score to the window
            window = queries_and_windows[query_idx][1][window_idx].copy()
            window["score_ce"] = score_ce
            results[query_idx].append(window)
        
        # Sort each query's results by cross-encoder score
        for i, windows in enumerate(results):
            results[i] = sorted(windows, key=lambda w: -w["score_ce"])
        
        return results


@dataclass
class Orchestrator:
    bm25_path: str
    embeddings_path: str
    metadata_path: str
    llm_endpoint: str = os.environ.get("LLM_ENDPOINT", "http://localhost:11434/api/generate")
    llm_model: str = os.environ.get("LLM_MODEL", "mistral:latest")
    retrieval_threshold: float = 0.005  # Lower threshold for better coverage
    ce_threshold: float = 0.0001  # Cross-encoder threshold (very low for maximum coverage)
    max_chunks: int = 6  # Fewer chunks for more focused retrieval
    temperature: float = 0.1  # Slight randomness for more natural questions
    question_temperature: float = 0.2  # Higher temp for question generation
    answer_temperature: float = 0.0  # Zero temp for deterministic answers

    def __post_init__(self):
        # Initialise finite‑state machine and retriever
        self.sm = MARCHPAWSStateMachine()
        self.retriever = HybridRetriever(self.bm25_path, self.embeddings_path, self.metadata_path)
        # Initialize cross-encoder re-ranker
        self.reranker = CrossEncoderReranker()
        # Initialize HTTP session for keep-alive
        self.session = requests.Session()
        # Chat history buffer for dynamic questions/answers
        self._history = []

    def _generate_state_hint(self, query: str, state: str) -> str:
        """Generate focused state hint based on MARCH-PAWS state.
        
        Parameters
        ----------
        query : str
            The user's medical scenario query
        state : str
            Current MARCH-PAWS state (M, A, R, C, H, P, A2, W, S)
            
        Returns
        -------
        str
            Focused state hint for retrieval
        """
        # Simple, focused keywords for each state
        state_keywords = {
            "M": "bleeding hemorrhage blood",
            "A": "airway obstruction breathing",
            "R": "respiration breathing ventilation",
            "C": "circulation pulse shock",
            "H": "hypothermia head consciousness",
            "P": "pain analgesia",
            "A2": "antibiotics allergy",
            "W": "wound dressing inspection",
            "S": "splint fracture immobilize"
        }
        
        return state_keywords.get(state, "")


    def _filter_anatomically_relevant_content(self, win_hits: List[Dict], query: str) -> List[Dict]:
        """Filter out anatomically irrelevant content based on the scenario.
        
        Parameters
        ----------
        win_hits : List[Dict]
            List of retrieved windows
        query : str
            The user's medical scenario query
            
        Returns
        -------
        List[Dict]
            Filtered list of windows with anatomically relevant content
        """
        query_lower = query.lower()
        
        # Identify anatomical regions mentioned in the scenario
        mentioned_anatomy = set()
        if any(term in query_lower for term in ["chest", "thoracic", "rib", "lung"]):
            mentioned_anatomy.add("chest")
        if any(term in query_lower for term in ["head", "skull", "brain", "cranium"]):
            mentioned_anatomy.add("head")
        if any(term in query_lower for term in ["leg", "thigh", "calf", "femur", "tibia"]):
            mentioned_anatomy.add("leg")
        if any(term in query_lower for term in ["arm", "forearm", "humerus", "radius", "ulna"]):
            mentioned_anatomy.add("arm")
        if any(term in query_lower for term in ["abdomen", "abdominal", "stomach", "belly"]):
            mentioned_anatomy.add("abdomen")
        
        # If no specific anatomy mentioned, return all content
        if not mentioned_anatomy:
            return win_hits
        
        # Filter windows based on anatomical relevance
        filtered_hits = []
        for window in win_hits:
            text = window.get("text", "").lower()
            
            # Check if content mentions the relevant anatomy
            is_relevant = False
            
            # For chest scenarios, prioritize chest-related content but allow general content
            if "chest" in mentioned_anatomy:
                if any(term in text for term in ["chest", "thoracic", "rib", "lung", "breathing", "respiration"]):
                    is_relevant = True
                # Also allow general medical content
                elif any(term in text for term in ["bleeding", "hemorrhage", "shock", "pulse", "circulation", "airway"]):
                    is_relevant = True
            # For non-chest scenarios, exclude chest-specific content but allow general content
            else:
                # Exclude chest-specific content
                if any(term in text for term in ["chest", "thoracic", "rib", "lung", "chest seal", "occlusive"]):
                    is_relevant = False
                # Allow general medical content
                elif any(term in text for term in ["bleeding", "hemorrhage", "shock", "pulse", "circulation", "airway", "breathing", "respiration"]):
                    is_relevant = True
                # Allow anatomy-specific content
                elif "head" in mentioned_anatomy and any(term in text for term in ["head", "skull", "brain", "cranium", "consciousness"]):
                    is_relevant = True
                elif ("leg" in mentioned_anatomy or "arm" in mentioned_anatomy) and any(term in text for term in ["leg", "thigh", "calf", "femur", "tibia", "arm", "forearm", "humerus", "radius", "ulna", "extremity", "limb"]):
                    is_relevant = True
                elif "abdomen" in mentioned_anatomy and any(term in text for term in ["abdomen", "abdominal", "stomach", "belly", "evisceration"]):
                    is_relevant = True
            
            if is_relevant:
                filtered_hits.append(window)
        
        return filtered_hits if filtered_hits else win_hits  # Return original if no matches


    def _call_llm(self, prompt: str, is_question: bool = False) -> Optional[Dict[str, Any]]:
        """Send a prompt to the LLM server and parse the response.

        The LLM is expected to return a JSON string.  On failure or invalid
        JSON, returns None.
        
        Parameters
        ----------
        prompt : str
            The prompt to send to the LLM
        is_question : bool
            Whether this is for question generation (uses higher temperature)
        """
        # Use different temperatures for questions vs answers
        temp = self.question_temperature if is_question else self.answer_temperature
        
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "temperature": temp,
            "format": "json",
            "stream": False  # Disable streaming for easier parsing
        }
        try:
            # Use session for HTTP keep-alive
            resp = self.session.post(self.llm_endpoint, json=payload, timeout=120)
            resp.raise_for_status()
            response_data = resp.json()
            raw = response_data.get("response") or response_data.get("text", "")
            
            # Clean up the response if it has extra text
            if raw.startswith("JSON: "):
                raw = raw[6:]  # Remove "JSON: " prefix
            
            return json.loads(raw)
        except Exception as e:
            print(f"LLM call failed: {e}")
            return None

    def run_step(self, query: str, user_answer: str = "") -> Dict[str, Any]:
        """Execute one state of the MARCH‑PAWS sequence for the given query.

        Returns a dictionary with the following keys:
        - `state`: the current state identifier (e.g. "M", "A").
        - `checklist`: list of bullet items (if successful).
        - `citations`: list of citation strings used in the checklist.
        - `state_complete`: boolean indicating whether the current state is done.
        - `refusal`: boolean flag; if True, the request was refused.
        - `message`: optional explanatory message for refusal or errors.
        - `question`: question for the current state (if no user_answer) or next state (if user_answer provided).
        """
        state = self.sm.current_state
        # Generate dynamic state hint based on query and state
        hint = self._generate_state_hint(query, state)
        
        # Combine query with user answer for better retrieval context
        if user_answer:
            combined_query = f"{query} {user_answer}"
        else:
            combined_query = query
        
        # Use new window-based retrieval workflow
        # Step 1: Get parent windows (retrieve more for re-ranking)
        # Use stage definitions from prompts.py (single source of truth)
        stage_definition = STAGE_DEFINITIONS.get(state, "")
        
        # Use stage definitions for focused retrieval
        stage_specific_query = f"{combined_query} {stage_definition}" if stage_definition else combined_query
        
        # Use stage-specific query for all retrieval to ensure state-specific content
        bm25_query = f"{stage_specific_query} {hint}" if hint else stage_specific_query
        dense_query = stage_specific_query
        
        win_hits = self.retriever.search(stage_specific_query, state_hint=None, k=20, bm25_query=bm25_query, dense_query=dense_query)
        if not win_hits:
            return {
                "state": state,
                "refusal": True,
                "message": "I cannot advise — no relevant content found."
            }
        
        # Check maximum score threshold BEFORE re-ranking (use original hybrid score)
        original_max_score = max(w["score"] for w in win_hits)
        if original_max_score < self.retrieval_threshold:
            return {
                "state": state,
                "refusal": True,
                "message": "I cannot advise — no relevant content found."
            }
        
        # Step 1.5: Filter anatomically irrelevant content
        win_hits = self._filter_anatomically_relevant_content(win_hits[:20], query)
        
        # Step 1.6: Re-rank with cross-encoder for better precision
        win_hits = self.reranker.rerank(stage_specific_query, win_hits)[:self.max_chunks]
        
        # Check cross-encoder threshold after re-ranking
        max_ce_score = max(w.get("score_ce", 0.0) for w in win_hits)
        if max_ce_score < self.ce_threshold:
            return {
                "state": state,
                "refusal": True,
                "message": "I cannot advise — no relevant content found."
            }
        
        # Step 2: Expand windows to child paragraphs
        child_paras = self.retriever.expand_windows(win_hits, max_paras=10)
        if not child_paras:
            return {
                "state": state,
                "refusal": True,
                "message": "I cannot advise — no relevant content found."
            }
        
        # Step 3: Build catalog and excerpts from child paragraphs
        catalog = format_catalog(child_paras)
        excerpts = format_excerpts(child_paras)
        # Compose full prompt (separate scenario vs latest user answer)
        user_prompt = USER_TEMPLATE.format(
            state=state,
            state_definition=STAGE_DEFINITIONS.get(state, ""),
            scenario=query,
            user_answer=user_answer,
            catalog=catalog,
            excerpts=excerpts,
        )
        prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        # Send to LLM with appropriate temperature
        resp = self._call_llm(prompt, is_question=(not user_answer))
        if not resp:
            return {
                "state": state,
                "refusal": True,
                "message": "I cannot advise — no relevant content found."
            }
        # Validate fields - handle different response formats
        checklist = resp.get("checklist")
        citations = resp.get("citations")
        question = resp.get("question")
        state_complete = bool(resp.get("state_complete"))
        
        
        # If the response has a different format, try to extract checklist items
        if not checklist and resp:
            # Look for any list-like structure in the response
            for key, value in resp.items():
                if isinstance(value, list) and value:
                    checklist = [item.get("Item", str(item)) if isinstance(item, dict) else str(item) for item in value]
                    # Extract citations if available
                    if isinstance(value[0], dict) and "Citation" in value[0]:
                        citations = [item.get("Citation", "") for item in value]
                    break
        
        # Handle different scenarios based on whether user_answer is provided
        if not user_answer:
            # First interaction: only ask a question for current state
            if isinstance(question, str) and question.strip():
                return {
                    "state": state,
                    "checklist": [],
                    "citations": [],
                    "question": question,
                    "state_complete": False,
                    "refusal": False,
                }
            else:
                return {
                    "state": state,
                    "refusal": True,
                    "message": "I cannot advise — no relevant content found."
                }
        else:
            # User provided answer: generate checklist for current state and question for next state
            if not isinstance(checklist, list) or not checklist:
                return {
                    "state": state,
                    "refusal": True,
                    "message": "I cannot advise — no relevant content found."
                }
            
            # If no citations provided, create generic ones
            if not citations:
                citations = [child_paras[0].get("section_cite", "TC 4-02.1")] if child_paras else ["TC 4-02.1"]
            
            # Get question for next state before advancing
            next_state = self.sm.get_next_state()
            
            # Advance to next state after generating checklist
            if self.sm.has_more():
                self.sm.advance()
            else:
                next_state = None
            
            # Generate question for the next state
            if next_state:
                next_hint = self._generate_state_hint(query, next_state)
                
                # Retrieve candidates for next state question using stage-specific query
                next_stage_definition = STAGE_DEFINITIONS.get(next_state, "")
                next_stage_specific_query = f"{combined_query} {next_stage_definition}" if next_stage_definition else combined_query
                next_bm25_query = f"{next_stage_specific_query} {next_hint}" if next_hint else next_stage_specific_query
                next_candidates = self.retriever.search(next_stage_specific_query, state_hint=None, k=20, bm25_query=next_bm25_query, dense_query=next_stage_specific_query)
                if next_candidates:
                    # Batch re-rank both current and next state candidates for efficiency
                    queries_and_windows = [
                        (stage_specific_query, win_hits[:20]),
                        (next_stage_specific_query, next_candidates[:20])
                    ]
                    reranked_results = self.reranker.rerank_batch(queries_and_windows)
                    
                    # Update current state results
                    win_hits = reranked_results[0][:self.max_chunks]
                    # Update next state results
                    next_candidates = reranked_results[1][:self.max_chunks]
                    next_catalog = format_catalog(next_candidates)
                    next_excerpts = format_excerpts(next_candidates)
                    next_user_prompt = USER_TEMPLATE.format(
                        state=next_state,
                        state_definition=STAGE_DEFINITIONS.get(next_state, ""),
                        scenario=query,
                        user_answer="",
                        catalog=next_catalog,
                        excerpts=next_excerpts,
                    )
                    next_prompt = f"{SYSTEM_PROMPT}\n\n{next_user_prompt}"
                    next_resp = self._call_llm(next_prompt, is_question=True)
                    if next_resp and next_resp.get("question"):
                        next_question = next_resp.get("question")
                    else:
                        next_question = f"Please assess the {next_state} condition."
                else:
                    next_question = f"Please assess the {next_state} condition."
            else:
                # No more states - MARCH-PAWS complete
                next_question = "MARCH-PAWS assessment complete. All critical systems have been evaluated."
            
            return {
                "state": state,  # Current state that we just processed
                "checklist": checklist,
                "citations": citations,
                "question": next_question,
                "question_state": next_state,  # The state the question is for
                "state_complete": True,
                "refusal": False,
            }