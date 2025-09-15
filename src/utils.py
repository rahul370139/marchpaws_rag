"""Utility functions shared across modules.

Includes both sync and async utilities for:
1. Text formatting (catalog, excerpts)
2. Caching (cache keys)
3. Timing (performance monitoring)
4. Async operations (HTTP requests, blocking operations)
"""

import asyncio
import functools
import json
import os
import re
from typing import Any, Dict, List, Optional

import aiohttp


# =============================================================================
# LLM Configuration
# =============================================================================

DEFAULT_LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:11434")
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "mistral:latest")
DEFAULT_TEMPERATURE = 0.0  # Deterministic responses


# =============================================================================
# Text Formatting Functions
# =============================================================================

def format_catalog(chunks: List[Dict]) -> str:
    """Build a human‑readable catalog of the retrieved chunks using exact para IDs.

    Format example:
      Ch1 §1-19 | Section II — Vital Body Systems | COMPONENTS OF THE RESPIRATORY SYSTEM [p.15–15] (ID:...)
    """
    lines = []
    for chunk in chunks:
        chapter = chunk.get("chapter")
        para = chunk.get("para")
        section_full = chunk.get("section_full") or chunk.get("section") or ""
        heading = chunk.get("heading") or ""
        pages = f"p.{chunk.get('page_start')}–{chunk.get('page_end')}"
        label = f"Ch{chapter} §{para}"
        context = " | ".join([p for p in [section_full, heading] if p])
        if context:
            line = f"{label} | {context} [{pages}]  (ID:{chunk.get('id')})"
        else:
            line = f"{label} [{pages}]  (ID:{chunk.get('id')})"
        lines.append(line)
    return "\n".join(lines)


def get_smart_paragraphs_from_windows(windows: List[Dict], query: str, reranker, retriever, max_paragraphs: int = 10) -> List[Dict]:
    """Extract the most relevant paragraphs from windows using cross-encoder scoring.
    
    This function:
    1. Gets all available paragraphs from windows using the retriever's para_map
    2. Scores each paragraph using cross-encoder
    3. Returns the top-scoring paragraphs sorted by relevance
    
    Parameters
    ----------
    windows : List[Dict]
        List of windows from the retriever
    query : str
        The query to score paragraphs against
    reranker : CrossEncoderReranker
        The cross-encoder reranker instance
    retriever : HybridRetriever
        The retriever instance (needed to access para_map)
    max_paragraphs : int
        Maximum number of paragraphs to return
        
    Returns
    -------
    List[Dict]
        List of top-scoring paragraphs with cross-encoder scores
    """
    if not windows:
        return []
    
    # Get all paragraphs from windows using retriever's para_map
    all_paragraphs = []
    seen_para_ids = set()
    
    for window in windows:
        para_ids = window.get("paragraph_ids", [])
        for para_id in para_ids:
            if para_id not in seen_para_ids and para_id in retriever.para_map:
                seen_para_ids.add(para_id)
                para_data = retriever.para_map[para_id].copy()
                all_paragraphs.append(para_data)
    
    if not all_paragraphs:
        return []
    
    # Score paragraphs using cross-encoder
    scored_paragraphs = reranker.score_paragraphs(query, all_paragraphs)
    
    # Return top paragraphs
    return scored_paragraphs[:max_paragraphs]


def format_excerpts(chunks: List[Dict], max_chars_per_chunk: int = 600, show_multiple_citations: bool = True) -> str:
    """Format the retrieved chunks for inclusion in the prompt using exact para IDs.
    
    Injects canonical citation IDs to help LLM generate proper citations.
    Can show multiple citation hints for the most relevant paragraphs.

    Header format example:
      [Ch1 §1-19 | Section II — Vital Body Systems | COMPONENTS OF THE RESPIRATORY SYSTEM | p.15–15]
    """
    out_lines = []
    
    # Collect all citation hints from chunks
    citation_hints = []
    for chunk in chunks:
        chapter = chunk.get("chapter")
        para = chunk.get("para")
        version = chunk.get("version", "Base")  # Use actual version from data
        if chapter and para:
            canonical_id = f"Ch{chapter} §{para} ({version})"
            citation_hints.append(canonical_id)
    
    for chunk in chunks:
        chapter = chunk.get("chapter")
        para = chunk.get("para")
        section_full = chunk.get("section_full") or chunk.get("section") or ""
        heading = chunk.get("heading") or ""
        label = f"Ch{chapter} §{para}"
        parts = [label]
        if section_full:
            parts.append(section_full)
        if heading:
            parts.append(heading)
        parts.append(f"p.{chunk.get('page_start')}–{chunk.get('page_end')}")
        header = f"[{' | '.join(parts)}]"
        
        text = chunk.get("text", "")
        truncated = text[:max_chars_per_chunk] + " …" if len(text) > max_chars_per_chunk else text
        
        out_lines.append(header)
        out_lines.append(truncated)
        out_lines.append("")
    
    # Add citation hints section at the end
    if citation_hints and show_multiple_citations:
        # Show top 3 most relevant citations
        top_citations = citation_hints[:3]
        citation_section = "AVAILABLE CITATIONS: " + " | ".join(top_citations)
        out_lines.append(citation_section)
        out_lines.append("")
    
    return "\n".join(out_lines)


# =============================================================================
# Caching and Timing Functions
# =============================================================================

def create_cache_key(scenario: str, state: str) -> tuple:
    """Create a cache key for scenario-state combinations."""
    return (hash(scenario), state)


class AsyncTimer:
    """Context-manager that records async wall-time in seconds."""
    
    def __enter__(self):
        self.start = asyncio.get_event_loop().time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = asyncio.get_event_loop().time()
    
    @property
    def elapsed(self) -> float:
        return getattr(self, "end", 0.0) - getattr(self, "start", 0.0)


# =============================================================================
# Citation Mapping Functions
# =============================================================================

def map_citation_format(cite: str, citation_db: Dict[str, Any] = None) -> Optional[str]:
    """Map LLM citation format to database ID format.
    
    Converts: "Ch6 §6-1, p.35–35" → "tc4-02.1:ch6:6-1@Base" or "tc4-02.1:ch6:6-1@C2"
    Tries both @Base and @C2 versions to find the correct one in the database.
    
    Parameters
    ----------
    cite : str
        Citation in LLM format (e.g., "Ch6 §6-1, p.35–35")
    citation_db : Dict[str, Any], optional
        Citation database to check for existence of mapped IDs
        
    Returns
    -------
    Optional[str]
        Database ID format or None if mapping fails
    """
    if not cite:
        return None
    
    # Strip page numbers and other suffixes first
    clean_cite = re.sub(r",?\s*p\.\d+.*$", "", cite.strip())
    
    # Pattern to match: Ch<number> §<number-number> with optional (Base)
    pattern = r"Ch(\d+)\s*§(\d+-?\d*)"
    match = re.match(pattern, clean_cite)
    
    if not match:
        return None
    
    chapter, para = match.groups()
    
    # Try both @Base and @C2 versions
    base_id = f"tc4-02.1:ch{chapter}:{para}@Base"
    c2_id = f"tc4-02.1:ch{chapter}:{para}@C2"
    
    # If citation_db is provided, check which version exists
    if citation_db is not None:
        if base_id in citation_db:
            return base_id
        elif c2_id in citation_db:
            return c2_id
        else:
            # Neither exists, return the @Base version as default
            return base_id
    
    # If no database provided, return @Base as default
    return base_id


def map_citations_to_database(citations: List[str], citation_db: Dict[str, Any]) -> List[str]:
    """Map a list of citations to their database canonical forms.
    
    Parameters
    ----------
    citations : List[str]
        List of citations in LLM format
    citation_db : Dict[str, Any]
        Citation database mapping IDs to citation data
        
    Returns
    -------
    List[str]
        List of mapped citations using short_ref from database
    """
    mapped = []
    seen = set()
    
    for citation in citations:
        if not citation:
            continue
            
        # Try to map to database format
        db_id = map_citation_format(citation, citation_db)
        
        if db_id and db_id in citation_db:
            # Use the short_ref from database
            short_ref = citation_db[db_id].get('anchors', {}).get('short_ref', citation)
            
            # Re-enable deduplication with tolerance for ≤2 duplicates
            if short_ref not in seen or len([x for x in mapped if x == short_ref]) < 2:
                mapped.append(short_ref)
                seen.add(short_ref)
            else:
                print(f"⚠️  Duplicate citation detected (max 2 allowed): {short_ref}")
        else:
            # If mapping fails, DROP the citation instead of keeping "SCENARIO"
            print(f"⚠️  Could not map citation, dropping: {citation}")
            # Don't append anything - just drop it
    
    return mapped


# =============================================================================
# Async Utility Functions
# =============================================================================

async def run_blocking(func, *args, **kwargs):
    """Run a blocking function in a ThreadPoolExecutor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, 
        functools.partial(func, *args, **kwargs)
    )


async def post_json_async(
    prompt: str,
    endpoint: str = DEFAULT_LLM_ENDPOINT,
    model: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.25,
    timeout: int = 120
) -> Dict[str, Any]:
    """Send async HTTP request to LLM server."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": 4096,
        }
    }
    
    url = f"{endpoint.rstrip('/')}/api/generate"
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        try:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                
                response_text = ""
                async for line in response.content:
                    line_text = line.decode('utf-8').strip()
                    if line_text:
                        try:
                            chunk = json.loads(line_text)
                            if 'response' in chunk:
                                response_text += chunk['response']
                        except json.JSONDecodeError:
                            continue
                
                if response_text:
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError:
                        return {
                            "checklist": [response_text],
                            "citations": [],
                            "state_complete": True,
                            "question": ""
                        }
                else:
                    raise ValueError("Empty response from LLM server")
                    
        except aiohttp.ClientError as e:
            raise RuntimeError(f"HTTP request failed: {e}")
        except asyncio.TimeoutError:
            raise RuntimeError(f"Request timeout after {timeout}s")


async def post_question_async(
    prompt: str,
    endpoint: str = DEFAULT_LLM_ENDPOINT,
    model: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.0,
    timeout: int = 60
) -> str:
    """Send async HTTP request for question generation only."""
    try:
        response = await post_json_async(prompt, endpoint, model, temperature, timeout)
        
        if isinstance(response, str):
            return response.strip()
        elif isinstance(response, dict) and "question" in response:
            return response["question"].strip()
        else:
            response_str = str(response)
            return response_str.strip()
            
    except Exception as e:
        return f"Unable to formulate question: {str(e)}"


async def batch_retrieve_async(loop: asyncio.AbstractEventLoop, retriever, queries: list) -> list:
    """Run multiple retrieval operations in parallel."""
    tasks = [
        run_blocking(loop, retriever.search, query, None, 6)
        for query in queries
    ]
    return await asyncio.gather(*tasks)


async def batch_rerank_async(loop: asyncio.AbstractEventLoop, reranker, query: str, hits_list: list) -> list:
    """Run multiple cross-encoder reranking operations in parallel."""
    tasks = [
        run_blocking(loop, reranker.rerank, query, hits)
        for hits in hits_list
    ]
    return await asyncio.gather(*tasks)


def load_generic_paras(state: str) -> List[Dict]:
    """Load generic fallback paragraphs for when no specific content is found.
    
    Parameters
    ----------
    state : str
        MARCH-PAWS state (M, A, R, C, H, P, A2, W, S)
        
    Returns
    -------
    List[Dict]
        List of generic excerpt dictionaries with standard structure
    """
    generic_paras = {
        'M': [
            {
                "heading": "Massive Hemorrhage Assessment",
                "section": "Initial Assessment",
                "text": "Assess for massive bleeding. If no life-threatening bleeding is present, continue to airway assessment. Apply direct pressure to any active bleeding sites. If bleeding cannot be controlled with direct pressure, consider tourniquet application for extremity bleeding.",
                "score": 0.8,
                "score_ce": 0.7,
                "window_id": "generic_m_001",
                "source": "TCCC Guidelines"
            }
        ],
        'A': [
            {
                "heading": "Airway Assessment",
                "section": "Basic Airway Management", 
                "text": "Ensure airway is patent. Clear any obstructions. If patient is unconscious, use head-tilt chin-lift or jaw thrust maneuver. Monitor breathing continuously.",
                "score": 0.8,
                "score_ce": 0.7,
                "window_id": "generic_a_001",
                "source": "TCCC Guidelines"
            }
        ],
        'R': [
            {
                "heading": "Respiratory Assessment",
                "section": "Breathing Evaluation",
                "text": "Assess breathing rate and quality. Look for signs of respiratory distress. If breathing is inadequate, provide rescue breathing or ventilation support as needed.",
                "score": 0.8,
                "score_ce": 0.7,
                "window_id": "generic_r_001",
                "source": "TCCC Guidelines"
            }
        ],
        'C': [
            {
                "heading": "Circulation Assessment",
                "section": "Pulse and Circulation",
                "text": "Check pulse quality and rate. Assess skin color and temperature. Look for signs of shock. If no pulse, begin CPR immediately.",
                "score": 0.8,
                "score_ce": 0.7,
                "window_id": "generic_c_001",
                "source": "TCCC Guidelines"
            }
        ],
        'H': [
            {
                "heading": "Hypothermia Prevention",
                "section": "Environmental Protection",
                "text": "Assess for hypothermia risk. Protect patient from cold environment. Remove wet clothing and provide insulation. Monitor body temperature.",
                "score": 0.8,
                "score_ce": 0.7,
                "window_id": "generic_h_001",
                "source": "TCCC Guidelines"
            }
        ],
        'P': [
            {
                "heading": "Pain Management",
                "section": "Pain Assessment",
                "text": "Assess pain level if patient is conscious and responsive. Provide appropriate pain management. Document pain assessment findings.",
                "score": 0.8,
                "score_ce": 0.7,
                "window_id": "generic_p_001",
                "source": "TCCC Guidelines"
            }
        ],
        'A2': [
            {
                "heading": "Antibiotics and Allergies",
                "section": "Medication Safety",
                "text": "Check for known allergies before administering any medications. Administer antibiotics for penetrating wounds if no allergies present.",
                "score": 0.8,
                "score_ce": 0.7,
                "window_id": "generic_a2_001",
                "source": "TCCC Guidelines"
            }
        ],
        'W': [
            {
                "heading": "Wound Assessment",
                "section": "Secondary Survey",
                "text": "Perform secondary survey for additional wounds, burns, or injuries. Dress wounds appropriately. Look for hidden injuries.",
                "score": 0.8,
                "score_ce": 0.7,
                "window_id": "generic_w_001",
                "source": "TCCC Guidelines"
            }
        ],
        'S': [
            {
                "heading": "Splinting",
                "section": "Fracture Management",
                "text": "Assess for fractures or deformities. Apply appropriate splinting for suspected fractures. Immobilize injured extremities.",
                "score": 0.8,
                "score_ce": 0.7,
                "window_id": "generic_s_001",
                "source": "TCCC Guidelines"
            }
        ]
    }
    
    return generic_paras.get(state, [])