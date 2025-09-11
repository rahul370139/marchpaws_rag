"""Section anchoring system for stable, user-friendly citations."""

import re
import fitz
from typing import List, Dict, Optional, Tuple
from heading_discovery import build_heading_catalog, get_heading_stack, get_page_headings, HeadingCandidate


def sanitize_chain(chain: List[str]) -> List[str]:
    """Clean up heading chain by removing empty entries and normalizing."""
    # Filter out empty or None entries
    filtered = [seg for seg in chain if seg and seg.strip()]
    
    # Basic normalization
    normalized = []
    for seg in filtered:
        # Convert to title case if it's all caps
        if seg.isupper() and len(seg) > 3:
            seg = seg.title()
        normalized.append(seg)
    
    return normalized

def get_proper_heading_chain(heading_catalog: Dict[str, HeadingCandidate], page_num: int, y0: float) -> List[str]:
    """Get proper heading chain using actual document headings and y-coordinate."""
    # Get the heading stack for this position
    stack = get_heading_stack(heading_catalog, page_num, y0)
    
    # Extract text from the stack
    chain = [heading.text for heading in stack]
    
    # Sanitize and return
    return sanitize_chain(chain)
