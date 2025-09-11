"""Paragraph segmentation and ID detection with proper continuation merging."""

import re
import hashlib
from cleaners import normalize_for_hashing

PARA_ID_RE = re.compile(r"^(?P<id>\d+-\d+)\.\s*(?P<body>.*)$")
# Sometimes digits get separated: "1 10." rather than "1-10."
LOST_DASH_RE = re.compile(r"^(\d)\s(\d{1,2})\.\s*(.*)$")
# Heading patterns for structural elements to skip
HEADING_RE = re.compile(r"^(chapter\s+\d+|section\s+[ivx]+|[A-Z0-9][A-Z0-9 \-—]+)$", re.I)

def sha1(s):
    """Generate SHA1 hash."""
    return hashlib.sha1(s.encode()).hexdigest()

def stitch_paragraphs_across_pages(paragraphs):
    """Stitch incomplete paragraphs across page boundaries."""
    if not paragraphs:
        return paragraphs
    
    stitched = []
    i = 0
    
    while i < len(paragraphs):
        para_id, text = paragraphs[i]
        
        # Check if this paragraph is incomplete (ends with conjunction and is short)
        is_incomplete = (
            len(text) < 200 and 
            re.search(r'\b(and|or|but|the|of|in|to|for|with|by)\s*$', text.strip(), re.I) and
            i + 1 < len(paragraphs)
        )
        
        if is_incomplete:
            next_para_id, next_text = paragraphs[i + 1]
            
            # Check if next paragraph is a continuation (starts with lowercase or is very short)
            if (re.match(r'^[a-z]', next_text.strip()) or 
                (len(next_text.strip()) < 50 and not re.match(r'^\d+-\d+\.', next_text.strip()))):
                # Merge the paragraphs
                merged_text = f"{text} {next_text}"
                stitched.append((para_id, merged_text))
                i += 2  # Skip both paragraphs
                continue
        
        # This hardcoded logic is now handled by the general algorithm in parse_tc4021.py
        # No special cases needed here
        
        stitched.append((para_id, text))
        i += 1
    
    return stitched

def paragraphs_from_page_blocks(text, current_chapter=None, page_blocks=None):
    """
    Merge all text after a numbered para 'X-Y.' into that para
    until next numbered para or a heading line (ALL CAPS or Section/Chapter).
    Now with y-coordinate tracking for accurate heading assignment.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    
    current = None
    out = []
    
    # Helper: find y0 for a paragraph id by scanning page lines (regex-based)
    def find_y0_for_para_id(para_id: str) -> float:
        if not page_blocks:
            return 0.0
        pid_pattern = re.compile(rf"^\s*{re.escape(para_id)}\s*\.\s*")
        for block in page_blocks:
            for line in block.get("lines", []):
                line_text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                if not line_text:
                    continue
                if pid_pattern.match(line_text):
                    return float(line.get("bbox", [0, 0, 0, 0])[1])
        return 0.0
    
    def flush():
        nonlocal current
        if current:
            # join and clean spaces, normalize bullets
            text = " ".join(current["lines"]).strip()
            text = normalize_bullets(text)
            current["text"] = re.sub(r"\s+", " ", text)
            # Include y-coordinate for heading assignment
            out.append((current["para"], current["text"], current.get("y0", 0)))
            current = None
    
    def normalize_bullets(text):
        """Normalize various bullet characters to standard dash."""
        # Replace various bullet characters with standard dash
        text = re.sub(r"[\uF0B7\u2022\u25CF\u25AA\u25E6\u2023\u2024•]+", "-", text)
        return text
    
    def is_leaked_heading(text):
        """Check if text ends with a leaked heading (ALL-CAPS standalone)."""
        # Look for ALL-CAPS words at the end that aren't WARNING/CAUTION/Note
        words = text.split()
        if len(words) >= 2:
            last_two = " ".join(words[-2:])
            if re.match(r"^[A-Z][A-Z\s]+$", last_two) and not re.match(r"^(WARNING|CAUTION|Note)", last_two):
                return True
        # Also check for patterns like "FIRST AID (self-aid AND BUDDY AID)"
        if re.search(r"\([A-Z\s]+\)$", text):
            return True
        # Check for ALL-CAPS headings at end (6+ chars)
        if re.search(r"\n[A-Z0-9 ()/\-]{6,}$", text):
            return True
        # Check for specific patterns that are clearly leaked headings
        if re.search(r"FIRST AID.*BUDDY AID", text, re.I):
            return True
        return False
    
    def split_leaked_heading(text):
        """Split text at leaked heading boundary."""
        words = text.split()
        if len(words) < 2:
            return text, ""
        
        # Find the split point (last ALL-CAPS sequence)
        for i in range(len(words) - 1, 0, -1):
            remaining = " ".join(words[i:])
            if re.match(r"^[A-Z][A-Z\s]+$", remaining) and not re.match(r"^(WARNING|CAUTION|Note)", remaining):
                return " ".join(words[:i]), remaining
        
        # Also check for patterns like "FIRST AID (self-aid AND BUDDY AID)"
        if re.search(r"\([A-Z\s]+\)$", text):
            # Find the last occurrence of this pattern
            match = re.search(r"(.+?)\s+([A-Z\s]+\([A-Z\s]+\))$", text)
            if match:
                return match.group(1).strip(), match.group(2).strip()
        
        return text, ""
    
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
            
        # Check for numbered paragraph
        m = PARA_ID_RE.match(s)
        if m:
            flush()  # Finish previous paragraph
            # Get y-coordinate for this paragraph using regex match
            y0 = find_y0_for_para_id(m.group("id"))
            current = {"para": m.group("id"), "lines": [m.group("body")], "y0": y0}
            continue
            
        # Check for heading patterns (structural elements to skip)
        if HEADING_RE.match(s):
            flush()  # Finish previous paragraph
            continue  # Skip structural headings
            
        # Check for lost dash repair
        if not m and current_chapter:
            m2 = LOST_DASH_RE.match(s)
            if m2 and int(m2.group(1)) == current_chapter:
                s = f"{m2.group(1)}-{m2.group(2)}. {m2.group(3)}"
                m = PARA_ID_RE.match(s)
                if m:
                    flush()
                    # Get y-coordinate for this paragraph using regex match
                    y0 = find_y0_for_para_id(m.group("id"))
                    current = {"para": m.group("id"), "lines": [m.group("body")], "y0": y0}
                    continue
        
        # Continuation line
        if current:
            # Check for leaked heading at the end
            if is_leaked_heading(s):
                content, leaked = split_leaked_heading(s)
                if content:
                    current["lines"].append(content)
                if leaked:
                    # Start new paragraph with leaked heading
                    flush()
                    # Try to extract paragraph ID from leaked heading
                    leaked_m = PARA_ID_RE.match(leaked)
                    if leaked_m:
                        y0 = line_to_y0.get(leaked, 0)
                        current = {"para": leaked_m.group("id"), "lines": [leaked_m.group("body")], "y0": y0}
                    else:
                        # If no paragraph ID, treat as orphan
                        current = None
            else:
                current["lines"].append(s)
        # else: orphan text before first para - ignore
    
    flush()
    return out


def hash_para(text):
    """Generate paragraph hash for deduplication."""
    return "sha1:" + sha1(normalize_for_hashing(text))