"""Heading discovery system for extracting actual document headings."""

import re
import hashlib
from typing import List, Dict, Tuple, Optional
import fitz
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class HeadingCandidate:
    """A discovered heading candidate."""
    text: str
    normalized_text: str
    page_num: int
    y0: float
    font_size: float
    is_all_caps: bool
    is_title_case: bool
    has_dot_leader: bool
    level: Optional[int] = None
    hID: Optional[str] = None

def _normalize(s: str) -> str:
    """Normalize text for comparison."""
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.replace('\xad', '')  # soft hyphen
    s = s.replace('—', '-').replace('–', '-')  # normalize dashes
    return s

def _locate_heading_on_page(page: fitz.Page, title_norm: str):
    """Find the first line on the page whose normalized text contains (or ≈) the ToC title."""
    d = page.get_text("dict")  # spans with real font sizes
    best = None
    for b in d["blocks"]:
        for l in b.get("lines", []):
            text = _normalize("".join(sp["text"] for sp in l["spans"]))
            if not text: 
                continue
            # strict contains first; fallback to token overlap ≥ 0.7
            if title_norm.lower() in text.lower():
                y0 = l["bbox"][1]
                font_max = max(sp["size"] for sp in l["spans"])
                return (y0, font_max)
            # loose match
            toks_t = set(re.findall(r'\w+', title_norm.lower()))
            toks_l = set(re.findall(r'\w+', text.lower()))
            if toks_t and len(toks_t & toks_l)/len(toks_t) >= 0.7:
                y0 = l["bbox"][1]
                font_max = max(sp["size"] for sp in l["spans"])
                best = best or (y0, font_max)
    return best  # may be None

def discover_toc_headings(doc: fitz.Document) -> List[HeadingCandidate]:
    """Use doc.get_toc(simple=True) and pin each ToC entry to a (page,y0)."""
    candidates = []
    toc = doc.get_toc(simple=True)  # [[level, title, page], ...]
    for level, title, page1 in toc:
        if not title or page1 <= 0: 
            continue
        page_idx = page1 - 1
        title_norm = _normalize(title)
        # skip obvious non-content
        if re.search(r'(index|glossary|references|contents)\b', title_norm, re.I):
            continue
        y0_font = _locate_heading_on_page(doc[page_idx], title_norm)
        if not y0_font:
            # If we cannot locate the title on the target page, skip to avoid polluting earlier pages
            continue
        y0, font_max = y0_font
        candidates.append(HeadingCandidate(
            text=title,
            normalized_text=normalize_heading_text(title),
            page_num=page_idx,
            y0=y0,
            font_size=font_max,
            is_all_caps=title.isupper(),
            is_title_case=title.istitle(),
            has_dot_leader=False,
            level=level
        ))
    return candidates

def discover_page_headings(doc: fitz.Document) -> List[HeadingCandidate]:
    """Pull likely headings by large-span lines and patterns."""
    candidates = []
    # compute global size distribution for thresholds
    sizes = []
    for p in doc:
        d = p.get_text("dict")
        for b in d["blocks"]:
            for l in b.get("lines", []):
                for sp in l["spans"]:
                    sizes.append(sp["size"])
    if not sizes:
        return candidates
    sizes.sort()
    large = sizes[int(0.80*len(sizes))]  # top 20% sizes threshold
    medium = sizes[int(0.60*len(sizes))]  # top 40% sizes threshold for short leaves

    HEADER_H = 70.0
    FOOTER_H = 90.0
    for page_num, page in enumerate(doc):
        d = page.get_text("dict")
        page_rect = page.rect
        for b in d["blocks"]:
            for l in b.get("lines", []):
                txt = _normalize("".join(sp["text"] for sp in l["spans"]))
                if not txt or len(txt) < 3:
                    continue
                y0 = l["bbox"][1]
                font_max = max(sp["size"] for sp in l["spans"])

                # Skip header/footer bands (running headers/footers)
                if y0 < (page_rect.y0 + HEADER_H) or y0 > (page_rect.y1 - FOOTER_H):
                    continue

                # patterns that really appear in this doc
                # ignore lines that are clearly paragraph ids or note/warning labels
                if re.match(r'^\d+\s*[-–]\s*\d+\.', txt) or re.match(r'^(note|warning|caution)\b', txt, re.I):
                    continue
                toks = txt.split()
                num_toks = len(toks)
                is_heading = (
                    re.match(r'^Chapter\s+\d+\b', txt, re.I) or
                    re.match(r'^Section\s+[IVXLC]+\s+[-—]', txt, re.I) or
                    (re.match(r'^[A-Z][A-Z\s]{6,}$', txt) and num_toks >= 2) or  # ALL CAPS multi-word
                    (font_max >= large and 1 <= num_toks <= 8 and
                     not txt.endswith('.') and not txt.endswith(',') and not txt.endswith(':') and
                     not re.search(r'\b(available|personnel|training|decisions|injured|sick|stages)\b', txt, re.I))
                )
                # Allow short Title-Case leaves at a lower size threshold; also allow mid-length leaves (<=6 tokens)
                if (not is_heading and 1 <= num_toks <= 6 and toks[0][0].isupper() and
                    not txt.endswith('.') and not txt.endswith(',') and not txt.endswith(':')):
                    is_heading = True

                # Additional filtering for noise and running headers with page markers
                if (is_heading and 
                    not re.search(r'intentionally left blank', txt, re.I) and
                    not re.search(r'^\w+\s*:$', txt) and  # Filter out "word:" patterns
                    not re.search(r'\b\d{1,2}\s*[-–]\s*\d{1,2}\b', txt) and  # running header page markers like 23-1
                    len(txt) > 5):  # Minimum length
                    candidates.append(HeadingCandidate(
                        text=txt,
                        normalized_text=normalize_heading_text(txt),
                        page_num=page_num,
                        y0=y0,
                        font_size=font_max,
                        is_all_caps=txt.isupper(),
                        is_title_case=txt.istitle(),
                        has_dot_leader=False
                    ))
    return candidates

def normalize_heading_text(text: str) -> str:
    """Normalize heading text for comparison."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove soft hyphens
    text = text.replace('\xad', '')
    # Normalize dashes
    text = re.sub(r'[–—]', '-', text)
    # Remove extra punctuation
    text = re.sub(r'[^\w\s\-]', '', text)
    return text.lower()

def compute_heading_id(text: str) -> str:
    """Compute stable ID for heading."""
    normalized = normalize_heading_text(text)
    return hashlib.sha1(normalized.encode()).hexdigest()[:12]

def infer_heading_levels(candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
    """Infer heading levels based on patterns and layout."""
    # prefer explicit signals, then size-based
    for c in candidates:
        t = c.text
        if re.match(r'^Chapter\s+\d+\b', t, re.I):
            c.level = 1
        elif re.match(r'^Section\s+[IVXLC]+\s+[-—]', t, re.I):
            c.level = 2
        else:
            # treat short ALL-CAPS or very large text as level 3
            if c.is_all_caps and len(t.split()) <= 8:
                c.level = 3
            elif c.font_size > 0:
                c.level = 3
            else:
                c.level = None
        c.hID = compute_heading_id(c.text)
    # drop Nones
    return [c for c in candidates if c.level is not None]

def deduplicate_headings(candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
    """Remove near-duplicate headings while preserving page-local instances.

    Keep at most one candidate per (normalized_text, page_num), chosen by font size then y-position.
    This allows short leaves (e.g., "Lungs") to persist on each page where they appear.
    """
    groups = defaultdict(list)
    for c in candidates:
        key = (c.normalized_text, c.page_num)
        groups[key].append(c)
    uniq = []
    for g in groups.values():
        # Best = larger font, higher on page (smaller y0)
        g.sort(key=lambda x: (-x.font_size, x.y0))
        uniq.append(g[0])
    return uniq

def build_heading_catalog(doc: fitz.Document) -> Dict[str, HeadingCandidate]:
    """Build the complete heading catalog for a document."""
    # Discover headings from TOC
    toc_headings = discover_toc_headings(doc)
    
    # Discover headings from page layout
    page_headings = discover_page_headings(doc)
    
    # Combine and deduplicate
    all_candidates = toc_headings + page_headings
    unique_candidates = deduplicate_headings(all_candidates)
    
    # Infer levels
    leveled_candidates = infer_heading_levels(unique_candidates)
    
    # Build catalog by hID
    catalog = {}
    for candidate in leveled_candidates:
        catalog[candidate.hID] = candidate
    
    return catalog

def find_nearest_heading(catalog: Dict[str, HeadingCandidate], page_num: int, y0: float) -> Optional[HeadingCandidate]:
    """Find the nearest heading above a given position."""
    candidates = [c for c in catalog.values() if c.page_num <= page_num and c.y0 <= y0]
    if not candidates:
        return None
    
    # Sort by page (descending) then by y0 (descending) to get the most recent
    candidates.sort(key=lambda c: (c.page_num, c.y0), reverse=True)
    return candidates[0]

def get_heading_stack(catalog: Dict[str, HeadingCandidate], page_num: int, y0: float) -> List[HeadingCandidate]:
    """Get the heading stack (L1, L2, L3) for a given position.

    - Include all headings from previous pages (ignore y0 for pages < page_num)
    - Include only headings above y0 on the current page
    """
    stack = []
    # Split selection by page
    prior_pages = [c for c in catalog.values() if c.page_num < page_num]
    same_page = [c for c in catalog.values() if c.page_num == page_num and c.y0 <= y0]
    candidates = prior_pages + same_page
    candidates.sort(key=lambda c: (c.page_num, c.y0))
    
    # Build stack maintaining level hierarchy
    for candidate in candidates:
        if candidate.level == 1:
            stack = [candidate]  # New chapter, reset stack
        elif candidate.level == 2:
            stack = stack[:1] + [candidate]  # New section, keep chapter
        elif candidate.level == 3:
            stack = stack[:2] + [candidate]  # New subsection, keep chapter and section
    
    return stack

def get_page_headings(catalog: Dict[str, HeadingCandidate], page_num: int) -> List[HeadingCandidate]:
    """Get all headings for a specific page, sorted by y-coordinate."""
    page_headings = [c for c in catalog.values() if c.page_num == page_num]
    page_headings.sort(key=lambda c: c.y0)
    return page_headings
