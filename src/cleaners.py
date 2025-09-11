"""Text cleaning utilities for PDF processing."""

import re
import unicodedata

HEADER_H, FOOTER_H = 80, 30

BANNED_LINE_RE = re.compile(
    r"(DISTRIBUTION RESTRICTION|Army Knowledge Online|AdminPubs|By Order of the Secretary|"
    r"Headquarters\s+Department of the Army|This page intentionally left blank\.?)",
    re.I,
)

# Footer patterns to remove
FOOTER_PATTERNS = [
    re.compile(r"^\s*\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s+TC\s*4-?02\.1\s+\d+\s*$", re.I),
    re.compile(r"^\s*TC\s*4-?02\.1\s*\w*\s*$", re.I),
    re.compile(r"^\s*\d+\s*$"),  # Page numbers only
]

def strip_regions(page):
    """Remove header and footer regions from page."""
    rect = page.rect
    kept = []
    for (x0, y0, x1, y1, txt, *_) in page.get_text("blocks"):
        if y0 < rect.y0 + HEADER_H:  # header
            continue
        if y1 > rect.y1 - FOOTER_H:  # footer
            continue
        lines = [ln for ln in txt.splitlines() if not BANNED_LINE_RE.search(ln.strip())]
        if lines:
            kept.append("\n".join(lines))
    return "\n".join(kept)

LIST_BULLET_RE = re.compile(r"^\s*([-\*\u2022\u25CF\u25AA\u25E6\uF0B7]|[0-9]+\.)\s+")

def reflow(text: str) -> str:
    """Join hard-wrapped lines inside paragraphs; keep real list lines and blank lines."""
    out, buf = [], []
    
    def flush():
        if buf:
            out.append(" ".join(buf).strip())
            buf.clear()
    
    t = text.replace("\xad", "")  # soft hyphen
    for ln in t.splitlines():
        if not ln.strip():
            flush()
            out.append("")
            continue
        if LIST_BULLET_RE.match(ln) or ln.strip().startswith(("Note.", "WARNING", "CAUTION")):
            flush()
            out.append(ln.strip())
            continue
        
        # Check if this line starts with a paragraph ID pattern (X-Y.)
        if re.match(r"^\d+-\d+\.", ln.strip()):
            flush()  # Flush any previous content
            out.append(ln.strip())  # Keep paragraph ID lines separate
            continue
            
        # join hyphenated line-ends, but be careful with paragraph IDs
        ln = re.sub(r"(\w)-\s*(\w)", r"\1\2", ln.strip())
        buf.append(ln)
    
    flush()
    return "\n".join(out)

def normalize_for_hashing(text: str) -> str:
    """Produce a normalized string for deduplication."""
    t = unicodedata.normalize("NFKD", text).lower()
    t = re.sub(r"[\u2022\u2023\u2024\u25CF\u25AA\u25E6\uF0B7]+", "-", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

UNMERGE = [
    (r"\blife ?threatening\b", "life-threatening"),
    (r"\bself ?aid\b", "self-aid"),
    (r"\binjured ?side\b", "injured side"),
    (r"\bmos ?specific\b", "MOS-specific"),
]

def polish_tokens(s: str) -> str:
    """Fix common tokenization issues."""
    for pat, rep in UNMERGE:
        s = re.sub(pat, rep, s, flags=re.I)
    
    # Normalize bullet characters
    s = re.sub(r"[\uF0B7\u2022\u25CF\u25AA\u25E6\u2023\u2024•\uF06C]+", "-", s)
    
    return s

def clean_footer_leakage(text: str) -> str:
    """Remove footer leakage that slipped past region filtering."""
    # First, try to remove footer patterns from the end of lines
    for pattern in FOOTER_PATTERNS:
        text = pattern.sub('', text)
    
    # Also remove common footer patterns that might be embedded in text
    footer_embedded_patterns = [
        re.compile(r'\s+\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s+TC\s*4-?02\.1\s+\d+\s*$', re.I),
        re.compile(r'\s+TC\s*4-?02\.1\s*\w*\s*$', re.I),
    ]
    
    for pattern in footer_embedded_patterns:
        text = pattern.sub('', text)
    
    return text

def fix_compound_words(text: str) -> str:
    """Fix common compound word gluing issues and normalize casing."""
    fixes = [
        (r'life\s+threatening', 'life-threatening'),
        (r'self\s+aid', 'self-aid'),
        (r'tactical\s+combat', 'tactical-combat'),
        (r'tacticalcombat', 'tactical-combat'),  # Handle already glued words
        (r'9\s+line', '9-line'),
        (r'9line', '9-line'),  # Handle already glued words
        (r'MOS\s+specific', 'MOS-specific'),
        (r'buddy\s+aid', 'buddy aid'),  # Keep as two words per source
        (r'buddyaid', 'buddy aid'),  # Handle already glued words
    ]
    
    for pattern, replacement in fixes:
        text = re.sub(pattern, replacement, text, flags=re.I)
    
    # Normalize casing for specific terms
    casing_fixes = [
        (r'\btactical-combat casualty care\b', 'Tactical combat casualty care', re.I),
        (r'\btc3\b', 'TC3', re.I),
        (r'\bmedevac\b', 'MEDEVAC', re.I),
    ]
    
    for pattern, replacement, flags in casing_fixes:
        text = re.sub(pattern, replacement, text, flags=flags)
    
    return text

def detect_version(raw_text: str) -> str:
    """Detect document version from text content."""
    if re.search(r"\bC2\b|\b7\s+December\s+2018\b", raw_text):
        return "C2"
    if re.search(r"\bC1\b|\b5\s+August\s+2016\b", raw_text):
        return "C1"
    return "Base"
