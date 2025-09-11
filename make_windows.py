#!/usr/bin/env python3
"""Create parent windows from paragraphs with overlap."""

import json
import itertools
import re
import hashlib
from pathlib import Path

MAX_TOK = 250
OVERLAP = 100

def tok_len(t):
    """Calculate token length of text using simple word splitting."""
    return len(t.split())

def read_paragraphs(path):
    """Read paragraphs from JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f]

def sent_splitter(text, target_tokens=120):
    """Split text on sentence boundaries to create sub-paragraphs."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    splits = []
    current = []
    current_tokens = 0
    
    for sent in sentences:
        sent_tokens = tok_len(sent)
        if current_tokens + sent_tokens > target_tokens and current:
            splits.append(" ".join(current))
            current = [sent]
            current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens
    
    if current:
        splits.append(" ".join(current))
    
    return splits

def make_windows(paras):
    """Create windows from paragraphs with overlap."""
    # 1. Guarantee canonical ordering - extract numeric para number for sorting
    def sort_key(p):
        chapter = p.get("chapter", 0)
        page = p.get("page_start", 0)
        para_str = p.get("para", "0")
        # Extract numeric part from para string like "1-7" -> 7
        try:
            para_num = int(para_str.split("-")[-1]) if "-" in para_str else int(para_str)
        except (ValueError, AttributeError):
            para_num = 0
        return (chapter, page, para_num)
    
    paras = sorted(paras, key=sort_key)
    
    # 2. Handle "giant" paragraphs (> MAX_TOK) first
    processed_paras = []
    for p in paras:
        p_tok = tok_len(p["text"])
        if p_tok > MAX_TOK:
            # Split on sentence boundaries
            splits = sent_splitter(p["text"], 120)
            for idx, sub_text in enumerate(splits):
                sub_paragraph = p.copy()
                sub_paragraph["id"] += f"_split{idx}"
                sub_paragraph["text"] = sub_text
                sub_paragraph["split_of"] = p["id"]  # Tag for citation traceability
                processed_paras.append(sub_paragraph)
        else:
            processed_paras.append(p)
    
    # 3. Create windows with simple sliding window approach
    windows = []
    i = 0
    stride_idx = 0
    
    print(f"Processing {len(processed_paras)} paragraphs...")
    
    while i < len(processed_paras):
        # Start a new window
        window_paras = []
        window_tokens = 0
        current_section = None
        
        # Add paragraphs until we hit the limit
        while i < len(processed_paras) and window_tokens < MAX_TOK:
            p = processed_paras[i]
            p_tok = tok_len(p["text"])
            
            # Check for semantic boundary (section change)
            para_section = p.get("anchors", {}).get("toc_path", "").split(" > ")[0] if p.get("anchors", {}).get("toc_path") else ""
            if current_section is None:
                current_section = para_section
            elif para_section and para_section != current_section and window_paras:
                # Section changed, cut new window even if under token limit
                print(f"  Section boundary detected: {current_section} -> {para_section}, cutting window")
                break
            
            if window_tokens + p_tok > MAX_TOK and window_paras:
                # Can't fit this paragraph, start new window
                break
                
            window_paras.append(p)
            window_tokens += p_tok
            i += 1
        
        if window_paras:
            windows.append(emit_window(window_paras, stride_idx))
            print(f"  Created window {len(windows)} with {len(window_paras)} paragraphs, {window_tokens} tokens")
            stride_idx += 1  # Always increment stride_idx for true monotonic sequence
        
        # For overlap: move back by overlap amount, but ensure we don't go backwards
        if i < len(processed_paras) and len(window_paras) > 1:
            # Move back by overlap amount, but at least move forward by 1
            overlap_count = min(OVERLAP // 50, len(window_paras) - 1)  # Roughly 2 paragraphs for 100 token overlap
            i = max(i - overlap_count, i - len(window_paras) + 1)  # Don't go backwards more than window size
            print(f"  Overlapping {overlap_count} paragraphs for next window, new i={i}")
    
    return windows

def emit_window(buf, stride_idx):
    """Emit a window record from buffer."""
    ids = [p["id"] for p in buf]
    chapters = {p["chapter"] for p in buf}
    pages = [p["page_start"] for p in buf] + [p["page_end"] for p in buf]
    
    # Delimit paragraphs with newlines for better embedding quality
    text = "\n\n".join(p["text"] for p in buf)
    
    # Stable but human-readable window_id with page info
    page_start = min(pages)
    page_end = max(pages)
    window_id = f"w_p{page_start}-{page_end}"
    
    # Token lengths for each child paragraph
    child_tok_lens = [tok_len(p["text"]) for p in buf]
    
    # Extract section/heading metadata for state-aware boosting
    section_paths = [p.get("anchors", {}).get("toc_path", "") for p in buf if p.get("anchors", {}).get("toc_path")]
    headings = list({p.get("heading", "") for p in buf if p.get("heading")})
    
    return {
        "window_id": window_id,
        "stride_idx": stride_idx,
        "paragraph_ids": ids,
        "chapter": min(chapters),
        "page_start": page_start,
        "page_end": page_end,
        "text": text,
        "token_len": sum(child_tok_lens),
        "child_tok_lens": child_tok_lens,
        "paragraph_short_refs": [p.get("anchors", {}).get("short_ref", f"Ch{p.get('chapter', '?')} §{p.get('para', '?')}") for p in buf],
        "section_paths": section_paths,
        "headings": headings
    }

def expand_windows(windows, para_map, max_paras=12):
    """5. Expansion & de-dup - expand windows to individual paragraphs."""
    seen, out = set(), []
    for w in windows:
        for pid in w["paragraph_ids"]:
            if pid not in seen and pid in para_map:
                out.append(para_map[pid])
                seen.add(pid)
                if len(out) >= max_paras:
                    return out
    return out

if __name__ == "__main__":
    # Read paragraphs from the parsed data
    paras = read_paragraphs("data/tc4-02.1_sections.jsonl")
    windows = make_windows(paras)
    
    # Write windows to file
    output_path = "data/windows.jsonl"
    Path(output_path).write_text("\n".join(json.dumps(w) for w in windows))
    print(f"Made {len(windows)} parent windows")
    print(f"Windows saved to {output_path}")
    
    # Statistics
    token_counts = [w["token_len"] for w in windows]
    print(f"\nToken distribution:")
    print(f"  Min: {min(token_counts)}")
    print(f"  Max: {max(token_counts)}")
    print(f"  Avg: {sum(token_counts)/len(token_counts):.1f}")
    print(f"  Windows >250 tokens: {sum(1 for t in token_counts if t > 250)}")
    print(f"  Windows <100 tokens: {sum(1 for t in token_counts if t < 100)}")
    
    # Overlap analysis
    overlaps = []
    for i in range(1, len(windows)):
        prev_ids = set(windows[i-1]["paragraph_ids"])
        curr_ids = set(windows[i]["paragraph_ids"])
        overlap = len(prev_ids & curr_ids)
        overlaps.append(overlap)
    
    if overlaps:
        print(f"\nOverlap analysis:")
        print(f"  Avg paragraph overlap: {sum(overlaps)/len(overlaps):.1f}")
        print(f"  Max paragraph overlap: {max(overlaps)}")
        print(f"  Windows with 0 overlap: {sum(1 for o in overlaps if o == 0)}")
    
    # Show example
    if windows:
        print("\nExample window:")
        print(json.dumps(windows[0], indent=2))
    
    # Test expansion function
    para_map = {p["id"]: p for p in paras}
    expanded = expand_windows(windows[:3], para_map, max_paras=8)
    print(f"\nExpansion test: {len(expanded)} paragraphs from first 3 windows")
