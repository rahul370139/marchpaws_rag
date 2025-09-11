"""Main PDF parser for TC 4-02.1 with proper paragraph-level citations."""

import fitz
import json
import re
from cleaners import strip_regions, reflow, polish_tokens, detect_version, clean_footer_leakage, fix_compound_words
from segmenters import paragraphs_from_page_blocks, stitch_paragraphs_across_pages, hash_para
from heading_discovery import build_heading_catalog, get_heading_stack, get_page_headings
from anchors import get_proper_heading_chain

DOC_ID = "tc4-02.1"

def validate_record(record):
    """Validate a record against the expected schema."""
    required_fields = ['id', 'chapter', 'para', 'section', 'heading', 'text', 'anchors']
    
    # Check for missing required fields
    missing_fields = [field for field in required_fields if field not in record]
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"
    
    # Check for empty or invalid values
    if not record['id'] or not record['para'] or not record['text']:
        return False, "Empty id, para, or text field"
    
    # Check for valid paragraph ID format
    if not re.match(r'^\d+-\d+$', record['para']):
        return False, f"Invalid paragraph ID format: {record['para']}"
    
    # Check for valid chapter number
    if record['chapter'] is not None and not isinstance(record['chapter'], int):
        return False, f"Invalid chapter number: {record['chapter']}"
    
    # Check for valid section and heading (can be None)
    if record['section'] is not None and not isinstance(record['section'], str):
        return False, f"Invalid section format: {record['section']}"
    
    if record['heading'] is not None and not isinstance(record['heading'], str):
        return False, f"Invalid heading format: {record['heading']}"
    
    # Check for valid anchors
    if not isinstance(record['anchors'], dict):
        return False, f"Invalid anchors format: {record['anchors']}"
    
    required_anchor_fields = ['short_ref', 'norm_ref', 'toc_path']
    missing_anchor_fields = [field for field in required_anchor_fields if field not in record['anchors']]
    if missing_anchor_fields:
        return False, f"Missing anchor fields: {missing_anchor_fields}"
    
    return True, "Valid"

def extract_page_heading_lines(page: fitz.Page):
    """Extract candidate heading lines with y0 from a page using span-level info."""
    d = page.get_text("dict")
    lines = []
    for b in d.get("blocks", []):
        for l in b.get("lines", []):
            text = "".join(sp.get("text", "") for sp in l.get("spans", []))
            t = re.sub(r"\s+", " ", text).strip()
            if not t:
                continue
            y0 = l.get("bbox", [0, 0, 0, 0])[1]
            # classify
            is_section = bool(re.match(r'^Section\s+[IVXLC]+\s*[—-]\s*', t, re.I))
            is_chapter = bool(re.match(r'^Chapter\s+\d+', t, re.I))
            # heading-ish: short, no trailing punctuation, not para id
            looks_heading = (
                4 <= len(t) < 100 and
                not re.match(r'^\d+\s*-\s*\d+\.', t) and
                not t.endswith('.') and not t.endswith(':')
            )
            if is_section or is_chapter or looks_heading:
                lines.append({
                    "text": t,
                    "y0": y0,
                    "is_section": is_section,
                    "is_chapter": is_chapter
                })
    lines.sort(key=lambda x: x["y0"])  # top to bottom
    return lines

def derive_section_and_heading_from_page_lines(page_heading_lines, y0, current_section_full, current_heading):
    """Use page heading lines to find nearest section and heading above paragraph y0."""
    section_full = current_section_full
    heading = current_heading
    # candidates above this paragraph
    prior = [ln for ln in page_heading_lines if ln["y0"] <= y0]
    if prior:
        # Only adopt a new section if the section banner is reasonably close above (avoid cross-column bleed)
        MAX_SECTION_GAP = 80.0
        MAX_HEADING_GAP = 100.0
        # Find nearest section above
        nearest_section = None
        for ln in reversed(prior):
            if ln["is_section"]:
                nearest_section = ln
                break
        if nearest_section and (y0 - nearest_section["y0"]) <= MAX_SECTION_GAP:
            section_full = re.sub(r"\s{2,}", " ", nearest_section["text"]).strip()
        # heading = nearest non-section, non-chapter above within gap
        nearest_heading = None
        for ln in reversed(prior):
            if not ln["is_section"] and not ln["is_chapter"]:
                if (y0 - ln["y0"]) <= MAX_HEADING_GAP:
                    nearest_heading = ln
                    break
        if nearest_heading:
            heading = re.sub(r"\s{2,}", " ", nearest_heading["text"]).strip()
    return section_full, heading


def derive_from_chain(heading_chain):
    """Derive chapter_title, section_full, leaf heading from a sanitized heading_chain."""
    chapter_title = None
    section_full = None
    heading = None
    # chapter = nearest Chapter in chain (last occurrence)
    for seg in reversed(heading_chain):
        if re.match(r'^Chapter\s+\d+', seg, re.I):
            chapter_title = seg.strip()
            break
    # section = last Section ... in chain
    for seg in heading_chain:
        if re.match(r'^Section\s+[IVXLC]+\s*[—-]\s*', seg, re.I):
            section_full = re.sub(r'\s{2,}', ' ', seg.strip())
    # heading = last non-Section, non-Chapter
    for seg in reversed(heading_chain):
        if re.match(r'^Chapter\s+\d+', seg, re.I) or re.match(r'^Section\s+[IVXLC]+\s*[—-]\s*', seg, re.I):
            continue
        heading = re.sub(r'\s{2,}', ' ', seg.strip())
        break
    # Sanitize chapter title (remove training codes, page markers)
    if chapter_title:
        chapter_title = re.sub(r"\(\d{3}[-A-Z0-9]+\)", "", chapter_title)  # remove codes like (081-COM-1046)
        chapter_title = re.sub(r"\b\d{1,2}\s*[\-–]\s*\d{1,2}\b", "", chapter_title)  # remove 23-1
        chapter_title = re.sub(r"\s+", " ", chapter_title).strip()
    return chapter_title, section_full, heading


def refine_heading_by_text(page_idx, y0, heading_catalog, page_heading_lines, paragraph_text, current_heading):
    """Refine leaf heading by matching short candidate headings above the paragraph to paragraph text.

    General rule: if a short leaf (1-3 tokens) appears above this paragraph and its tokens are present in
    the paragraph text, prefer it over the existing leaf.
    """
    if not paragraph_text:
        return current_heading
    text_l = paragraph_text.lower()
    best = current_heading
    best_score = 0.0
    # Detect component-style paragraph
    has_components_cue = bool(re.search(r"\bconsists of\b|\bconsist of\b", text_l)) or bool(re.search(r"(^|\n)\s*-\s+", paragraph_text))
    # Build candidate list from catalog and page heading lines
    catalog_cands = [
        {"text": c.text, "y0": c.y0}
        for c in get_page_headings(heading_catalog, page_idx)
        if c.y0 <= y0
    ]
    line_cands = [
        {"text": ln["text"], "y0": ln["y0"]}
        for ln in page_heading_lines
        if ln["y0"] <= y0 and not ln["is_section"] and not ln["is_chapter"]
    ]
    merged = catalog_cands + line_cands
    # Dedup by normalized text and pick nearest above
    seen = {}
    for c in merged:
        key = re.sub(r"\s+", " ", c["text"].strip()).lower()
        if key not in seen or y0 - c["y0"] < y0 - seen[key]["y0"]:
            seen[key] = c
    for cand in seen.values():
        if cand["y0"] > y0:
            continue  # only consider headings above
        if re.match(r'^Chapter\s+\d+', cand["text"], re.I):
            continue
        if re.match(r'^Section\s+[IVXLC]+\s*[—-]\s*', cand["text"], re.I):
            continue
        toks = re.findall(r"[a-zA-Z]+", cand["text"])
        if not (1 <= len(toks) <= 6):
            continue
        # score = token overlap fraction + proximity weight
        present = sum(1 for t in toks if re.search(rf"\b{re.escape(t.lower())}\b", text_l))
        token_score = present / max(1, len(toks))
        # proximity: closer headings get bonus; cap distance to avoid huge values
        dist = max(1.0, y0 - cand["y0"])
        proximity_score = max(0.0, 1.0 - min(dist, 400.0) / 400.0)  # within 400px window
        score = 0.7 * token_score + 0.3 * proximity_score
        # Prefer "Components of ..." when paragraph has components cue
        if has_components_cue and re.match(r'(?i)^components\s+of\s+', cand["text" ].strip()):
            score += 0.5
        if score > best_score and token_score >= 0.5:
            best = cand["text"]
            best_score = score
    return best

def process_paragraph_with_section(para_id, text, page_idx, current_chapter, version, y0, 
                                 current_section_full, current_heading, heading_catalog, 
                                 records, seen_exact, page_heading_lines=None):
    """Process a single paragraph with section and heading assignment."""
    # Skip empty/blank-page fragments
    if re.fullmatch(r"\s*this page intentionally left blank\.?\s*", text, re.I):
        return current_section_full, current_heading
    
    # Clean up text
    text = text.strip()
    if not text:
        return current_section_full, current_heading
    
    # Infer chapter from paragraph ID
    chapter = None
    if para_id and '-' in para_id:
        try:
            chapter = int(para_id.split('-')[0])
        except ValueError:
            pass
    
    # Update current chapter
    if chapter:
        current_chapter = chapter
    
    # Exact paragraph dedupe across doc - use para_id as primary key
    if para_id in seen_exact:
        return current_section_full, current_heading
    seen_exact.add(para_id)
    
    # Use anchors to get proper heading chain at (page, y0)
    chain = get_proper_heading_chain(heading_catalog, page_idx, y0)
    chapter_title, section, heading = derive_from_chain(chain)
    
    # Check for heading leakage - improved detection
    leaked_heading = None
    lines = text.split('\n')
    if lines:
        last_line = lines[-1].strip()
        # Check if last line looks like a leaked heading
        if (len(last_line) > 5 and len(last_line) < 100 and
            (last_line.isupper() or 
             re.match(r'^[A-Z][A-Z\s]+\([A-Z\s]+\)$', last_line) or
             re.match(r'^[A-Z][A-Z\s]+$', last_line)) and
            not last_line.endswith('.') and not last_line.endswith(':')):
            leaked_heading = last_line
            # Remove the leaked heading from text
            text = '\n'.join(lines[:-1]).strip()
    
    # Additional check for specific patterns (normalize known leaked subheads)
    leaked_patterns = [
        r'FIRST AID \(self-aid AND buddy aid\)',
        r'\bNOTE\.?\s+.*$',
    ]
    for pat in leaked_patterns:
        text = re.sub(pat, '', text, flags=re.I).strip()
    
    # Generate citations
    # We'll build citations after normalizing section/heading for consistency
    
    # Build proper citation strings - never use "Ch?"
    if para_id and chapter:
        short_ref = f"Ch{chapter} §{para_id} ({version})"
        norm_ref = f"TC 4-02.1 Ch{chapter} §{para_id} ({version})"
    elif chapter:
        short_ref = f"Ch{chapter} ({version})"
        norm_ref = f"TC 4-02.1 Ch{chapter} ({version})"
    else:
        short_ref = f"§{para_id} ({version})"
        norm_ref = f"TC 4-02.1 §{para_id} ({version})"
    
    h = hash_para(f"{para_id}||{text}")
    
    # Extract section short name from full section name dynamically
    section_short = None
    if section:
        # Normalize to canonical label
        msec = re.match(r'(?i)^section\s+([ivxlc]+)\s*[—-]\s*(.+)$', section.strip(), re.I)
        if msec:
            roman = msec.group(1).upper()
            name = re.sub(r'\s+', ' ', msec.group(2).strip())
            # Title-case the name lightly (keep ALLCAPS words)
            name_tc = " ".join([w if w.isupper() else w.title() for w in name.split(' ')])
            section = f"Section {roman} — {name_tc}"
            section_short = name_tc
        else:
            section_short = re.sub(r'\s+', ' ', section.strip())

    # Remove any content-aware fallback heuristics; rely purely on anchor chain

    # Normalize heading case lightly and collapse spaces
    if heading:
        heading = re.sub(r'\s+', ' ', heading.strip())
        if not heading.isupper():
            heading = " ".join([w if w.isupper() else w.title() for w in heading.split(' ')])

    # Refine leaf heading using paragraph content and page-local candidates (global rule)
    heading = refine_heading_by_text(page_idx, y0, heading_catalog, page_heading_lines, text, heading)
    if heading:
        heading = re.sub(r'\s+', ' ', heading.strip())
        if not heading.isupper():
            heading = " ".join([w if w.isupper() else w.title() for w in heading.split(' ')])

    # Build toc_path: always include Chapter root using centered dot separator
    toc_parts = []
    if chapter_title:
        m = re.match(r'^Chapter\s+(\d+)\s*[·\-–—]?\s*(.*)$', chapter_title, re.I)
        if m:
            chn = int(m.group(1))
            cht = re.sub(r'\s+', ' ', m.group(2).strip())
            # If chapter title looks noisy or empty, use simple fallback
            if not cht or re.search(r'\d{1,2}\s*[\-–]\s*\d{1,2}', chapter_title):
                chapter_title = f"Chapter {chn}"
            else:
                chapter_title = f"Chapter {chn} · {cht}" if cht else f"Chapter {chn}"
        toc_parts.append(chapter_title)
    if section:
        toc_parts.append(section)
    if heading and (not chapter_title or heading.lower() not in chapter_title.lower()):
        toc_parts.append(heading)
    citations = {
        "short_ref": f"Ch{chapter} §{para_id} ({version})" if chapter else f"§{para_id} ({version})",
        "norm_ref": f"TC 4-02.1 Ch{chapter} §{para_id} ({version})" if chapter else f"TC 4-02.1 §{para_id} ({version})",
        "toc_path": " > ".join([p for p in toc_parts if p])
    }
    
    record = {
        "id": f"tc4-02.1:ch{chapter if chapter else '?'}:{para_id}@{version}",
        "doc_id": "tc4-02.1",
        "chapter": chapter,
        "para": para_id,
        "section": section_short,
        "section_full": section,
        "heading": heading,
        "version": version,
        "page_start": page_idx + 1,
        "page_end": page_idx + 1,
        "anchors": {
            "short_ref": short_ref,
            "norm_ref": norm_ref,
            "toc_path": citations["toc_path"],
        },
        "text": text,
        "source": "data/tc4-02.1wc1x2.pdf",
        "hash": h,
        "change_note": None
    }
    
    # Validate record before adding
    is_valid, validation_msg = validate_record(record)
    if not is_valid:
        print(f"[WARNING] Invalid record {record['id']}: {validation_msg}")
        print(f"         Skipping record...")
        return section, heading
    
    records.append(record)
    return section, heading

def parse_pdf(pdf_path, out_jsonl):
    """Parse PDF and generate structured JSONL output."""
    doc = fitz.open(pdf_path)
    
    print(f"[INFO] Starting PDF parsing...")
    # Build heading catalog once (ToC + span-level headings)
    heading_catalog = build_heading_catalog(doc)

    records = []
    seen_exact = set()
    all_paragraphs = []  # Collect all paragraphs for stitching
    
    # Track current section and heading globally
    current_section = None
    current_heading = None

    for i, page in enumerate(doc):
        raw_page = page.get_text("text")
        version = detect_version(raw_page)

        body = reflow(strip_regions(page))
        body = clean_footer_leakage(body)  # Remove footer leakage
        body = polish_tokens(body)
        body = fix_compound_words(body)  # Fix compound word gluing

        # Check for section headers in this page - only if it's not a TOC page
        if not re.search(r'contents|table of contents', body, re.I):
            section_matches = re.findall(r'SECTION\s+([IVXLC]+)\s*—\s*([^\n]+)', body, re.I)
            for section_num, section_name in section_matches:
                # Clean up the section name (remove dots and page numbers)
                section_name = re.sub(r'[.\s]+$', '', section_name.strip())
                section_name = re.sub(r'\d+$', '', section_name).strip()
                # Only update if this looks like a real section header, not TOC content
                if len(section_name) < 100:  # Reasonable section name length
                    current_section = f"Section {section_num} — {section_name}"
                    current_heading = current_section
                    print(f"[INFO] Found section on page {i+1}: {current_section}")
        
        # No per-page heuristic headings; rely on y0-based heading catalog

        # Determine current chapter from heading catalog
        # This will be handled by the heading discovery system
        current_chapter = None

        # Extract page blocks with y-coordinates for accurate heading assignment
        page_blocks = page.get_text("dict")["blocks"]
        # Extract page heading lines (sections/chapters/subheads) using spans
        page_heading_lines = extract_page_heading_lines(page)
        
        # split into paragraphs with para IDs when present
        parts = paragraphs_from_page_blocks(body, current_chapter=current_chapter, page_blocks=page_blocks)
        
        # Capture orphaned text that might be continuations of previous paragraphs
        # Look for text that doesn't start with a paragraph ID but might be a continuation
        lines = body.split('\n')
        for line in lines:
            line = line.strip()
            if (line and 
                not re.match(r'^\d+-\d+\.', line) and  # Not a paragraph ID
                not re.match(r'^[A-Z\s]+$', line) and  # Not a heading
                not re.match(r'^(WARNING|CAUTION|Note)', line, re.I) and  # Not a warning
                len(line) > 20):  # Substantial text
                # This might be orphaned continuation text
                all_paragraphs.append((None, line, i, current_chapter, version))
        
        # Process paragraphs immediately to get correct section assignment
        for para_data in parts:
            if len(para_data) == 3:  # (pid, content, y0)
                pid, content, y0 = para_data
                if pid:  # Only process paragraphs with IDs
                    # Process this paragraph with current section context (y0-based stack)
                    # Prefer page heading lines for precise y0 mapping; fallback to catalog stack
                    sec_h, head_h = derive_section_and_heading_from_page_lines(page_heading_lines, y0, current_section, current_heading)
                    if not sec_h and not head_h:
                        # basic fallback: keep current section/heading
                        sec_h, head_h = current_section, current_heading
                    new_section, new_heading = process_paragraph_with_section(
                        pid, content, i, current_chapter, version, y0,
                        sec_h or current_section, head_h or current_heading, heading_catalog,
                        records, seen_exact, page_heading_lines
                    )
                    if new_section:
                        current_section = new_section
                    if new_heading:
                        current_heading = new_heading
            else:  # fallback for old format
                pid, content = para_data
                if pid:  # Only process paragraphs with IDs
                    sec_h, head_h = derive_section_and_heading_from_page_lines(page_heading_lines, 0, current_section, current_heading)
                    if not sec_h and not head_h:
                        sec_h, head_h = current_section, current_heading
                    new_section, new_heading = process_paragraph_with_section(
                        pid, content, i, current_chapter, version, 0,
                        sec_h or current_section, head_h or current_heading, heading_catalog,
                        records, seen_exact, page_heading_lines
                    )
                    if new_section:
                        current_section = new_section
                    if new_heading:
                        current_heading = new_heading

    # Skipping stitched re-processing to avoid conflicting anchors; immediate processing above is authoritative.

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Final validation of output file
    print(f"[OK] Wrote {len(records)} records to {out_jsonl}")
    
    # Validate the output file
    validation_errors = 0
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                is_valid, msg = validate_record(data)
                if not is_valid:
                    print(f"[ERROR] Line {i+1}: {msg}")
                    validation_errors += 1
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {i+1}: JSON decode error - {e}")
                validation_errors += 1
    
    if validation_errors == 0:
        print(f"[OK] All {len(records)} records passed validation")
    else:
        print(f"[WARNING] {validation_errors} records failed validation")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python parse_tc4021.py <pdf_path> <output_jsonl>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2]
    parse_pdf(pdf_path, output_path)