"""Utility functions shared across modules."""

from typing import List, Dict


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


def format_excerpts(chunks: List[Dict], max_chars_per_chunk: int = 1800) -> str:
    """Format the retrieved chunks for inclusion in the prompt using exact para IDs.

    Header format example:
      [Ch1 §1-19 | Section II — Vital Body Systems | COMPONENTS OF THE RESPIRATORY SYSTEM | p.15–15]
    """
    out_lines = []
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
    return "\n".join(out_lines)