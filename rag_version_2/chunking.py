"""Chunking utilities for RAG v2.

Provides a subsection-aware QA chunker that splits on both top-level (1.) and
subsection (1.1, 1.2.3) headings, while preserving parent context.
"""

import re
from typing import Iterable


def _chunk_by_words(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    buf: list[str] = []
    for w in words:
        buf.append(w)
        if len(buf) >= chunk_size:
            chunks.append(" ".join(buf))
            buf = []
    if buf:
        chunks.append(" ".join(buf))
    return [c.strip() for c in chunks if c.strip()]


def _chunk_by_qa_sections_v2(text: str, chunk_size: int) -> list[str]:
    text = text.lstrip("\ufeff")
    pattern = re.compile(
        r"(?ms)^[ \t]*(\d+(?:\.\d+)*\.?[ \t]*[^\n]{3,})\n(.*?)(?=^[ \t]*\d+(?:\.\d+)*\.?[ \t]*[^\n]{3,}\n|\Z)"
    )

    sections: list[str] = []
    parent_heading = ""

    for match in pattern.finditer(text):
        heading = match.group(1).strip()
        body = match.group(2).strip()
        is_subsection = re.match(r"^\d+\.\d+", heading) is not None
        if is_subsection and parent_heading:
            full_text = f"{parent_heading}\n{heading}\n{body}".strip()
        else:
            full_text = f"{heading}\n{body}".strip()
            parent_heading = heading

        if len(full_text.split()) > int(chunk_size * 1.5):
            heading_words = len(heading.split()) + 2
            body_chunk_size = max(chunk_size - heading_words, 80)
            parts = _chunk_by_words(body, body_chunk_size)
            sections.extend([f"{heading}\n{part}" for part in parts if part.strip()])
        else:
            sections.append(full_text)

    return [s.strip() for s in sections if s.strip()]


def chunk_text_v2(text: str, chunk_size: int, mode: str) -> list[str]:
    """Split text into chunks.

    Modes:
      - qa_v2: subsection-aware (1.1, 2.3.4) QA chunking
      - auto:  use qa_v2 if it finds enough sections, else word chunking
      - words: always word chunking
    """
    mode = (mode or "auto").strip().lower()
    if mode in {"qa_v2", "auto"}:
        qa_chunks = _chunk_by_qa_sections_v2(text, chunk_size)
        if mode == "qa_v2":
            return qa_chunks if qa_chunks else _chunk_by_words(text, chunk_size)
        if len(qa_chunks) >= 5:
            return qa_chunks

    chunks = _chunk_by_words(text, chunk_size)
    if not chunks and text.strip():
        chunks = [text.strip()]
    return chunks


def join_chunks(chunks: Iterable[str]) -> str:
    return "\n\n---\n\n".join(c for c in chunks if c.strip())
