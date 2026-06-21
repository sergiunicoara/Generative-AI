"""Semantic chunking using LangChain's RecursiveCharacterTextSplitter.

Chunking is heading-aware: after the document is split into chunks of
`chunk_size` as usual, each chunk has the nearest preceding markdown
heading (lines starting with 1-6 `#`) prepended, if not already present.
This keeps heading vocabulary (e.g. "## 10. INDICATORI KPI CU VALORI TARGET
NUMERICE") attached to every chunk under that heading -- including table
rows or list items that would otherwise end up isolated from their
section's context -- without changing chunk boundaries/granularity.
"""

from __future__ import annotations

import bisect
import re

from graphrag.core.config import get_settings
from graphrag.core.models import Chunk, Document

_HEADING_RE = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)


def _heading_positions(raw_text: str) -> tuple[list[int], list[str]]:
    """Return (offsets, headings) of each markdown heading line in raw_text."""
    offsets: list[int] = []
    headings: list[str] = []
    for m in _HEADING_RE.finditer(raw_text):
        offsets.append(m.start())
        headings.append(m.group(1).strip())
    return offsets, headings


def chunk_document(document: Document) -> list[Chunk]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    cfg = get_settings().ingestion
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.get("chunk_size", 512),
        chunk_overlap=cfg.get("chunk_overlap", 64),
        length_function=len,
    )

    raw_text = document.raw_text
    offsets, headings = _heading_positions(raw_text)

    texts: list[str] = []
    cursor = 0
    for chunk_text in splitter.split_text(raw_text):
        pos = raw_text.find(chunk_text, cursor)
        if pos == -1:
            pos = raw_text.find(chunk_text)
        if pos != -1:
            cursor = pos

        idx = bisect.bisect_right(offsets, pos if pos != -1 else cursor) - 1
        heading = headings[idx] if idx >= 0 else None
        if heading and not chunk_text.lstrip().startswith(heading):
            chunk_text = f"{heading}\n\n{chunk_text}"
        texts.append(chunk_text)

    return [
        Chunk(
            document_id=document.id,
            text=text,
            chunk_index=i,
            tenant=document.tenant,
        )
        for i, text in enumerate(texts)
    ]
