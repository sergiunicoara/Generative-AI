"""Section-aware semantic chunking.

Documents are split at section-heading boundaries first, then each section is
kept whole when it fits in `chunk_size` (or sub-split with LangChain's
RecursiveCharacterTextSplitter when it doesn't). A "heading" is either a
markdown heading (lines starting with 1-6 `#`) or an all-caps numbered
section header (e.g. "3. BUDGET & KPI TARGETS", "10. INDICATORI KPI").

Why section-first rather than pure fixed-size splitting: a fixed-size splitter
packs whatever fits into 512 chars, so it routinely glues the tail of one
section to the start of the next — e.g. a budget section landing in a chunk
that *leads* with the prior section's boilerplate. A cross-encoder then scores
that chunk on its dominant (leading) content and buries the section that
actually answers the query. Splitting on section boundaries keeps each
section's content in a chunk that leads with that section's heading, so it
scores for what it contains.

When a section is larger than `chunk_size`, its heading is prepended to every
sub-chunk so heading vocabulary stays attached to table rows / list items that
would otherwise be isolated from their section's context.

Note: this changes chunk boundaries versus the previous fixed-size behavior.
Re-ingesting a tenant whose golden set was tuned to the old boundaries
(automotive, aerospace) should re-validate that golden set.
"""

from __future__ import annotations

import re

from graphrag.core.config import get_settings
from graphrag.core.models import Chunk, Document

# A heading line is either a markdown heading (`#`..`######`) or an all-caps
# numbered section header. The numbered branch requires the whole title after
# the number to be upper-case (letters/digits/punctuation only) so it matches
# "3. BUDGET & KPI TARGETS" but never a prose line like "3. This is a sentence."
_HEADING_RE = re.compile(
    r"^(#{1,6}\s+.+|\d{1,2}\.\s+[A-Z][A-Z0-9 &/,'\"()-]{1,60})\s*$",
    re.MULTILINE,
)


def _heading_boundaries(raw_text: str) -> tuple[list[int], dict[int, str]]:
    """Return (sorted start offsets of each section, {offset: heading text})."""
    offsets: list[int] = []
    headings: dict[int, str] = {}
    for m in _HEADING_RE.finditer(raw_text):
        offsets.append(m.start())
        headings[m.start()] = m.group(1).strip()
    return offsets, headings


def chunk_document(document: Document) -> list[Chunk]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    cfg = get_settings().ingestion
    chunk_size = cfg.get("chunk_size", 512)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=cfg.get("chunk_overlap", 64),
        length_function=len,
    )

    raw_text = document.raw_text
    offsets, headings = _heading_boundaries(raw_text)

    # Section boundaries: every heading start, plus 0 (preamble before the first
    # heading) and the end of the document.
    bounds = list(offsets)
    if not bounds or bounds[0] != 0:
        bounds.insert(0, 0)
    bounds.append(len(raw_text))

    texts: list[str] = []
    for i in range(len(bounds) - 1):
        segment = raw_text[bounds[i] : bounds[i + 1]].strip()
        if not segment:
            continue
        heading = headings.get(bounds[i])
        if len(segment) <= chunk_size:
            texts.append(segment)
        else:
            # Section too large — sub-split and keep the heading on each piece.
            for sub in splitter.split_text(segment):
                if heading and not sub.lstrip().startswith(heading):
                    sub = f"{heading}\n\n{sub}"
                texts.append(sub)

    return [
        Chunk(
            document_id=document.id,
            text=text,
            chunk_index=i,
            tenant=document.tenant,
        )
        for i, text in enumerate(texts)
    ]
