"""Section-aware semantic chunking.

Documents are split at section-heading boundaries first, then each section is
kept whole when it fits in `chunk_size` (or sub-split with LangChain's
RecursiveCharacterTextSplitter when it doesn't). A "heading" is a markdown
heading (lines starting with 1-6 `#`), an all-caps numbered section header
(e.g. "3. BUDGET & KPI TARGETS", "10. INDICATORI KPI"), or a plain
un-numbered all-caps title line (e.g. "CRITICAL FINDING", "REQUIRED
ACTIONS:") — reports and regulatory documents routinely title sections this
way with no leading number, and previously fell entirely to naive
fixed-size splitting as a result. See CON-02 in evals/golden_set.json for a
case this caused: a 512-char boundary landed exactly between a compliance
status line and the reconciling sentence one paragraph below it, in a
section-titled document with zero numbered headings.

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

# A heading line is a markdown heading (`#`..`######`), an all-caps numbered
# section header, or a plain all-caps title line. Each all-caps branch
# requires the whole line to be upper-case (letters/digits/punctuation only)
# so it matches "3. BUDGET & KPI TARGETS" / "CRITICAL FINDING" but never a
# prose line like "3. This is a sentence." or "FAA has approved an AMOC."
#
# The un-numbered branch additionally requires the line to either have no
# colon or end with one (`:?$`), which is what distinguishes a bare section
# title ("REQUIRED ACTIONS:") from an all-caps metadata line with a value
# after the colon ("MSN: 44567") — the latter must stay naive-split content,
# not become a spurious one-line "section."
_HEADING_RE = re.compile(
    r"^(#{1,6}\s+.+"
    r"|\d{1,2}\.\s+[A-Z][A-Z0-9 &/,'\"().-]{1,60}"
    r"|[A-Z][A-Z0-9 &/,'\"()—-]{1,68}:?)"
    r"\s*$",
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
    from pathlib import Path

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    cfg = get_settings().ingestion
    chunk_size = cfg.get("chunk_size", 512)
    # A section only modestly over budget still gets force-split at whatever
    # paragraph boundary falls closest to chunk_size — which is often the
    # boundary between two paragraphs that belong together (a status line and
    # the sentence qualifying it, a finding and its resolution). Letting a
    # section run up to this soft cap before it gets split keeps such
    # sections intact; only genuinely long sections still get recursively
    # split. See CON-02 in evals/golden_set.json: a ~768-char section (1.5x
    # chunk_size) was being split exactly between a compliance status line
    # and the sentence immediately qualifying it.
    soft_cap = chunk_size * cfg.get("section_soft_cap_multiplier", 1.6)
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
        if len(segment) <= soft_cap:
            texts.append(segment)
        else:
            # Section too large — sub-split and keep the heading on each piece.
            for sub in splitter.split_text(segment):
                if heading and not sub.lstrip().startswith(heading):
                    sub = f"{heading}\n\n{sub}"
                texts.append(sub)

    # Prepend the document's own identifier to every chunk. Sections that
    # only ever refer to their own document as "this AD" / "this directive"
    # (routine in REFERENCES/cross-reference sections — see INF-01 in
    # evals/golden_set.json) are otherwise unsearchable on that document's ID:
    # BM25 and embedding similarity for a query naming the document score
    # against a chunk that never contains the name. Filename is the
    # identifier convention already used throughout the corpus (e.g.
    # "FAA-AD-2024-01-02.txt" -> "FAA-AD-2024-01-02", matching
    # expected_citations exactly), so no new metadata is required.
    doc_label = Path(document.filename).stem
    texts = [f"[{doc_label}]\n\n{text}" for text in texts]

    return [
        Chunk(
            document_id=document.id,
            text=text,
            chunk_index=i,
            tenant=document.tenant,
            metadata={"source_system": document.metadata_envelope.source_system},
        )
        for i, text in enumerate(texts)
    ]
