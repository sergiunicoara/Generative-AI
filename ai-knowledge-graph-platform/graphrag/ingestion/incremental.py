"""Content-hash checkpoint for relational ingestion.

Ports the same skip/re-ingest/new three-way decision
`scripts/ingest_corpus.py` already applies to document ingestion (via
`graphrag/core/content_hash.py`) onto `RelationalGraphIngestor`, which today
always re-reads every source table and re-writes the full graph on every
call -- see `relational.py`'s `RelationalGraphIngestor.ingest()`, which never
computes or checks a content hash at all.
"""

from __future__ import annotations

import json
from typing import Any

from graphrag.core.content_hash import compute_content_hash, content_changed


def compute_relational_snapshot_hash(rows: list[dict[str, Any]]) -> str:
    """Deterministic content hash of a fetched relational snapshot.

    `rows` should be every row read from every table this ingestion run
    touches -- entity tables and relation tables together, so a change to
    either is detected, not only a change to entity data. Canonicalised via
    sort_keys JSON (the same approach `RelationalGraphIngestor.ingest()`
    already uses for the Document.raw_text payload) so field order never
    produces a spurious hash change, and hashed with the same
    `compute_content_hash` document ingestion uses -- one hash function for
    "did this source change", not two independently-maintained ones.
    """
    canonical = json.dumps(rows, sort_keys=True, default=str)
    return compute_content_hash(canonical)


def should_skip_ingest(previous_hash: str | None, current_hash: str) -> bool:
    """True when this snapshot is unchanged from the last recorded ingest.

    Delegates to `content_changed()` rather than re-implementing its
    semantics: a missing/blank `previous_hash` means "assume changed, do
    not skip" -- exactly the rule document ingestion already applies, so a
    relational source ingested before this checkpoint existed is re-ingested
    once rather than silently frozen out forever.
    """
    return not content_changed(previous_hash or "", current_hash)


__all__ = ["compute_relational_snapshot_hash", "should_skip_ingest"]
