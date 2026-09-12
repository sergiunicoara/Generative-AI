"""Row-level checkpoint primitives for incremental relational ingestion.

Row-level counterpart to `graphrag/ingestion/incremental.py`'s whole-
snapshot hash: that module answers "did anything in this source change at
all"; this one answers "which specific rows are new/changed, and which
disappeared" -- the two questions a durable, crash-safe incremental
ingest needs answered separately. See
`graphrag/ingestion/relational.py`'s `RelationalGraphIngestor.ingest_incremental()`
for how this is wired into an actual durable checkpoint (persisted via
`graphrag/graph/neo4j_client.py`'s `RelationalSourceCheckpoint` methods).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from graphrag.core.content_hash import compute_content_hash


def compute_row_hash(row: dict[str, Any]) -> str:
    """SHA-256 of one row's canonical JSON -- same approach
    `graphrag/ingestion/incremental.py`'s `compute_relational_snapshot_hash`
    already uses for a whole snapshot, applied per row instead."""
    return compute_content_hash(json.dumps(row, sort_keys=True, default=str))


@dataclass(frozen=True)
class RowDiff:
    """Which row keys are new-or-changed, gone, or identical between two
    checkpoints. Row keys are the caller's own identifiers (this module has
    no opinion on their shape) -- `RelationalGraphIngestor` uses
    `"{table}:{source_key}"`."""

    upserted: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)


def diff_rows(previous: dict[str, str], current: dict[str, str]) -> RowDiff:
    """Compare a previous checkpoint's `{row_key: row_hash}` against the
    current run's, in one pass.

    A row key present in both with the same hash is unchanged. Present in
    both with a different hash, or present only in `current`, is upserted.
    Present only in `previous` is deleted -- the row disappeared from the
    source between runs.
    """
    upserted = sorted(key for key, current_hash in current.items()
                       if previous.get(key) != current_hash)
    deleted = sorted(key for key in previous if key not in current)
    unchanged = sorted(key for key, current_hash in current.items()
                        if previous.get(key) == current_hash)
    return RowDiff(upserted=upserted, deleted=deleted, unchanged=unchanged)


__all__ = ["RowDiff", "compute_row_hash", "diff_rows"]
