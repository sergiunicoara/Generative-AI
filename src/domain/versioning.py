"""§6 source-version reconciliation, as pure state-transition functions over
src/domain/crm.py's SourceRecord/SourceSnapshot — no I/O, no Neo4j. The actual
write path (persisting the transition, touching dependent Claims) is Increment 4's
(P2) src/ingestion/reconciliation.py; this module is only the decision logic:
identical content is a no-op, changed content supersedes, and a record is
tombstoned only when the caller has already established a trustworthy deletion
signal — the "trustworthy" judgment itself belongs to the P2 adapter layer
(e.g. an explicit `supports_deletion_signal` capability flag), not here.
"""

from __future__ import annotations

from datetime import datetime

from src.domain.crm import SourceRecord, SourceSnapshot
from src.domain.enums import SourceStatus


def is_identical(prev_snapshot: SourceSnapshot | None, new_content_hash: str) -> bool:
    """§6 — 'Identical content is a no-op.' True only when a prior snapshot exists
    and its content_hash matches exactly."""
    if prev_snapshot is None:
        return False
    return prev_snapshot.content_hash == new_content_hash


def supersede(
    prev: SourceSnapshot,
    *,
    new_snapshot_id: str,
    new_content_hash: str,
    ingestion_run_id: str,
    captured_at: datetime,
) -> tuple[SourceSnapshot, SourceSnapshot]:
    """§6 — 'Changed content creates a new source snapshot and triggers
    reconciliation.' Returns (superseded_prev, new_snapshot). Never mutates `prev`
    in place — SourceSnapshot is frozen.
    """
    if prev.content_hash == new_content_hash:
        raise ValueError("supersede() called with identical content_hash — use is_identical() first")

    superseded_prev = prev.model_copy(update={"superseded": True})
    new_snapshot = SourceSnapshot(
        snapshot_id=new_snapshot_id,
        source_record_id=prev.source_record_id,
        source_version=prev.source_version + 1,
        content_hash=new_content_hash,
        ingestion_run_id=ingestion_run_id,
        captured_at=captured_at,
        superseded=False,
    )
    return superseded_prev, new_snapshot


def tombstone(record: SourceRecord, *, observed_at: datetime) -> SourceRecord:
    """§6 — 'Deleted or missing source records are tombstoned only when the
    adapter supplies a trustworthy deletion or complete-snapshot signal.'

    This function performs the state transition unconditionally — the caller
    (P2's reconciliation layer) is responsible for having already established
    that the deletion signal is trustworthy before calling it. It is not this
    pure function's job to re-derive adapter trust; that would require I/O this
    package deliberately has none of.
    """
    if record.source_status == SourceStatus.DELETED:
        return record
    return record.model_copy(update={
        "source_status": SourceStatus.DELETED,
        "last_seen_at": observed_at,
    })
