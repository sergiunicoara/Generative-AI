"""§6 — the real write path around src/domain/versioning.py's pure functions.
'Identical content is a no-op. Changed content creates a new source snapshot and
triggers reconciliation. Deleted/missing source records are tombstoned only when
the adapter supplies a trustworthy deletion or complete-snapshot signal.'

The 'trustworthy' judgment is the caller's (the adapter declares
`supports_deletion_signal`, per src/ingestion/adapters/*.py) — this module
enforces that the judgment was actually made, not what it should be.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from src.domain.crm import SourceRecord, SourceSnapshot
from src.domain.enums import SourceStatus
from src.domain.identity import crm_entity_id, snapshot_id as _snapshot_id
from src.domain.versioning import is_identical, supersede, tombstone
from src.graph.repositories.source_repository import SourceRepository


class ReconciliationOutcome(StrEnum):
    CREATED = "CREATED"
    NO_OP = "NO_OP"
    SUPERSEDED = "SUPERSEDED"
    TOMBSTONED = "TOMBSTONED"


@dataclass(frozen=True)
class ReconciliationResult:
    outcome: ReconciliationOutcome
    source_record_id: str
    snapshot_id: str | None = None


async def reconcile_source_record(
    source_repo: SourceRepository,
    *,
    workspace_id: str,
    source_system: str,
    object_type: str,
    external_id: str,
    content_hash: str,
    ingestion_run_id: str,
    observed_at: datetime,
) -> ReconciliationResult:
    """Idempotent per (workspace, source_system, object_type, external_id,
    content_hash) — re-running with identical content is always NO_OP, never a
    duplicate SourceRecord/SourceSnapshot."""
    source_record_id = crm_entity_id(workspace_id, source_system, object_type, external_id)
    existing = await source_repo.get_source_record(workspace_id, source_record_id)

    if existing is None:
        snap_id = _snapshot_id(source_record_id, 1)
        record = SourceRecord(
            source_record_id=source_record_id,
            workspace_id=workspace_id,
            source_system=source_system,
            object_type=object_type,
            external_id=external_id,
            source_status=SourceStatus.ACTIVE,
            first_seen_at=observed_at,
            last_seen_at=observed_at,
            current_snapshot_id=snap_id,
        )
        snapshot = SourceSnapshot(
            snapshot_id=snap_id,
            source_record_id=source_record_id,
            source_version=1,
            content_hash=content_hash,
            ingestion_run_id=ingestion_run_id,
            captured_at=observed_at,
        )
        await source_repo.upsert_source_record(record)
        await source_repo.create_snapshot(workspace_id, snapshot)
        return ReconciliationResult(ReconciliationOutcome.CREATED, source_record_id, snap_id)

    latest = await source_repo.get_latest_snapshot(workspace_id, source_record_id)

    if is_identical(latest, content_hash):
        await source_repo.upsert_source_record(existing.model_copy(update={"last_seen_at": observed_at}))
        return ReconciliationResult(
            ReconciliationOutcome.NO_OP, source_record_id, latest.snapshot_id if latest else None
        )

    if latest is None:
        # existing SourceRecord with no snapshot at all shouldn't happen given
        # this module's own write path always creates one alongside the record —
        # treat defensively as "first snapshot" rather than crash on a state this
        # module never itself produces.
        next_version = 1
    else:
        next_version = latest.source_version + 1
    new_snap_id = _snapshot_id(source_record_id, next_version)

    if latest is not None:
        superseded_prev, new_snapshot = supersede(
            latest,
            new_snapshot_id=new_snap_id,
            new_content_hash=content_hash,
            ingestion_run_id=ingestion_run_id,
            captured_at=observed_at,
        )
        await source_repo.mark_snapshot_superseded(workspace_id, source_record_id, superseded_prev.snapshot_id)
    else:
        new_snapshot = SourceSnapshot(
            snapshot_id=new_snap_id,
            source_record_id=source_record_id,
            source_version=next_version,
            content_hash=content_hash,
            ingestion_run_id=ingestion_run_id,
            captured_at=observed_at,
        )

    await source_repo.create_snapshot(workspace_id, new_snapshot)
    await source_repo.upsert_source_record(existing.model_copy(update={
        "last_seen_at": observed_at,
        "current_snapshot_id": new_snap_id,
        "source_status": SourceStatus.ACTIVE,
    }))
    return ReconciliationResult(ReconciliationOutcome.SUPERSEDED, source_record_id, new_snap_id)


async def reconcile_deletion(
    source_repo: SourceRepository,
    *,
    workspace_id: str,
    source_system: str,
    object_type: str,
    external_id: str,
    observed_at: datetime,
    adapter_supports_deletion_signal: bool,
) -> ReconciliationResult:
    if not adapter_supports_deletion_signal:
        raise ValueError(
            "adapter does not declare a trustworthy deletion signal "
            "(supports_deletion_signal=False) — refusing to tombstone. §6: "
            "tombstoning requires the adapter to supply a trustworthy deletion "
            "or complete-snapshot signal."
        )
    source_record_id = crm_entity_id(workspace_id, source_system, object_type, external_id)
    existing = await source_repo.get_source_record(workspace_id, source_record_id)
    if existing is None:
        return ReconciliationResult(ReconciliationOutcome.NO_OP, source_record_id, None)

    tombstoned = tombstone(existing, observed_at=observed_at)
    await source_repo.upsert_source_record(tombstoned)
    return ReconciliationResult(ReconciliationOutcome.TOMBSTONED, source_record_id, tombstoned.current_snapshot_id)
