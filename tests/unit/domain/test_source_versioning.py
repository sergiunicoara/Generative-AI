"""§6 — 'Identical content is a no-op. Changed content creates a new source
snapshot and triggers reconciliation. Deleted/missing source records are
tombstoned...' Pure state-transition tests over src/domain/versioning.py, no DB.
"""

from datetime import datetime, timezone

import pytest

from src.domain.crm import SourceRecord, SourceSnapshot
from src.domain.enums import SourceStatus
from src.domain.versioning import is_identical, supersede, tombstone

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_T1 = datetime(2026, 1, 2, tzinfo=timezone.utc)


def _snapshot(**overrides) -> SourceSnapshot:
    base = dict(
        snapshot_id="snap-1",
        source_record_id="rec-1",
        source_version=1,
        content_hash="hash-a",
        ingestion_run_id="run-1",
        captured_at=_T0,
        superseded=False,
    )
    base.update(overrides)
    return SourceSnapshot(**base)


def _record(**overrides) -> SourceRecord:
    base = dict(
        source_record_id="rec-1",
        workspace_id="ws-1",
        source_system="salesforce",
        object_type="Account",
        external_id="001xx",
        source_status=SourceStatus.ACTIVE,
        first_seen_at=_T0,
        last_seen_at=_T0,
        current_snapshot_id="snap-1",
    )
    base.update(overrides)
    return SourceRecord(**base)


def test_identical_content_is_a_no_op():
    prev = _snapshot(content_hash="hash-a")
    assert is_identical(prev, "hash-a") is True


def test_changed_content_is_not_identical():
    prev = _snapshot(content_hash="hash-a")
    assert is_identical(prev, "hash-b") is False


def test_no_prior_snapshot_is_never_identical():
    assert is_identical(None, "hash-a") is False


def test_supersede_marks_prior_snapshot_superseded_and_bumps_version():
    prev = _snapshot(content_hash="hash-a", source_version=1, superseded=False)

    superseded_prev, new_snapshot = supersede(
        prev,
        new_snapshot_id="snap-2",
        new_content_hash="hash-b",
        ingestion_run_id="run-2",
        captured_at=_T1,
    )

    assert superseded_prev.superseded is True
    assert superseded_prev.content_hash == "hash-a"  # prior content preserved, not overwritten
    assert new_snapshot.superseded is False
    assert new_snapshot.content_hash == "hash-b"
    assert new_snapshot.source_version == 2
    assert new_snapshot.source_record_id == prev.source_record_id


def test_supersede_rejects_identical_content():
    prev = _snapshot(content_hash="hash-a")
    with pytest.raises(ValueError, match="identical content_hash"):
        supersede(
            prev,
            new_snapshot_id="snap-2",
            new_content_hash="hash-a",
            ingestion_run_id="run-2",
            captured_at=_T1,
        )


def test_tombstone_marks_record_deleted():
    record = _record(source_status=SourceStatus.ACTIVE)
    tombstoned = tombstone(record, observed_at=_T1)
    assert tombstoned.source_status == SourceStatus.DELETED
    assert tombstoned.last_seen_at == _T1
    # original record is untouched — SourceRecord is frozen
    assert record.source_status == SourceStatus.ACTIVE


def test_tombstone_is_idempotent():
    record = _record(source_status=SourceStatus.DELETED, last_seen_at=_T0)
    tombstoned = tombstone(record, observed_at=_T1)
    assert tombstoned.source_status == SourceStatus.DELETED
    assert tombstoned.last_seen_at == _T0  # already-deleted record is returned unchanged
