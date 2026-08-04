"""§11 — 'Never claim restart safety with process memory only.' Proves the MVP
in-process store's limitation directly rather than only describing it in prose.
"""

from __future__ import annotations

from datetime import datetime, timezone

from api.state import IngestionJob, InMemoryIngestionStore
from src.domain.enums import IngestionState


def _job(ingestion_id: str = "job-1") -> IngestionJob:
    now = datetime.now(timezone.utc)
    return IngestionJob(
        ingestion_id=ingestion_id, workspace_id="ws-1", kind="crm",
        state=IngestionState.COMPLETED, created_at=now, updated_at=now,
    )


def test_store_persists_within_the_same_instance():
    store = InMemoryIngestionStore()
    job = _job()
    store.put(job)
    assert store.get("job-1") is job


def test_store_does_not_survive_a_new_instance():
    """A new InMemoryIngestionStore() stands in for a process restart — it has
    no memory of jobs recorded by a previous instance."""
    store_before_restart = InMemoryIngestionStore()
    store_before_restart.put(_job())

    store_after_restart = InMemoryIngestionStore()
    assert store_after_restart.get("job-1") is None


def test_unknown_ingestion_id_returns_none():
    store = InMemoryIngestionStore()
    assert store.get("never-existed") is None
