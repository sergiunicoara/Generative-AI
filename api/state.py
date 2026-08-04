"""§11 — 'For the MVP, an in-process bounded worker may be used, but its
interface must support later replacement with a durable queue.' This is that
in-process store — and its limitation is real, not hidden: a process restart
loses every job. See tests/unit/api/test_ingestion_store.py for the explicit
proof (never claim restart safety with process memory only, per the Codex
prompt's own non-negotiable rules). The get/put interface is shaped so a
Redis/Postgres-backed implementation can replace this class later without
changing any caller in api/routes/ingestions.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.domain.enums import IngestionState


@dataclass
class IngestionJob:
    ingestion_id: str
    workspace_id: str
    kind: str  # "crm" | "content-assets"
    state: IngestionState
    created_at: datetime
    updated_at: datetime
    item_results: list[dict] = field(default_factory=list)
    error: str | None = None


class InMemoryIngestionStore:
    def __init__(self):
        self._jobs: dict[str, IngestionJob] = {}

    def put(self, job: IngestionJob) -> None:
        self._jobs[job.ingestion_id] = job

    def get(self, ingestion_id: str) -> IngestionJob | None:
        return self._jobs.get(ingestion_id)
