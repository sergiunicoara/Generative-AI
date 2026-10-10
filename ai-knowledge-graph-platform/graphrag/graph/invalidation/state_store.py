"""Durable state of derived artifacts and of invalidation events.

``(:DerivedArtifactState {tenant, kind, artifact_id, state, version})`` sits
beside the artifact instead of editing it: CGDecision / CGContextManifest are
integrity-hashed and immutable, and a snapshot keeps its original summary. Each
invalidation adds ``(state)-[:INVALIDATED_BY]->(:InvalidationEvent)``, which is
the trace of why an artifact became stale. ``version`` increases on every
transition, so a recompute that finishes after a newer invalidation cannot
overwrite it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from graphrag.graph.invalidation.models import (
    COMPLETION_STATES,
    ArtifactKind,
    ArtifactState,
    InvalidationEvent,
)
from graphrag.graph.validation.graph_checks import run_read_only

_UPSERT_EVENT = """
MERGE (e:InvalidationEvent {tenant: $tenant, id: $id})
ON CREATE SET e.kind = $kind, e.reason = $reason, e.actor = $actor, e.cause = $cause,
              e.subjects_json = $subjects, e.additive = $additive,
              e.occurred_at = datetime($occurred_at), e.created_at = datetime($now),
              e.status = 'PROCESSING'
RETURN e.status AS status, e.summary_json AS summary_json, (e.created_at = datetime($now)) AS created
"""

_FINISH_EVENT = """
MATCH (e:InvalidationEvent {tenant: $tenant, id: $id})
SET e.status = 'PROCESSED', e.processed_at = datetime(), e.summary_json = $summary
"""

_MARK = """
MATCH (e:InvalidationEvent {tenant: $tenant, id: $event_id})
UNWIND $ids AS artifact_id
MERGE (a:DerivedArtifactState {tenant: $tenant, kind: $kind, artifact_id: artifact_id})
ON CREATE SET a.state = 'VALID', a.version = 0, a.created_at = datetime()
WITH a, e, EXISTS { (a)-[:INVALIDATED_BY]->(e) } AS seen
MERGE (a)-[:INVALIDATED_BY]->(e)
WITH a, seen WHERE NOT seen
SET a.previous_state = a.state, a.state = 'NEEDS_REVIEW', a.version = a.version + 1,
    a.updated_at = datetime(), a.last_event_id = $event_id, a.reason = $reason,
    a.requires_human_review = false
RETURN count(a) AS marked
"""

# Visible marker on the artifact itself (no content change) so readers can tell
# an inferred edge / snapshot is under review.
_FLAG_INFERRED = """
UNWIND $keys AS k
MATCH (s:Entity {tenant: $tenant, name: k.src_name, type: k.src_type})
      -[r:RELATES_TO {relation: k.relation}]->
      (o:Entity {tenant: $tenant, name: k.tgt_name, type: k.tgt_type})
WHERE r.tenant = $tenant AND r.source_type = 'inferred'
SET r.review_state = 'NEEDS_REVIEW'
"""

_FLAG_SNAPSHOTS = """
MATCH (s:CommunitySummarySnapshot {tenant: $tenant}) WHERE s.id IN $ids
SET s.review_state = 'NEEDS_REVIEW'
"""

_CLAIM = """
MATCH (a:DerivedArtifactState {tenant: $tenant, state: 'NEEDS_REVIEW'})
WHERE coalesce(a.requires_human_review, false) = false
  AND ($kinds IS NULL OR a.kind IN $kinds)
WITH a ORDER BY a.updated_at LIMIT $limit
SET a.state = 'RECOMPUTING', a.version = a.version + 1, a.claimed_at = datetime()
RETURN a.kind AS kind, a.artifact_id AS artifact_id, a.version AS version, a.last_event_id AS event_id
"""

_COMPLETE = """
MATCH (a:DerivedArtifactState {tenant: $tenant, kind: $kind, artifact_id: $artifact_id})
WHERE a.state = 'RECOMPUTING' AND a.version = $version
SET a.previous_state = a.state, a.state = $state, a.version = a.version + 1,
    a.updated_at = datetime(), a.detail = $detail, a.requires_human_review = $human
RETURN count(a) AS n
"""


class StateStore:
    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    async def begin_event(self, event: InvalidationEvent) -> tuple[bool, dict | None]:
        """Returns (should_process, previous_summary). A processed event is a no-op."""
        now = datetime.now(timezone.utc).isoformat()
        rows = await self._neo4j.run(
            _UPSERT_EVENT, tenant=event.tenant, id=event.id, kind=event.kind.value,
            reason=event.reason, actor=event.actor, cause=event.cause,
            subjects=event.subjects_json(), additive=event.additive,
            occurred_at=event.occurred_at.isoformat(), now=now,
        )
        row = rows[0] if rows else {}
        if row.get("status") == "PROCESSED":
            return False, json.loads(row.get("summary_json") or "{}")
        return True, None

    async def finish_event(self, event: InvalidationEvent, summary: dict) -> None:
        await self._neo4j.run(_FINISH_EVENT, tenant=event.tenant, id=event.id,
                              summary=json.dumps(summary, sort_keys=True))

    async def mark(self, event: InvalidationEvent, kind: ArtifactKind, ids: list[str]) -> int:
        if not ids:
            return 0
        rows = await self._neo4j.run(_MARK, tenant=event.tenant, event_id=event.id, kind=kind.value,
                                     ids=sorted(set(ids)), reason=f"{event.kind.value}: {event.reason}"[:300])
        return int(rows[0]["marked"]) if rows else 0

    async def flag_inferred(self, tenant: str, edges: list[dict]) -> None:
        if edges:
            await self._neo4j.run(_FLAG_INFERRED, tenant=tenant, keys=[
                {k: e[k] for k in ("src_name", "src_type", "relation", "tgt_name", "tgt_type")}
                for e in edges])

    async def flag_snapshots(self, tenant: str, ids: list[str]) -> None:
        if ids:
            await self._neo4j.run(_FLAG_SNAPSHOTS, tenant=tenant, ids=sorted(ids))

    async def claim(self, tenant: str, *, limit: int = 50, kinds: list[str] | None = None) -> list[dict]:
        return await self._neo4j.run(_CLAIM, tenant=tenant, limit=int(limit), kinds=kinds)

    async def complete(self, tenant: str, claimed: dict, state: ArtifactState, detail: str = "",
                       *, human: bool = False) -> bool:
        if state not in COMPLETION_STATES:
            raise ValueError(f"{state} is not a completion state")
        rows = await self._neo4j.run(
            _COMPLETE, tenant=tenant, kind=claimed["kind"], artifact_id=claimed["artifact_id"],
            version=claimed["version"], state=state.value, detail=detail[:500], human=human)
        return bool(rows and rows[0]["n"])

    async def get(self, tenant: str, kind: str, artifact_id: str) -> dict | None:
        rows = await run_read_only(
            self._neo4j,
            """
            MATCH (a:DerivedArtifactState {tenant: $tenant, kind: $kind, artifact_id: $id})
            OPTIONAL MATCH (a)-[:INVALIDATED_BY]->(e:InvalidationEvent)
            RETURN a.state AS state, a.version AS version, a.detail AS detail,
                   a.requires_human_review AS requires_human_review,
                   collect({id: e.id, kind: e.kind, reason: e.reason, cause: e.cause}) AS events
            """,
            tenant=tenant, kind=kind, id=artifact_id,
        )
        return rows[0] if rows else None

    async def list(self, tenant: str, *, state: str | None = None, kind: str | None = None,
                   limit: int = 100) -> list[dict]:
        return await run_read_only(
            self._neo4j,
            """
            MATCH (a:DerivedArtifactState {tenant: $tenant})
            WHERE ($state IS NULL OR a.state = $state) AND ($kind IS NULL OR a.kind = $kind)
            RETURN a.kind AS kind, a.artifact_id AS artifact_id, a.state AS state,
                   a.version AS version, a.reason AS reason, a.detail AS detail,
                   a.requires_human_review AS requires_human_review,
                   toString(a.updated_at) AS updated_at
            ORDER BY a.updated_at DESC LIMIT $limit
            """,
            tenant=tenant, state=state, kind=kind, limit=max(1, min(int(limit), 1000)),
        )
