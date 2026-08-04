"""Tenant-safe repository for SourceRecord/SourceSnapshot — same pattern as
crm_repository.py. Backs src/ingestion/reconciliation.py's write path.
"""

from __future__ import annotations

from src.domain.crm import SourceRecord, SourceSnapshot
from src.graph.execution import GraphExecutor, scoped_match

_RECORD_RETURN = (
    "r.source_record_id AS source_record_id, r.workspace_id AS workspace_id, "
    "r.source_system AS source_system, r.object_type AS object_type, "
    "r.external_id AS external_id, r.source_status AS source_status, "
    "r.first_seen_at AS first_seen_at, r.last_seen_at AS last_seen_at, "
    "r.current_snapshot_id AS current_snapshot_id"
)

_SNAPSHOT_RETURN = (
    "s.snapshot_id AS snapshot_id, s.source_record_id AS source_record_id, "
    "s.source_version AS source_version, s.content_hash AS content_hash, "
    "s.ingestion_run_id AS ingestion_run_id, s.captured_at AS captured_at, "
    "s.superseded AS superseded"
)


class SourceRepository:
    def __init__(self, executor: GraphExecutor | None = None):
        self._executor = executor or GraphExecutor()

    async def upsert_source_record(self, record: SourceRecord) -> None:
        match = scoped_match("SourceRecord", "r", source_record_id="source_record_id")
        await self._executor.tenant_query(
            f"""
            MERGE {match}
            SET r.source_system = $source_system,
                r.object_type = $object_type,
                r.external_id = $external_id,
                r.source_status = $source_status,
                r.first_seen_at = $first_seen_at,
                r.last_seen_at = $last_seen_at,
                r.current_snapshot_id = $current_snapshot_id
            """,
            workspace_id=record.workspace_id,
            source_record_id=record.source_record_id,
            source_system=record.source_system,
            object_type=record.object_type,
            external_id=record.external_id,
            source_status=record.source_status.value,
            first_seen_at=record.first_seen_at.isoformat(),
            last_seen_at=record.last_seen_at.isoformat(),
            current_snapshot_id=record.current_snapshot_id,
        )

    async def get_source_record(self, workspace_id: str, source_record_id: str) -> SourceRecord | None:
        match = scoped_match("SourceRecord", "r", source_record_id="source_record_id")
        rows = await self._executor.tenant_query(
            f"MATCH {match} RETURN {_RECORD_RETURN}",
            workspace_id=workspace_id,
            source_record_id=source_record_id,
        )
        return SourceRecord(**rows[0]) if rows else None

    async def create_snapshot(self, workspace_id: str, snapshot: SourceSnapshot) -> None:
        """SourceSnapshot carries no workspace_id of its own — it's reached only
        by traversal from a workspace-scoped SourceRecord (same routing principle
        §10 applies to Claim/Mention: never fan out from a bare, unscoped node)."""
        record_match = scoped_match("SourceRecord", "r", source_record_id="source_record_id")
        await self._executor.tenant_query(
            f"""
            MATCH {record_match}
            MERGE (s:SourceSnapshot {{snapshot_id: $snapshot_id}})
            SET s.source_record_id = $source_record_id,
                s.source_version = $source_version,
                s.content_hash = $content_hash,
                s.ingestion_run_id = $ingestion_run_id,
                s.captured_at = $captured_at,
                s.superseded = $superseded
            MERGE (r)-[:HAS_SNAPSHOT]->(s)
            """,
            workspace_id=workspace_id,
            source_record_id=snapshot.source_record_id,
            snapshot_id=snapshot.snapshot_id,
            source_version=snapshot.source_version,
            content_hash=snapshot.content_hash,
            ingestion_run_id=snapshot.ingestion_run_id,
            captured_at=snapshot.captured_at.isoformat(),
            superseded=snapshot.superseded,
        )

    async def get_latest_snapshot(self, workspace_id: str, source_record_id: str) -> SourceSnapshot | None:
        record_match = scoped_match("SourceRecord", "r", source_record_id="source_record_id")
        rows = await self._executor.tenant_query(
            f"""
            MATCH {record_match}
            MATCH (r)-[:HAS_SNAPSHOT]->(s:SourceSnapshot)
            RETURN {_SNAPSHOT_RETURN}
            ORDER BY s.source_version DESC
            LIMIT 1
            """,
            workspace_id=workspace_id,
            source_record_id=source_record_id,
        )
        return SourceSnapshot(**rows[0]) if rows else None

    async def list_snapshots(self, workspace_id: str, source_record_id: str) -> list[SourceSnapshot]:
        """Full version history, newest first — used to verify that superseding a
        snapshot preserves rather than overwrites the prior one (§6)."""
        record_match = scoped_match("SourceRecord", "r", source_record_id="source_record_id")
        rows = await self._executor.tenant_query(
            f"""
            MATCH {record_match}
            MATCH (r)-[:HAS_SNAPSHOT]->(s:SourceSnapshot)
            RETURN {_SNAPSHOT_RETURN}
            ORDER BY s.source_version DESC
            """,
            workspace_id=workspace_id,
            source_record_id=source_record_id,
        )
        return [SourceSnapshot(**row) for row in rows]

    async def mark_snapshot_superseded(self, workspace_id: str, source_record_id: str, snapshot_id: str) -> None:
        record_match = scoped_match("SourceRecord", "r", source_record_id="source_record_id")
        await self._executor.tenant_query(
            f"""
            MATCH {record_match}
            MATCH (r)-[:HAS_SNAPSHOT]->(s:SourceSnapshot {{snapshot_id: $snapshot_id}})
            SET s.superseded = true
            """,
            workspace_id=workspace_id,
            source_record_id=source_record_id,
            snapshot_id=snapshot_id,
        )
