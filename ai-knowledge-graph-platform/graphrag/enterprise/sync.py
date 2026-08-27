"""Provider-neutral content synchronization plane.

Connectors translate webhooks, delta APIs or full scans into ``SyncChange``
records.  The plane persists cursors and runs independently from retrieval, but
publishes normal ``Document`` messages so extraction and downstream graph views
receive exactly the same content revision.
"""

from __future__ import annotations

from uuid import uuid4

from graphrag.core.config import get_settings
from graphrag.core.models import Document
from graphrag.enterprise.models import SyncChange, SyncChangeType
from graphrag.graph.neo4j_client import get_neo4j
from graphrag.messaging.publishers import publish_document


class ContentSyncService:
    def __init__(self, neo4j_client=None, publisher=publish_document):
        self._neo4j = neo4j_client or get_neo4j()
        self._publisher = publisher

    async def apply_changes(
        self, source_id: str, changes: list[SyncChange], tenant: str, *, cursor: str = "",
        trigger: str = "delta",
    ) -> dict:
        """Apply webhook/delta changes and advance the durable source cursor."""
        run_id = str(uuid4())
        await self._ensure_source(source_id, tenant)
        queued = 0
        deleted = 0
        for change in changes:
            if change.change_type == SyncChangeType.DELETE:
                deleted += await self._tombstone_external_ids(source_id, [change.external_id], tenant)
                continue
            # ``external_id`` is the source-system identity used by delta
            # deletes and full reconciliation.  Keep the top-level change
            # identity authoritative even when a connector omitted it from
            # its metadata envelope.
            metadata = change.metadata.model_copy(update={
                "external_id": change.metadata.external_id or change.external_id,
            })
            doc = Document(
                filename=change.filename,
                source_path=metadata.source_url or change.filename,
                raw_text=change.text,
                metadata_envelope=metadata,
                access_policy=change.access_policy,
                outbound_links=change.document_links,
                lineage_assertions=change.lineage,
                obligation_drafts=change.obligations,
                valid_from=metadata.effective_from,
                valid_to=metadata.effective_to,
                tenant=tenant,
                source_id=source_id,
            )
            await self._publisher(doc, priority="high")
            queued += 1
        effective_cursor = cursor or next((c.cursor for c in reversed(changes) if c.cursor), "")
        await self._neo4j.run(
            """
            MATCH (s:ContentSyncSource {id: $source_id, tenant: $tenant})
            SET s.delta_cursor = CASE WHEN $cursor = '' THEN s.delta_cursor ELSE $cursor END,
                s.last_sync_at = datetime(), s.last_sync_trigger = $trigger,
                s.updated_at = datetime()
            CREATE (r:ContentSyncRun {
                id: $run_id, tenant: $tenant, source_id: $source_id,
                trigger: $trigger, queued: $queued, deleted: $deleted,
                cursor: $cursor, started_at: datetime(), completed_at: datetime()
            })
            MERGE (r)-[:FOR_SOURCE]->(s)
            """,
            run_id=run_id, source_id=source_id, tenant=tenant, trigger=trigger,
            queued=queued, deleted=deleted, cursor=effective_cursor,
        )
        return {"run_id": run_id, "queued": queued, "tombstoned": deleted, "cursor": effective_cursor}

    async def reconcile(
        self, source_id: str, discovered_external_ids: list[str], tenant: str,
    ) -> dict:
        """Full-review reconciliation: soft-delete missing source items only."""
        await self._ensure_source(source_id, tenant)
        rows = await self._neo4j.run(
            """
            MATCH (d:Document {tenant: $tenant, source_id: $source_id})
            WHERE coalesce(d.external_id, '') <> ''
              AND NOT d.external_id IN $discovered_external_ids
              AND coalesce(d.is_deleted, false) = false
            SET d.is_deleted = true, d.deleted_at = datetime(),
                d.sync_tombstone_reason = 'full_reconciliation'
            RETURN count(d) AS tombstoned
            """,
            source_id=source_id,
            tenant=tenant,
            discovered_external_ids=sorted(set(discovered_external_ids)),
        )
        tombstoned = int(rows[0].get("tombstoned", 0)) if rows else 0
        interval = int(get_settings().content_sync.get("full_review_interval_seconds", 604800))
        await self._neo4j.run(
            """
            MATCH (s:ContentSyncSource {id: $source_id, tenant: $tenant})
            SET s.last_full_review_at = datetime(),
                s.next_full_review_at = datetime() + duration({seconds: $interval}),
                s.updated_at = datetime()
            """,
            source_id=source_id, tenant=tenant, interval=interval,
        )
        return {"source_id": source_id, "tombstoned": tombstoned}

    async def due_full_reviews(self, tenant: str) -> list[dict]:
        return await self._neo4j.run(
            """
            MATCH (s:ContentSyncSource {tenant: $tenant})
            WHERE s.next_full_review_at IS NULL OR s.next_full_review_at <= datetime()
            RETURN s.id AS source_id, s.delta_cursor AS cursor,
                   s.last_full_review_at AS last_full_review_at,
                   s.next_full_review_at AS next_full_review_at
            ORDER BY s.next_full_review_at ASC
            """,
            tenant=tenant,
        )

    async def current_cursor(self, source_id: str, tenant: str) -> str:
        """Return the durable provider cursor without exposing Neo4j to connectors."""
        rows = await self._neo4j.run(
            """
            MATCH (s:ContentSyncSource {id: $source_id, tenant: $tenant})
            RETURN coalesce(s.delta_cursor, '') AS cursor
            LIMIT 1
            """,
            source_id=source_id, tenant=tenant,
        )
        return str(rows[0].get("cursor", "")) if rows else ""

    async def sources(self, tenant: str) -> list[dict]:
        return await self._neo4j.run(
            """
            MATCH (s:ContentSyncSource {tenant: $tenant})
            RETURN s {.*} AS source
            ORDER BY s.id
            """,
            tenant=tenant,
        )

    async def _ensure_source(self, source_id: str, tenant: str) -> None:
        await self._neo4j.run(
            """
            MERGE (s:ContentSyncSource {id: $source_id, tenant: $tenant})
            ON CREATE SET s.created_at = datetime(), s.delta_cursor = ''
            SET s.updated_at = datetime()
            """,
            source_id=source_id, tenant=tenant,
        )

    async def _tombstone_external_ids(self, source_id: str, external_ids: list[str], tenant: str) -> int:
        rows = await self._neo4j.run(
            """
            MATCH (d:Document {tenant: $tenant, source_id: $source_id})
            WHERE d.external_id IN $external_ids AND coalesce(d.is_deleted, false) = false
            SET d.is_deleted = true, d.deleted_at = datetime(), d.sync_tombstone_reason = 'delta_delete'
            RETURN count(d) AS tombstoned
            """,
            source_id=source_id, tenant=tenant, external_ids=external_ids,
        )
        return int(rows[0].get("tombstoned", 0)) if rows else 0
