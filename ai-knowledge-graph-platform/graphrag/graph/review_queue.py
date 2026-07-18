"""Human review queue for ambiguous entity alias matches.

When alias resolution finds a candidate in the ambiguous band (fuzzy 70-84
or embedding 0.85-0.92) it enqueues a ReviewQueueItem rather than silently
creating a duplicate node. A human reviewer then approves (merge → register
alias) or rejects (keep separate).

Architecture mirrors QuarantineService: Neo4j node + service class +
API endpoints. Tenant-scoped throughout.
"""

from __future__ import annotations

from uuid import uuid4

import structlog

from graphrag.graph.alias_registry import get_alias_registry
from graphrag.graph.neo4j_client import get_neo4j

log = structlog.get_logger(__name__)


class ReviewQueueService:
    """
    Manage the human review queue for ambiguous alias matches.

    Usage::

        svc = ReviewQueueService()
        item_id = await svc.enqueue(
            raw_name="ISO IATF", raw_type="CONCEPT",
            candidate_name="IATF 16949:2016", candidate_type="CONCEPT",
            score=75.0, match_type="fuzzy",
            source_doc="doc-abc", tenant="automotive",
        )
        await svc.approve(item_id, reviewed_by="admin", tenant="automotive")
    """

    def __init__(self, neo4j_client=None):
        self._neo4j = neo4j_client or get_neo4j()

    async def enqueue(
        self,
        raw_name: str,
        raw_type: str,
        candidate_name: str,
        candidate_type: str,
        score: float,
        match_type: str,
        source_doc: str,
        tenant: str,
    ) -> str:
        """Create a ReviewQueueItem node and return its item_id."""
        item_id = str(uuid4())
        await self._neo4j.run(
            """
            CREATE (r:ReviewQueueItem {
                id:             $item_id,
                tenant:         $tenant,
                raw_name:       $raw_name,
                raw_type:       $raw_type,
                candidate_name: $candidate_name,
                candidate_type: $candidate_type,
                score:          $score,
                match_type:     $match_type,
                source_doc:     $source_doc,
                status:         'pending',
                created_at:     datetime(),
                reviewed_by:    null,
                reviewed_at:    null
            })
            """,
            item_id=item_id,
            tenant=tenant,
            raw_name=raw_name,
            raw_type=raw_type,
            candidate_name=candidate_name,
            candidate_type=candidate_type,
            score=score,
            match_type=match_type,
            source_doc=source_doc,
        )
        log.info(
            "review_queue.enqueued",
            item_id=item_id,
            raw=raw_name,
            candidate=candidate_name,
            score=score,
            match_type=match_type,
            tenant=tenant,
        )
        return item_id

    async def approve(self, item_id: str, reviewed_by: str, tenant: str) -> dict:
        """Approve the merge: register raw_name as alias of candidate, close item."""
        rows = await self._neo4j.run(
            """
            MATCH (r:ReviewQueueItem {id: $item_id, tenant: $tenant, status: 'pending'})
            SET r.status      = 'approved',
                r.reviewed_by = $reviewed_by,
                r.reviewed_at = datetime()
            RETURN r.raw_name       AS raw_name,
                   r.raw_type       AS raw_type,
                   r.candidate_name AS candidate_name,
                   r.candidate_type AS candidate_type,
                   r.source_doc     AS source_doc
            """,
            item_id=item_id,
            tenant=tenant,
            reviewed_by=reviewed_by,
        )
        if not rows:
            return {"error": f"Item {item_id} not found or already resolved"}

        r = rows[0]
        # Register the alias so future ingestion resolves it automatically
        registry = get_alias_registry(self._neo4j, tenant=tenant)
        await registry.register_alias(
            raw_value=r["raw_name"],
            canonical_name=r["candidate_name"],
            canonical_type=r["candidate_type"],
            source_doc_id=r["source_doc"] or "",
            confidence=1.0,
        )
        log.info(
            "review_queue.approved",
            item_id=item_id,
            raw=r["raw_name"],
            canonical=r["candidate_name"],
            reviewed_by=reviewed_by,
            tenant=tenant,
        )
        return {
            "item_id": item_id,
            "status": "approved",
            "alias_registered": f"{r['raw_name']} → {r['candidate_name']}",
        }

    async def reject(self, item_id: str, reviewed_by: str, tenant: str) -> dict:
        """Reject the merge: keep entities separate, close item."""
        rows = await self._neo4j.run(
            """
            MATCH (r:ReviewQueueItem {id: $item_id, tenant: $tenant, status: 'pending'})
            SET r.status      = 'rejected',
                r.reviewed_by = $reviewed_by,
                r.reviewed_at = datetime()
            RETURN r.raw_name AS raw_name, r.candidate_name AS candidate_name
            """,
            item_id=item_id,
            tenant=tenant,
            reviewed_by=reviewed_by,
        )
        if not rows:
            return {"error": f"Item {item_id} not found or already resolved"}

        r = rows[0]
        log.info(
            "review_queue.rejected",
            item_id=item_id,
            raw=r["raw_name"],
            candidate=r["candidate_name"],
            reviewed_by=reviewed_by,
            tenant=tenant,
        )
        return {"item_id": item_id, "status": "rejected"}

    async def list_pending(self, tenant: str, limit: int = 50) -> list[dict]:
        """Return open ReviewQueueItem nodes for this tenant, newest first."""
        return await self._neo4j.run(
            """
            MATCH (r:ReviewQueueItem {tenant: $tenant, status: 'pending'})
            RETURN r.id             AS item_id,
                   r.raw_name       AS raw_name,
                   r.raw_type       AS raw_type,
                   r.candidate_name AS candidate_name,
                   r.candidate_type AS candidate_type,
                   r.score          AS score,
                   r.match_type     AS match_type,
                   r.source_doc     AS source_doc,
                   r.created_at     AS created_at
            ORDER BY r.created_at DESC
            LIMIT $limit
            """,
            tenant=tenant,
            limit=limit,
        )

    async def list_all(self, tenant: str, limit: int = 100) -> list[dict]:
        """Return all ReviewQueueItems (any status) for this tenant."""
        return await self._neo4j.run(
            """
            MATCH (r:ReviewQueueItem {tenant: $tenant})
            RETURN r.id             AS item_id,
                   r.raw_name       AS raw_name,
                   r.raw_type       AS raw_type,
                   r.candidate_name AS candidate_name,
                   r.candidate_type AS candidate_type,
                   r.score          AS score,
                   r.match_type     AS match_type,
                   r.source_doc     AS source_doc,
                   r.status         AS status,
                   r.created_at     AS created_at,
                   r.reviewed_by    AS reviewed_by,
                   r.reviewed_at    AS reviewed_at
            ORDER BY r.created_at DESC
            LIMIT $limit
            """,
            tenant=tenant,
            limit=limit,
        )
