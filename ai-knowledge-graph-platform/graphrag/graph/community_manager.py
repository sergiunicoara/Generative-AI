"""Community manager — versioning and staleness detection for Leiden communities.

Problems solved
---------------
1. Community quality instability — Leiden communities change materially as
   the graph evolves. Summaries become stale, retrieval behavior drifts,
   and there is no visibility into how much the structure has shifted.

2. No rebuild trigger — communities are rebuilt only by manual script runs.
   Without staleness metrics, operators don't know when a rebuild is overdue.

3. No version history — there is no way to compare retrieval quality before
   and after a community rebuild.

Architecture
------------
- CommunitySnapshot nodes record entity count, edge count, and community
  count at a point in time.
- Staleness = relative change in entity/edge counts since last snapshot.
- should_rebuild() returns True when staleness exceeds a threshold.
- mark_rebuilt() records the new version after a successful rebuild.
"""

from __future__ import annotations

from uuid import uuid4

import structlog

from graphrag.graph.staleness import staleness_score as compute_staleness

log = structlog.get_logger(__name__)

DEFAULT_STALENESS_THRESHOLD = 0.15   # 15% graph change triggers rebuild


class CommunityManager:
    """
    Tracks community version snapshots and staleness.

    Usage::

        mgr = CommunityManager(neo4j_client)
        await mgr.snapshot()                          # record current state
        stale = await mgr.check_staleness()
        if stale["should_rebuild"]:
            # trigger community_builder.build() + community_summarizer.summarize_all()
            await mgr.mark_rebuilt()
    """

    def __init__(
        self,
        neo4j_client,
        staleness_threshold: float = DEFAULT_STALENESS_THRESHOLD,
    ):
        self._neo4j = neo4j_client
        self._threshold = staleness_threshold

    # ── Snapshot ───────────────────────────────────────────────────────────────

    async def snapshot(self, tenant: str = "default") -> dict:
        """
        Record current graph state as a CommunitySnapshot.
        Returns the snapshot dict.
        """
        stats_rows = await self._neo4j.run(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE (c.tenant = $tenant)
              AND coalesce(e.quarantined, false) = false
            WITH count(DISTINCT e) AS entity_count
            MATCH ()-[r:RELATES_TO {tenant: $tenant}]->()
            WITH entity_count, count(r) AS edge_count
            MATCH (c:Community)
            WHERE c.tenant = $tenant
            RETURN entity_count, edge_count, count(c) AS community_count
            """,
            tenant=tenant,
        )
        if not stats_rows:
            return {}

        stats = stats_rows[0]
        snap_id = str(uuid4())

        await self._neo4j.run(
            """
            CREATE (s:CommunitySnapshot {
                id:              $id,
                entity_count:    $entity_count,
                edge_count:      $edge_count,
                community_count: $community_count,
                tenant:          $tenant,
                recorded_at:     datetime()
            })
            """,
            id=snap_id,
            entity_count=stats["entity_count"],
            edge_count=stats["edge_count"],
            community_count=stats["community_count"],
            tenant=tenant,
        )

        log.info(
            "community_manager.snapshot",
            snapshot_id=snap_id,
            entities=stats["entity_count"],
            edges=stats["edge_count"],
            communities=stats["community_count"],
        )
        return {
            "snapshot_id": snap_id,
            "entity_count": stats["entity_count"],
            "edge_count": stats["edge_count"],
            "community_count": stats["community_count"],
        }

    # ── Staleness ──────────────────────────────────────────────────────────────

    async def check_staleness(self, tenant: str = "default") -> dict:
        """
        Compare current graph state against the most recent snapshot.

        Returns:
            staleness_score: 0.0 (no change) to 1.0 (completely different)
            should_rebuild: True if staleness_score > threshold
            delta: dict with absolute changes per metric
        """
        # Most recent snapshot
        snap_rows = await self._neo4j.run(
            """
            MATCH (s:CommunitySnapshot)
            WHERE s.tenant = $tenant
            RETURN s.entity_count    AS entity_count,
                   s.edge_count      AS edge_count,
                   s.community_count AS community_count,
                   s.recorded_at     AS recorded_at
            ORDER BY s.recorded_at DESC
            LIMIT 1
            """,
            tenant=tenant,
        )
        if not snap_rows:
            return {"staleness_score": 1.0, "should_rebuild": True, "delta": {}}

        snap = snap_rows[0]

        # Current state
        curr_rows = await self._neo4j.run(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE (c.tenant = $tenant)
              AND coalesce(e.quarantined, false) = false
            WITH count(DISTINCT e) AS entities
            MATCH ()-[r:RELATES_TO {tenant: $tenant}]->()
            RETURN entities, count(r) AS edges
            """,
            tenant=tenant,
        )
        if not curr_rows:
            return {"staleness_score": 0.0, "should_rebuild": False, "delta": {}}

        curr = curr_rows[0]

        staleness_score, entity_drift, edge_drift = compute_staleness(
            snap["entity_count"], snap["edge_count"],
            curr["entities"], curr["edges"],
        )
        should_rebuild = staleness_score > self._threshold

        result = {
            "staleness_score": staleness_score,
            "should_rebuild": should_rebuild,
            "threshold": self._threshold,
            "delta": {
                "entity_drift_pct": round(entity_drift * 100, 1),
                "edge_drift_pct":   round(edge_drift * 100, 1),
            },
            "snapshot_recorded_at": snap.get("recorded_at"),
        }

        log.info("community_manager.staleness_check", **result)
        return result

    async def mark_rebuilt(self, tenant: str = "default") -> str:
        """
        Record that communities were just rebuilt.
        Takes a new snapshot and links it as the current version.
        Returns the new snapshot ID.
        """
        snap = await self.snapshot(tenant=tenant)
        snap_id = snap.get("snapshot_id", "")

        # Mark this snapshot as a rebuild milestone. snap_id is already
        # tenant-derived (from self.snapshot(tenant=tenant) above), but the
        # match itself carried no tenant filter -- adding one for the same
        # defense-in-depth reason as every other id-only match fixed in this
        # pass, not because a collision is likely here.
        await self._neo4j.run(
            """
            MATCH (s:CommunitySnapshot {id: $id, tenant: $tenant})
            SET s.is_rebuild_milestone = true
            """,
            id=snap_id,
            tenant=tenant,
        )

        log.info("community_manager.rebuild_recorded", snapshot_id=snap_id)
        return snap_id

    async def get_version_history(
        self,
        limit: int = 10,
        tenant: str = "default",
    ) -> list[dict]:
        """Return the most recent community snapshots."""
        return await self._neo4j.run(
            """
            MATCH (s:CommunitySnapshot)
            WHERE s.tenant = $tenant
            RETURN s.id              AS snapshot_id,
                   s.entity_count   AS entity_count,
                   s.edge_count     AS edge_count,
                   s.community_count AS community_count,
                   s.recorded_at    AS recorded_at,
                   coalesce(s.is_rebuild_milestone, false) AS is_rebuild
            ORDER BY s.recorded_at DESC
            LIMIT $limit
            """,
            limit=limit,
            tenant=tenant,
        )
