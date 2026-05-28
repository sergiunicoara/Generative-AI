"""Dirty flags and materialized aggregates for hierarchical graph nodes.

Problems solved
---------------
1. Change propagation overload — when a leaf node status changes, every
   ancestor must know, but recomputing the full tree on every change is
   O(N) writes.  Dirty flags defer recomputation to query time.

2. Structural recomputation cost — "how complete is the avionics system?"
   requires traversing potentially thousands of nodes.  Materialized
   aggregates cache the answer on each parent and recompute lazily.

Architecture
------------
On write:
    leaf node status changes → mark all ancestors `status_dirty = true`

On read:
    if ancestor.status_dirty: recompute subtree → clear flag
    else: return cached aggregate (instant)

Aggregates stored per node:
    missing_count       int    number of missing descendants
    blocked_count       int    number of blocked descendants
    ready_percentage    float  % of subtree that is available
    computed_status     str    worst status across subtree
    last_computed       datetime
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)

MAX_PROPAGATION_DEPTH = 10   # safety limit for ancestor traversal


class PropagationService:
    """
    Manages dirty flag propagation and lazy aggregate recomputation.

    Usage::

        svc = PropagationService(neo4j_client)
        await svc.mark_dirty("entity_or_block_name")
        status = await svc.get_computed_status("Avionics System")
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    # ── Dirty flag propagation ─────────────────────────────────────────────────

    async def mark_dirty(self, entity_name: str) -> int:
        """
        Mark all ancestors of entity_name as dirty.
        Returns the number of nodes flagged.
        """
        rows = await self._neo4j.run(
            f"""
            MATCH path = (ancestor:Entity)-[:RELATES_TO*1..{MAX_PROPAGATION_DEPTH}]
                         ->(changed:Entity {{name: $name}})
            WITH DISTINCT ancestor
            SET ancestor.status_dirty = true
            RETURN count(ancestor) AS flagged
            """,
            name=entity_name,
        )
        flagged = rows[0]["flagged"] if rows else 0
        log.info("propagation.dirty_flagged", entity=entity_name, ancestors=flagged)
        return flagged

    async def clear_dirty(self, entity_name: str) -> None:
        """Clear dirty flag after recomputation."""
        await self._neo4j.run(
            """
            MATCH (e:Entity {name: $name})
            SET e.status_dirty = false,
                e.last_computed = datetime()
            """,
            name=entity_name,
        )

    # ── Aggregate recomputation ────────────────────────────────────────────────

    async def recompute_aggregates(self, entity_name: str) -> dict:
        """
        Recompute materialized aggregates for entity_name based on
        the current status of all its descendants.

        Returns the computed aggregate dict.
        """
        rows = await self._neo4j.run(
            """
            MATCH (root:Entity {name: $name})
            OPTIONAL MATCH (root)-[:RELATES_TO*1..10]->(desc:Entity)
            WITH root,
                 count(desc)                                                    AS total,
                 count(CASE WHEN desc.own_status = 'missing'   THEN 1 END)     AS missing,
                 count(CASE WHEN desc.own_status = 'blocked'   THEN 1 END)     AS blocked,
                 count(CASE WHEN desc.own_status = 'available' THEN 1 END)     AS available
            SET root.missing_count    = missing,
                root.blocked_count    = blocked,
                root.ready_percentage = CASE WHEN total > 0
                                             THEN toFloat(available) / total * 100
                                             ELSE 100.0
                                        END,
                root.computed_status  = CASE
                                            WHEN missing > 0 THEN 'incomplete'
                                            WHEN blocked > 0 THEN 'blocked'
                                            ELSE 'ready'
                                        END,
                root.status_dirty     = false,
                root.last_computed    = datetime()
            RETURN root.missing_count    AS missing_count,
                   root.blocked_count    AS blocked_count,
                   root.ready_percentage AS ready_percentage,
                   root.computed_status  AS computed_status
            """,
            name=entity_name,
        )
        result = dict(rows[0]) if rows else {}
        log.info("propagation.recomputed", entity=entity_name, **result)
        return result

    async def get_computed_status(self, entity_name: str) -> dict:
        """
        Return the aggregate status for entity_name.
        Recomputes lazily if dirty flag is set.
        """
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity {name: $name})
            RETURN e.status_dirty     AS dirty,
                   e.computed_status  AS computed_status,
                   e.missing_count    AS missing_count,
                   e.blocked_count    AS blocked_count,
                   e.ready_percentage AS ready_percentage,
                   e.last_computed    AS last_computed
            """,
            name=entity_name,
        )
        if not rows:
            return {"computed_status": "unknown"}

        row = rows[0]
        if row.get("dirty") or row.get("computed_status") is None:
            return await self.recompute_aggregates(entity_name)

        return {
            "computed_status":  row.get("computed_status"),
            "missing_count":    row.get("missing_count", 0),
            "blocked_count":    row.get("blocked_count", 0),
            "ready_percentage": row.get("ready_percentage", 100.0),
            "last_computed":    row.get("last_computed"),
        }

    async def batch_recompute_dirty(self, limit: int = 100) -> int:
        """
        Recompute all currently dirty nodes in a batch.
        Returns number of nodes recomputed.
        Intended for background jobs or post-ingestion cleanup.
        """
        dirty_rows = await self._neo4j.run(
            """
            MATCH (e:Entity)
            WHERE e.status_dirty = true
            RETURN e.name AS name
            LIMIT $limit
            """,
            limit=limit,
        )
        count = 0
        for row in dirty_rows:
            await self.recompute_aggregates(row["name"])
            count += 1
        log.info("propagation.batch_recomputed", count=count)
        return count

    async def propagate_status_change(
        self, entity_name: str, new_status: str
    ) -> None:
        """
        Full workflow: update own_status → mark ancestors dirty → stop
        propagation early if parent computed_status doesn't change.
        """
        # Update the leaf
        await self._neo4j.run(
            """
            MATCH (e:Entity {name: $name})
            SET e.own_status = $status, e.status_updated_at = datetime()
            """,
            name=entity_name,
            status=new_status,
        )

        # Get direct parents
        parents = await self._neo4j.run(
            """
            MATCH (parent:Entity)-[:RELATES_TO]->(e:Entity {name: $name})
            RETURN parent.name AS name, parent.computed_status AS computed_status
            """,
            name=entity_name,
        )

        for parent in parents:
            old_status = parent.get("computed_status")
            new_agg = await self.recompute_aggregates(parent["name"])
            # Only propagate upward if the parent's own computed status changed
            if new_agg.get("computed_status") != old_status:
                await self.mark_dirty(parent["name"])

        log.info(
            "propagation.status_change_complete",
            entity=entity_name,
            new_status=new_status,
            parents_checked=len(parents),
        )
