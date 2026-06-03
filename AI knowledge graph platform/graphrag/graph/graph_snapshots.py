"""Graph snapshots and time travel — point-in-time KG state queries.

Problems solved
---------------
1. No audit baseline — when a report shows "graph quality degraded", there is
   no stored reference point to compare against.  Snapshots create named
   checkpoints that survive indefinitely.

2. Deployment rollback — if a bad ingestion batch corrupts the graph, there
   is no mechanism to identify exactly what changed.  Snapshot diffs surface
   the delta between any two checkpoints.

3. Quarterly/release-gate reporting — operators need to attest that the graph
   state on a specific date met quality thresholds.  Named snapshots with
   health metrics satisfy this requirement.

Architecture
------------
- GraphSnapshot nodes store summary statistics (counts, health metrics) as
  a lightweight checkpoint — not a full copy of the graph.
- create_snapshot()    — capture current state with a named label.
- diff_snapshots()     — compute entity/edge delta between two checkpoints.
- list_snapshots()     — enumerate checkpoints for a tenant.
- restore_summary()    — return the stored metrics from a past snapshot.
- time_travel_query()  — delegate to BitemporalStore for point-in-time queries
                         using stored recorded_at transaction timestamps.

Full graph versioning (storing complete node/edge copies per snapshot) is
deliberately out of scope — it would double storage cost.  Instead, snapshots
record statistics plus the recorded_at timestamp of the youngest fact at
snapshot time, which lets BitemporalStore reconstruct the state precisely
(provided the underlying data has recorded_at set, which merge_entity and
merge_relation now both do).

GraphSnapshot node properties
------------------------------
  id                UUID
  label             str         human-readable name ("post-v2-ingest", "Q1-2025")
  tenant            str
  entity_count      int
  edge_count        int
  negative_count    int         NEGATIVE_RELATES_TO edges
  conflict_count    int         open Conflict nodes
  community_count   int
  orphan_count      int
  avg_confidence    float
  recorded_at       datetime    transaction-time upper bound for time travel
  created_at        datetime    when the snapshot was taken
"""

from __future__ import annotations

from uuid import uuid4

import structlog

log = structlog.get_logger(__name__)


class GraphSnapshotService:
    """
    Create, diff, and list lightweight graph snapshots.

    Usage::

        svc = GraphSnapshotService(neo4j_client)

        # Create a named checkpoint
        snap_id = await svc.create_snapshot(label="post-Q1-ingest", tenant="acme")

        # List all checkpoints
        snaps = await svc.list_snapshots(tenant="acme")

        # Compare two checkpoints
        diff = await svc.diff_snapshots(snap_id_a=snaps[0]["id"],
                                         snap_id_b=snaps[1]["id"],
                                         tenant="acme")

        # Retrieve stored metrics from a past snapshot
        metrics = await svc.restore_summary(snap_id, tenant="acme")

        # Full bitemporal time-travel: reconstruct graph as of snapshot time
        from graphrag.graph.bitemporal import BitemporalStore
        bt = BitemporalStore(neo4j_client)
        state = await bt.time_travel_report(
            valid_time=metrics["recorded_at"],
            transaction_time=metrics["recorded_at"],
            tenant="acme",
        )
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    # ── Create ─────────────────────────────────────────────────────────────────

    async def create_snapshot(
        self,
        label: str,
        tenant: str = "default",
        include_health: bool = True,
    ) -> str:
        """
        Capture current graph statistics and store as a named snapshot.

        Parameters
        ----------
        label          : human-readable name (e.g. "pre-reindex", "Q1-2025")
        tenant         : tenant scope
        include_health : if True, also run GraphEvaluator for health metrics

        Returns the snapshot node ID.
        """
        stats = await self._collect_stats(tenant)

        health_data: dict = {}
        if include_health:
            try:
                from graphrag.graph.graph_evaluator import GraphEvaluator
                evaluator = GraphEvaluator(self._neo4j)
                report = await evaluator.full_report(tenant=tenant)
                health_data = {
                    "alias_coverage":      report.get("entity_resolution", {}).get("alias_coverage", 0.0),
                    "high_conf_rate":      report.get("relation_precision", {}).get("high_conf_rate", 0.0),
                    "contradiction_rate":  report.get("contradiction", {}).get("conflicts_per_1k_edges", 0.0),
                    "orphan_rate":         report.get("orphan_growth", {}).get("orphan_rate", 0.0),
                    "community_coherence": report.get("community_coherence", {}).get("avg_community_coherence", 0.0),
                }
            except (KeyError, TypeError, AttributeError) as exc:
                log.warning("graph_snapshots.health_skipped", error=str(exc))

        snap_id = str(uuid4())
        await self._neo4j.run(
            """
            CREATE (s:GraphSnapshot {
                id:               $id,
                label:            $label,
                tenant:           $tenant,
                entity_count:     $entity_count,
                edge_count:       $edge_count,
                negative_count:   $negative_count,
                conflict_count:   $conflict_count,
                community_count:  $community_count,
                orphan_count:     $orphan_count,
                avg_confidence:   $avg_confidence,
                alias_coverage:   $alias_coverage,
                high_conf_rate:   $high_conf_rate,
                contradiction_rate: $contradiction_rate,
                orphan_rate:      $orphan_rate,
                community_coherence: $community_coherence,
                recorded_at:      datetime(),
                created_at:       datetime()
            })
            """,
            id=snap_id,
            label=label,
            tenant=tenant,
            entity_count=stats["entity_count"],
            edge_count=stats["edge_count"],
            negative_count=stats["negative_count"],
            conflict_count=stats["conflict_count"],
            community_count=stats["community_count"],
            orphan_count=stats["orphan_count"],
            avg_confidence=stats["avg_confidence"],
            alias_coverage=health_data.get("alias_coverage", 0.0),
            high_conf_rate=health_data.get("high_conf_rate", 0.0),
            contradiction_rate=health_data.get("contradiction_rate", 0.0),
            orphan_rate=health_data.get("orphan_rate", 0.0),
            community_coherence=health_data.get("community_coherence", 0.0),
        )

        log.info(
            "graph_snapshots.created",
            snap_id=snap_id,
            label=label,
            tenant=tenant,
            entity_count=stats["entity_count"],
            edge_count=stats["edge_count"],
        )
        return snap_id

    # ── List / retrieve ────────────────────────────────────────────────────────

    async def list_snapshots(
        self,
        tenant: str = "default",
        limit: int = 50,
    ) -> list[dict]:
        """Return snapshot metadata ordered by creation time, newest first."""
        return await self._neo4j.run(
            """
            MATCH (s:GraphSnapshot)
            WHERE ($tenant = 'default' OR s.tenant = $tenant)
            RETURN s.id              AS id,
                   s.label           AS label,
                   s.tenant          AS tenant,
                   s.entity_count    AS entity_count,
                   s.edge_count      AS edge_count,
                   s.conflict_count  AS conflict_count,
                   s.orphan_count    AS orphan_count,
                   s.avg_confidence  AS avg_confidence,
                   s.alias_coverage  AS alias_coverage,
                   s.high_conf_rate  AS high_conf_rate,
                   s.contradiction_rate AS contradiction_rate,
                   s.orphan_rate     AS orphan_rate,
                   s.community_coherence AS community_coherence,
                   s.recorded_at     AS recorded_at,
                   s.created_at      AS created_at
            ORDER BY s.created_at DESC
            LIMIT $limit
            """,
            tenant=tenant,
            limit=limit,
        )

    async def restore_summary(
        self,
        snap_id: str,
        tenant: str = "default",
    ) -> dict | None:
        """
        Return the stored statistics from a specific snapshot.

        This does NOT restore the graph — use BitemporalStore.as_of_* for
        that.  restore_summary() returns the metrics captured at snapshot time.
        """
        rows = await self._neo4j.run(
            """
            MATCH (s:GraphSnapshot {id: $snap_id})
            WHERE ($tenant = 'default' OR s.tenant = $tenant)
            RETURN s {.*} AS props
            """,
            snap_id=snap_id,
            tenant=tenant,
        )
        if not rows:
            return None
        return dict(rows[0].get("props") or {})

    # ── Diff ───────────────────────────────────────────────────────────────────

    async def diff_snapshots(
        self,
        snap_id_a: str,
        snap_id_b: str,
        tenant: str = "default",
    ) -> dict:
        """
        Compute the statistical delta between two snapshots.

        snap_id_a is treated as "before"; snap_id_b as "after".
        Returns a diff dict with absolute changes and percentage changes.

        For entity/edge identity diff (which specific entities were added or
        removed) use BitemporalStore.transaction_diff() with the recorded_at
        timestamps from each snapshot.
        """
        rows = await self._neo4j.run(
            """
            MATCH (a:GraphSnapshot {id: $id_a}), (b:GraphSnapshot {id: $id_b})
            WHERE ($tenant = 'default' OR (a.tenant = $tenant AND b.tenant = $tenant))
            RETURN a.entity_count      AS a_entities,   b.entity_count      AS b_entities,
                   a.edge_count        AS a_edges,       b.edge_count        AS b_edges,
                   a.conflict_count    AS a_conflicts,   b.conflict_count    AS b_conflicts,
                   a.orphan_count      AS a_orphans,     b.orphan_count      AS b_orphans,
                   a.avg_confidence    AS a_conf,        b.avg_confidence    AS b_conf,
                   a.alias_coverage    AS a_alias,       b.alias_coverage    AS b_alias,
                   a.contradiction_rate AS a_crate,      b.contradiction_rate AS b_crate,
                   a.community_coherence AS a_coh,       b.community_coherence AS b_coh,
                   a.recorded_at       AS a_recorded,    b.recorded_at       AS b_recorded,
                   a.label             AS a_label,       b.label             AS b_label
            """,
            id_a=snap_id_a,
            id_b=snap_id_b,
            tenant=tenant,
        )
        if not rows:
            return {"error": "One or both snapshot IDs not found"}

        r = rows[0]

        def delta(a, b):
            a, b = float(a or 0), float(b or 0)
            return {"before": a, "after": b, "delta": round(b - a, 4),
                    "pct_change": round((b - a) / a * 100, 1) if a else 0.0}

        return {
            "snap_a":        {"id": snap_id_a, "label": r["a_label"], "recorded_at": r["a_recorded"]},
            "snap_b":        {"id": snap_id_b, "label": r["b_label"], "recorded_at": r["b_recorded"]},
            "tenant":        tenant,
            "entities":      delta(r["a_entities"],  r["b_entities"]),
            "edges":         delta(r["a_edges"],     r["b_edges"]),
            "conflicts":     delta(r["a_conflicts"], r["b_conflicts"]),
            "orphans":       delta(r["a_orphans"],   r["b_orphans"]),
            "avg_confidence":       delta(r["a_conf"],   r["b_conf"]),
            "alias_coverage":       delta(r["a_alias"],  r["b_alias"]),
            "contradiction_rate":   delta(r["a_crate"],  r["b_crate"]),
            "community_coherence":  delta(r["a_coh"],    r["b_coh"]),
        }

    # ── Private ────────────────────────────────────────────────────────────────

    async def _collect_stats(self, tenant: str) -> dict:
        """Query live graph for current statistics."""
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity)
            WHERE ($tenant = 'default' OR e.tenant = $tenant)
              AND coalesce(e.quarantined, false) = false
            WITH count(e) AS entity_count
            OPTIONAL MATCH ()-[r:RELATES_TO]->()
            WHERE ($tenant = 'default' OR r.tenant = $tenant)
            WITH entity_count, count(r) AS edge_count, avg(r.confidence) AS avg_conf
            OPTIONAL MATCH ()-[nr:NEGATIVE_RELATES_TO]->()
            WHERE ($tenant = 'default' OR nr.tenant = $tenant)
            WITH entity_count, edge_count, avg_conf, count(nr) AS neg_count
            OPTIONAL MATCH (c:Conflict {status: 'open'})
            WHERE ($tenant = 'default' OR c.tenant = $tenant)
            WITH entity_count, edge_count, avg_conf, neg_count, count(c) AS conflict_count
            OPTIONAL MATCH (cm:Community)
            WHERE ($tenant = 'default' OR cm.tenant = $tenant)
            WITH entity_count, edge_count, avg_conf, neg_count, conflict_count,
                 count(cm) AS community_count
            OPTIONAL MATCH (orphan:Entity)
            WHERE ($tenant = 'default' OR orphan.tenant = $tenant)
              AND coalesce(orphan.quarantined, false) = false
              AND NOT (orphan)<-[:MENTIONS]-(:Chunk)
            RETURN entity_count, edge_count, avg_conf, neg_count,
                   conflict_count, community_count, count(orphan) AS orphan_count
            """,
            tenant=tenant,
        )
        r = rows[0] if rows else {}
        return {
            "entity_count":    r.get("entity_count", 0),
            "edge_count":      r.get("edge_count", 0),
            "negative_count":  r.get("neg_count", 0),
            "conflict_count":  r.get("conflict_count", 0),
            "community_count": r.get("community_count", 0),
            "orphan_count":    r.get("orphan_count", 0),
            "avg_confidence":  round(float(r.get("avg_conf") or 0.0), 4),
        }
