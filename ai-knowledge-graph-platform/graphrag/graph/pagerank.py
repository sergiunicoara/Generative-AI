"""Compute PageRank centrality over the tenant's entity graph via Neo4j GDS.

Recompute triggers
-------------------
PageRank was previously never recomputed automatically — only via a manual
script or admin API call. A live check found real, uneven drift: one tenant
at 45% entity coverage (stale for weeks), another at 100% coverage but 11
days stale, two tenants never computed at all. `check_staleness()` below
adds three triggers, mirroring the proven `CommunityManager` staleness
pattern (`graphrag/graph/community_manager.py`) rather than inventing a new
one:

1. Growth drift — entity/edge counts changed enough since the last
   `PageRankSnapshot` (same relative-drift formula as community staleness).
2. Re-ingestion of an existing document — verified this is NOT subsumed by
   growth drift: `merge_relation` sets `r.extracted_at` and boosts
   `r.confidence` via the Bayesian noisy-OR merge unconditionally on every
   match, not just on create. Since `run_pagerank` weights edges by
   `coalesce(r.weight, r.confidence, 1.0)`, re-ingesting a document changes
   PageRank's actual inputs with zero change to entity/edge count — a
   growth check alone would miss this entirely.
3. A decay-conditional time ceiling — `gnn_confidence_half_life_days`
   (retrieval config) decays edge confidence purely with elapsed time, so a
   tenant can drift with zero ingestion at all. But if decay is disabled
   (half-life 0) for a tenant, nothing drifts between ingestions, so this
   ceiling only evaluates when decay is actually enabled — otherwise it
   would recompute an unchanged result on a timer for no reason.
"""

from __future__ import annotations

from uuid import uuid4

import structlog

from graphrag.core.config import get_settings, resolve_tenant_config
from graphrag.graph.corpus_revision import CorpusMutation
from graphrag.graph.neo4j_client import get_neo4j
from graphrag.graph.staleness import staleness_score as compute_staleness

log = structlog.get_logger(__name__)

DEFAULT_GROWTH_THRESHOLD = 0.15
DEFAULT_TIME_CEILING_DAYS = 60


class PageRankComputer:
    """Runs GDS PageRank for one tenant and persists scores onto Entity nodes."""

    def __init__(self, tenant: str = "default"):
        self._cfg = get_settings().graph
        self._retrieval_cfg = get_settings().retrieval
        self._neo4j = get_neo4j()
        self._tenant = tenant

    async def compute_and_persist(self, *, publish_revision: bool = True) -> dict:
        """Compute PageRank with cache reads disabled during persisted writes."""
        if not publish_revision:
            return await self._compute_and_persist()
        async with CorpusMutation(self._neo4j, self._tenant, "pagerank_recompute"):
            return await self._compute_and_persist()

    async def _compute_and_persist(self) -> dict:
        log.info("pagerank.start", tenant=self._tenant)

        scores = await self._neo4j.run_pagerank(
            tenant=self._tenant,
            damping_factor=self._cfg.get("pagerank_damping_factor", 0.85),
            max_iterations=self._cfg.get("pagerank_iterations", 20),
        )

        if not scores:
            log.warning("pagerank.no_entities", tenant=self._tenant)
            return {"tenant": self._tenant, "entities_scored": 0, "top_entity": None}

        await self._neo4j.write_pagerank_scores(self._tenant, scores)
        await self.snapshot()
        log.info(
            "pagerank.done",
            tenant=self._tenant,
            entities_scored=len(scores),
            top_entity=scores[0]["name"],
            top_score=round(scores[0]["score"], 4),
        )
        return {
            "tenant": self._tenant,
            "entities_scored": len(scores),
            "top_entity": scores[0]["name"],
            "top_score": round(scores[0]["score"], 4),
        }

    # ── Snapshot ───────────────────────────────────────────────────────────────

    async def snapshot(self) -> dict:
        """Record current graph state as a PageRankSnapshot.

        Called at the end of every real compute_and_persist() run, so each
        recompute establishes the next staleness baseline — same idiom as
        CommunityManager.mark_rebuilt() calling snapshot().
        """
        stats_rows = await self._neo4j.run(
            """
            MATCH (e:Entity {tenant: $tenant})
            WHERE coalesce(e.quarantined, false) = false
            WITH count(e) AS entity_count
            MATCH ()-[r:RELATES_TO {tenant: $tenant}]->()
            RETURN entity_count, count(r) AS edge_count
            """,
            tenant=self._tenant,
        )
        if not stats_rows:
            return {}

        stats = stats_rows[0]
        snap_id = str(uuid4())

        await self._neo4j.run(
            """
            CREATE (s:PageRankSnapshot {
                id:           $id,
                entity_count: $entity_count,
                edge_count:   $edge_count,
                tenant:       $tenant,
                recorded_at:  datetime()
            })
            """,
            id=snap_id,
            entity_count=stats["entity_count"],
            edge_count=stats["edge_count"],
            tenant=self._tenant,
        )

        log.info(
            "pagerank.snapshot",
            snapshot_id=snap_id,
            entities=stats["entity_count"],
            edges=stats["edge_count"],
        )
        return {
            "snapshot_id": snap_id,
            "entity_count": stats["entity_count"],
            "edge_count": stats["edge_count"],
        }

    # ── Staleness ──────────────────────────────────────────────────────────────

    async def check_staleness(self, is_reingest: bool = False) -> dict:
        """
        Decide whether this tenant's PageRank should be recomputed.

        Returns:
            should_recompute: bool
            reason: "reingest" | "growth_drift" | "decay_time_ceiling" |
                    "never_computed" | "up_to_date"
            staleness_score / delta: present for the growth-drift path
        """
        if is_reingest:
            result = {"should_recompute": True, "reason": "reingest"}
            log.info("pagerank.staleness_check", tenant=self._tenant, **result)
            return result

        snap_rows = await self._neo4j.run(
            """
            MATCH (s:PageRankSnapshot {tenant: $tenant})
            RETURN s.entity_count AS entity_count,
                   s.edge_count   AS edge_count,
                   s.recorded_at  AS recorded_at
            ORDER BY s.recorded_at DESC
            LIMIT 1
            """,
            tenant=self._tenant,
        )
        if not snap_rows:
            result = {"should_recompute": True, "reason": "never_computed"}
            log.info("pagerank.staleness_check", tenant=self._tenant, **result)
            return result

        snap = snap_rows[0]

        curr_rows = await self._neo4j.run(
            """
            MATCH (e:Entity {tenant: $tenant})
            WHERE coalesce(e.quarantined, false) = false
            WITH count(e) AS entities
            MATCH ()-[r:RELATES_TO {tenant: $tenant}]->()
            RETURN entities, count(r) AS edges
            """,
            tenant=self._tenant,
        )
        curr = curr_rows[0] if curr_rows else {"entities": 0, "edges": 0}

        staleness_score, entity_drift, edge_drift = compute_staleness(
            snap["entity_count"], snap["edge_count"],
            curr["entities"], curr["edges"],
        )
        threshold = self._cfg.get("pagerank_growth_threshold", DEFAULT_GROWTH_THRESHOLD)

        if staleness_score > threshold:
            result = {
                "should_recompute": True,
                "reason": "growth_drift",
                "staleness_score": staleness_score,
                "delta": {
                    "entity_drift_pct": round(entity_drift * 100, 1),
                    "edge_drift_pct":   round(edge_drift * 100, 1),
                },
            }
            log.info("pagerank.staleness_check", tenant=self._tenant, **result)
            return result

        # Decay-conditional time ceiling: only meaningful when confidence
        # actually decays over time for this tenant — otherwise a tenant with
        # decay disabled would recompute an unchanged result on a timer.
        tenant_retrieval_cfg = resolve_tenant_config(self._retrieval_cfg, self._tenant)
        half_life = tenant_retrieval_cfg.get("gnn_confidence_half_life_days", 0)
        if half_life > 0:
            recorded_at = snap.get("recorded_at")
            if recorded_at is not None:
                age_days = (_now() - _as_datetime(recorded_at)).days
                ceiling = self._cfg.get("pagerank_time_ceiling_days", DEFAULT_TIME_CEILING_DAYS)
                if age_days > ceiling:
                    result = {
                        "should_recompute": True,
                        "reason": "decay_time_ceiling",
                        "age_days": age_days,
                    }
                    log.info("pagerank.staleness_check", tenant=self._tenant, **result)
                    return result

        result = {
            "should_recompute": False,
            "reason": "up_to_date",
            "staleness_score": staleness_score,
        }
        log.info("pagerank.staleness_check", tenant=self._tenant, **result)
        return result


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


def _as_datetime(value):
    """Normalise a Neo4j datetime value to a timezone-aware Python datetime
    for age-in-days arithmetic. Mirrors the exact handling in
    incremental_community.py (string/to_native/native cases) plus the
    naive-datetime guard from gnn_scorer.py's confidence-decay code — both
    to_native() and a bare fromisoformat() parse can return a naive
    datetime, which would crash a subtraction against the tz-aware `now`
    from _now()."""
    from datetime import datetime, timezone
    if isinstance(value, str):
        dt = datetime.fromisoformat(value)
    elif hasattr(value, "to_native"):
        dt = value.to_native()
    else:
        dt = value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
