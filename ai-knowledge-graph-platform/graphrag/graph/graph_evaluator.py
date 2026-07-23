"""Graph quality evaluator — semantic graph health metrics beyond RAGAS.

Problems solved
---------------
RAGAS measures answer quality (faithfulness, relevancy, precision, recall)
but says nothing about the graph itself. These blind spots accumulate silently:

1. Entity resolution quality — are there still duplicates the alias registry
   missed? What fraction of entities have known aliases vs. none?

2. Relation precision — how many RELATES_TO edges are high-confidence vs.
   noise? What is the confidence distribution shape?

3. Contradiction rate — how many open Conflict nodes exist per 1000 edges?
   A rising rate signals that new ingestions disagree with existing facts.

4. Orphan growth — are orphan entities accumulating faster than they are
   being resolved? A growing rate signals an extractor problem.

5. Merge/split error proxy — entities with suspiciously high alias counts
   may be over-merged; entities with no aliases may be under-merged.

6. Community coherence — average ratio of intra-community edges to total
   edges per entity. Low cohesion means Leiden found poor structure.

All metrics accept an optional ``tenant`` parameter so that in multi-tenant
deployments each tenant's health can be tracked independently.  Pass
``tenant="default"`` (the default) to query across all tenants.

All metrics feed into GraphHealthSnapshot nodes for trend tracking.
"""

from __future__ import annotations

from uuid import uuid4

import structlog

log = structlog.get_logger(__name__)


class GraphEvaluator:
    """
    Compute semantic graph health metrics and persist snapshots.

    Usage::

        evaluator = GraphEvaluator(neo4j_client)
        report = await evaluator.full_report(tenant="acme")
        await evaluator.persist_snapshot(report, tenant="acme")
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    # ── Individual metrics ─────────────────────────────────────────────────────

    async def entity_resolution_quality(self, tenant: str = "default") -> dict:
        """
        Entity resolution health:
        - alias_coverage: fraction of entities that have at least one alias
          (higher = better dedup tracking)
        - high_alias_count: entities with 5+ aliases (over-merge candidates)
        - no_alias_count: entities with zero aliases (under-merge risk)
        """
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity)
            WHERE coalesce(e.quarantined, false) = false
              AND ($tenant = 'default' OR e.tenant = $tenant)
            OPTIONAL MATCH (e)<-[:ALIAS_OF]-(a:Alias)
            WITH e, count(a) AS alias_count
            RETURN count(e)                                          AS total_entities,
                   count(CASE WHEN alias_count > 0  THEN 1 END)     AS with_aliases,
                   count(CASE WHEN alias_count >= 5 THEN 1 END)     AS high_alias_count,
                   count(CASE WHEN alias_count = 0  THEN 1 END)     AS no_alias_count,
                   avg(toFloat(alias_count))                         AS avg_aliases
            """,
            tenant=tenant,
        )
        r = rows[0] if rows else {}
        total = r.get("total_entities") or 1
        return {
            "total_entities":    r.get("total_entities", 0),
            "alias_coverage":    round((r.get("with_aliases", 0) / total), 4),
            "high_alias_count":  r.get("high_alias_count", 0),
            "no_alias_count":    r.get("no_alias_count", 0),
            "avg_aliases_per_entity": round(r.get("avg_aliases") or 0, 2),
        }

    async def relation_precision(self, tenant: str = "default") -> dict:
        """
        Relation edge quality:
        - high_conf_rate: fraction of edges with confidence >= 0.7
        - avg_confidence: mean confidence across all RELATES_TO edges
        - noise_edges: edges with confidence < 0.3 (likely LLM noise)
        - source_type_distribution: breakdown by source type
        - corroborated_edge_rate: fraction of edges independently asserted by 2+
          non-superseding documents (written by
          ContradictionDetector._record_corroboration). This replaces the old
          `multi_source` conflict count, which reported the same underlying
          signal as if agreement were a defect.
        """
        rows = await self._neo4j.run(
            """
            MATCH ()-[r:RELATES_TO]->()
            WHERE ($tenant = 'default' OR r.tenant = $tenant)
            RETURN count(r)                                             AS total_edges,
                   avg(r.confidence)                                    AS avg_confidence,
                   count(CASE WHEN r.confidence >= 0.7 THEN 1 END)     AS high_conf,
                   count(CASE WHEN r.confidence < 0.3  THEN 1 END)     AS noise_edges,
                   count(CASE WHEN r.source_type = 'document' THEN 1 END) AS doc_edges,
                   count(CASE WHEN r.source_type = 'llm'      THEN 1 END) AS llm_edges,
                   count(CASE WHEN r.source_type = 'inferred' THEN 1 END) AS inferred_edges,
                   count(CASE WHEN r.source_type = 'manual'   THEN 1 END) AS manual_edges,
                   count(CASE WHEN coalesce(r.independent_source_count, 0) > 1
                              THEN 1 END)                                 AS corroborated_edges
            """,
            tenant=tenant,
        )
        r = rows[0] if rows else {}
        total = r.get("total_edges") or 1
        return {
            "total_edges":    r.get("total_edges", 0),
            "avg_confidence": round(r.get("avg_confidence") or 0, 4),
            "high_conf_rate": round((r.get("high_conf", 0) / total), 4),
            "noise_edge_rate": round((r.get("noise_edges", 0) / total), 4),
            "corroborated_edge_rate": round((r.get("corroborated_edges", 0) / total), 4),
            "source_distribution": {
                "document": r.get("doc_edges", 0),
                "llm":      r.get("llm_edges", 0),
                "inferred": r.get("inferred_edges", 0),
                "manual":   r.get("manual_edges", 0),
            },
        }

    async def contradiction_rate(self, tenant: str = "default") -> dict:
        """
        Open conflicts per 1000 edges — rising rate signals new ingestions
        disagree with existing facts.
        """
        rows = await self._neo4j.run(
            """
            MATCH ()-[r:RELATES_TO]->()
            WHERE ($tenant = 'default' OR r.tenant = $tenant)
            WITH count(r) AS total_edges
            OPTIONAL MATCH (c:Conflict {status: 'open'})
            WHERE ($tenant = 'default' OR c.tenant = $tenant)
            WITH total_edges, count(c) AS open_conflicts
            OPTIONAL MATCH (cr:Conflict {status: 'resolved_manual'})
            WHERE ($tenant = 'default' OR cr.tenant = $tenant)
            RETURN total_edges, open_conflicts, count(cr) AS resolved_conflicts
            """,
            tenant=tenant,
        )
        r = rows[0] if rows else {}
        total = r.get("total_edges") or 1
        open_c = r.get("open_conflicts", 0)
        return {
            "open_conflicts":      open_c,
            "resolved_conflicts":  r.get("resolved_conflicts", 0),
            "conflicts_per_1k_edges": round(open_c / total * 1000, 2),
        }

    async def orphan_growth_rate(self, tenant: str = "default") -> dict:
        """
        Orphan entity rate — entities with no MENTIONS link.
        Compare to previous health snapshot to detect acceleration.
        """
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity)
            WHERE coalesce(e.quarantined, false) = false
              AND ($tenant = 'default' OR e.tenant = $tenant)
            WITH count(e) AS total
            MATCH (e2:Entity)
            WHERE NOT (e2)<-[:MENTIONS]-(:Chunk)
              AND coalesce(e2.quarantined, false) = false
              AND ($tenant = 'default' OR e2.tenant = $tenant)
            RETURN total, count(e2) AS orphans
            """,
            tenant=tenant,
        )
        r = rows[0] if rows else {}
        total = r.get("total") or 1
        orphans = r.get("orphans", 0)

        # Compare to last snapshot for this tenant
        prev_rows = await self._neo4j.run(
            """
            MATCH (h:GraphHealthSnapshot)
            WHERE ($tenant = 'default' OR h.tenant = $tenant)
            RETURN h.orphan_count AS prev_orphans
            ORDER BY h.recorded_at DESC
            LIMIT 1
            """,
            tenant=tenant,
        )
        prev_orphans = prev_rows[0]["prev_orphans"] if prev_rows else orphans

        return {
            "orphan_count":    orphans,
            "orphan_rate":     round(orphans / total, 4),
            "orphan_delta":    orphans - (prev_orphans or orphans),
        }

    async def merge_split_error_proxy(self, tenant: str = "default") -> dict:
        """
        Proxy for merge/split errors:
        - over_merge_candidates: entities with >=5 aliases (may be over-merged)
        - split_entities: entities with status='split' (corrected over-merges)
        - quarantined_count: currently quarantined entities
        """
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity)
            WHERE ($tenant = 'default' OR e.tenant = $tenant)
            OPTIONAL MATCH (e)<-[:ALIAS_OF]-(a:Alias)
            WITH e, count(a) AS alias_count
            RETURN count(CASE WHEN alias_count >= 5 THEN 1 END) AS over_merge_candidates,
                   count(CASE WHEN e.status = 'split' THEN 1 END) AS split_entities,
                   count(CASE WHEN e.quarantined = true THEN 1 END) AS quarantined_count
            """,
            tenant=tenant,
        )
        return dict(rows[0]) if rows else {}

    async def community_coherence(self, tenant: str = "default") -> dict:
        """
        Average intra-community edge density.
        A coherent community has most of its edges connecting members
        to each other rather than to outside entities.
        """
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity)-[:MEMBER_OF]->(c:Community)
            WHERE ($tenant = 'default' OR c.tenant = $tenant)
            WITH c, collect(e.name) AS members
            MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
            WHERE s.name IN members
              AND ($tenant = 'default' OR r.tenant = $tenant)
            WITH c, members,
                 count(CASE WHEN t.name IN members THEN 1 END) AS intra_edges,
                 count(r) AS total_edges
            WITH avg(
                CASE WHEN total_edges > 0
                     THEN toFloat(intra_edges) / total_edges
                     ELSE 0.0
                END
            ) AS avg_coherence,
            count(c) AS community_count
            RETURN avg_coherence, community_count
            """,
            tenant=tenant,
        )
        r = rows[0] if rows else {}
        return {
            "avg_community_coherence": round(r.get("avg_coherence") or 0, 4),
            "community_count":         r.get("community_count", 0),
        }

    # ── Full report ────────────────────────────────────────────────────────────

    async def full_report(self, tenant: str = "default") -> dict:
        """Run all metrics scoped to ``tenant`` and return combined report."""
        resolution    = await self.entity_resolution_quality(tenant)
        precision     = await self.relation_precision(tenant)
        contradiction = await self.contradiction_rate(tenant)
        orphans       = await self.orphan_growth_rate(tenant)
        merge_proxy   = await self.merge_split_error_proxy(tenant)
        coherence     = await self.community_coherence(tenant)

        report = {
            "tenant":               tenant,
            "entity_resolution":    resolution,
            "relation_precision":   precision,
            "contradiction":        contradiction,
            "orphan_growth":        orphans,
            "merge_split_proxy":    merge_proxy,
            "community_coherence":  coherence,
        }

        log.info("graph_evaluator.full_report", tenant=tenant, **{
            "alias_coverage":      resolution.get("alias_coverage"),
            "high_conf_rate":      precision.get("high_conf_rate"),
            "conflicts_per_1k":    contradiction.get("conflicts_per_1k_edges"),
            "orphan_rate":         orphans.get("orphan_rate"),
            "community_coherence": coherence.get("avg_community_coherence"),
        })
        return report

    async def persist_snapshot(self, report: dict, tenant: str = "default") -> str:
        """Persist a graph health snapshot and fire threshold alerts."""
        snap_id = str(uuid4())
        await self._neo4j.run(
            """
            CREATE (h:GraphHealthSnapshot {
                id:                  $id,
                tenant:              $tenant,
                alias_coverage:      $alias_coverage,
                high_conf_rate:      $high_conf_rate,
                contradiction_rate:  $contradiction_rate,
                orphan_rate:         $orphan_rate,
                orphan_count:        $orphan_count,
                community_coherence: $community_coherence,
                recorded_at:         datetime()
            })
            """,
            id=snap_id,
            tenant=tenant,
            alias_coverage=report.get("entity_resolution", {}).get("alias_coverage", 0),
            high_conf_rate=report.get("relation_precision", {}).get("high_conf_rate", 0),
            contradiction_rate=report.get("contradiction", {}).get("conflicts_per_1k_edges", 0),
            orphan_rate=report.get("orphan_growth", {}).get("orphan_rate", 0),
            orphan_count=report.get("orphan_growth", {}).get("orphan_count", 0),
            community_coherence=report.get("community_coherence", {}).get("avg_community_coherence", 0),
        )
        log.info("graph_evaluator.snapshot_persisted", snapshot_id=snap_id, tenant=tenant)

        # Fire threshold-based alerts after every snapshot so operators see
        # health degradation in their log aggregator without polling the graph.
        try:
            from graphrag.monitoring.alerts import get_alert_service
            alerts = get_alert_service().check_and_fire(report, tenant=tenant)
            if alerts:
                log.warning(
                    "graph_evaluator.alerts_fired",
                    count=len(alerts),
                    metrics=[a["metric"] for a in alerts],
                    tenant=tenant,
                )
        except Exception as exc:  # never let alerting crash the evaluator
            log.warning("graph_evaluator.alert_error", error=str(exc))

        return snap_id

    async def get_trend(self, limit: int = 10, tenant: str = "default") -> list[dict]:
        """Return recent health snapshots for trend analysis, scoped to tenant."""
        return await self._neo4j.run(
            """
            MATCH (h:GraphHealthSnapshot)
            WHERE ($tenant = 'default' OR h.tenant = $tenant)
            RETURN h.tenant              AS tenant,
                   h.alias_coverage      AS alias_coverage,
                   h.high_conf_rate      AS high_conf_rate,
                   h.contradiction_rate  AS contradiction_rate,
                   h.orphan_rate         AS orphan_rate,
                   h.community_coherence AS community_coherence,
                   h.recorded_at         AS recorded_at
            ORDER BY h.recorded_at DESC
            LIMIT $limit
            """,
            limit=limit,
            tenant=tenant,
        )
