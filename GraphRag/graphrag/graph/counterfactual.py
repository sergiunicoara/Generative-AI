"""Counterfactual / ablation queries — "what changes if we remove document X?"

Problem solved
--------------
When a document is re-processed, updated, or flagged for removal, operators
want to know the impact before committing:
  - Which entities are exclusively sourced from this document?
  - Which conflicts would be resolved if this document were gone?
  - Which inferred edges depend on edges from this document?
  - Which communities would change size or disappear?

Without counterfactual analysis, operators must either:
  (a) Remove the document and observe the fallout (risky in production), or
  (b) Guess the impact from inspection (error-prone at scale).

Architecture
------------
CounterfactualAnalyzer.simulate_retraction(doc_id, tenant, dry_run=True)
  runs a staged analysis without modifying any data:

  1. exclusive_entities — entities whose only MENTIONS source is this doc.
  2. affected_edges — RELATES_TO edges with this doc in source_doc_ids.
  3. surviving_edges — affected edges that have at least one other source doc.
  4. removed_edges — affected edges that would disappear entirely.
  5. resolved_conflicts — open Conflict nodes that would close (all sources gone).
  6. orphaned_entities — entities that would become orphans (no remaining MENTIONS).
  7. inferred_edges_at_risk — inferred edges derived from a removed edge.
  8. community_impact — which communities lose members.

Impact score (0–1) = weighted combination of:
    exclusive_entities / total_entities  * 0.3
    removed_edges / total_edges          * 0.4
    resolved_conflicts / open_conflicts  * 0.2
    orphaned_entities / total_entities   * 0.1
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)


class CounterfactualAnalyzer:
    """
    Simulate the impact of retracting a document without modifying data.

    Usage::

        analyzer = CounterfactualAnalyzer(neo4j_client)

        # Simulate removing a document
        report = await analyzer.simulate_retraction("doc_abc", tenant="acme")
        print(report["impact_score"])   # 0.0–1.0
        print(report["removed_edges"])  # list of edge descriptors

        # If the impact is acceptable, pass to GDPRService.forget_document()
        if report["impact_score"] < 0.1:
            await gdpr_svc.forget_document("doc_abc", tenant="acme")
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    async def simulate_retraction(
        self,
        doc_id: str,
        tenant: str = "default",
    ) -> dict:
        """
        Analyse what would change if doc_id were removed.

        Returns a comprehensive impact report.  No data is modified.
        """
        log.info("counterfactual.simulation_start", doc_id=doc_id, tenant=tenant)

        # ── Baseline counts ──────────────────────────────────────────────────
        baseline = await self._baseline_counts(tenant)

        # ── 1. Entities exclusive to this document ───────────────────────────
        exclusive_entities = await self._exclusive_entities(doc_id, tenant)

        # ── 2. Affected RELATES_TO edges (doc_id in source_doc_ids) ──────────
        affected_edges = await self._affected_edges(doc_id, tenant)

        # ── 3. Surviving vs. removed edges ───────────────────────────────────
        surviving = [e for e in affected_edges if e["other_source_count"] > 0]
        removed   = [e for e in affected_edges if e["other_source_count"] == 0]

        # ── 4. Conflicts that would resolve ──────────────────────────────────
        resolved_conflicts = await self._resolved_conflicts(doc_id, tenant)

        # ── 5. Entities that would become orphans ─────────────────────────────
        orphaned = await self._orphaned_entities(doc_id, tenant)

        # ── 6. Inferred edges that depend on removed edges ────────────────────
        inferred_at_risk = await self._inferred_edges_at_risk(
            [(e["src"], e["relation"], e["tgt"]) for e in removed],
            tenant,
        )

        # ── 7. Community impact ───────────────────────────────────────────────
        community_impact = await self._community_impact(exclusive_entities, tenant)

        # ── Impact score ──────────────────────────────────────────────────────
        score = self._compute_score(
            exclusive_entities=len(exclusive_entities),
            removed_edges=len(removed),
            resolved_conflicts=len(resolved_conflicts),
            orphaned_entities=len(orphaned),
            baseline=baseline,
        )

        report = {
            "doc_id":                doc_id,
            "tenant":                tenant,
            "impact_score":          round(score, 4),
            "impact_level":          ("low" if score < 0.10 else "medium" if score < 0.30 else "high"),
            "baseline":              baseline,
            "exclusive_entities":    exclusive_entities,
            "affected_edge_count":   len(affected_edges),
            "surviving_edges":       len(surviving),
            "removed_edges":         removed[:20],   # cap for readability
            "resolved_conflicts":    resolved_conflicts,
            "orphaned_entities":     orphaned,
            "inferred_edges_at_risk": inferred_at_risk[:20],
            "community_impact":      community_impact,
            "recommendation":        self._recommendation(score, resolved_conflicts),
        }
        log.info(
            "counterfactual.simulation_complete",
            doc_id=doc_id,
            impact_score=score,
            removed_edges=len(removed),
            exclusive_entities=len(exclusive_entities),
        )
        return report

    # ── Sub-analyses ───────────────────────────────────────────────────────────

    async def _baseline_counts(self, tenant: str) -> dict:
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity)
            WHERE ($tenant = 'default' OR e.tenant = $tenant)
              AND NOT e.quarantined = true
            WITH count(e) AS entity_count
            OPTIONAL MATCH ()-[r:RELATES_TO]->()
            WHERE ($tenant = 'default' OR r.tenant = $tenant)
            WITH entity_count, count(r) AS edge_count
            OPTIONAL MATCH (c:Conflict {status: 'open'})
            WHERE ($tenant = 'default' OR c.tenant = $tenant)
            RETURN entity_count, edge_count, count(c) AS open_conflicts
            """,
            tenant=tenant,
        )
        r = rows[0] if rows else {}
        return {
            "entity_count":    r.get("entity_count", 0),
            "edge_count":      r.get("edge_count", 0),
            "open_conflicts":  r.get("open_conflicts", 0),
        }

    async def _exclusive_entities(self, doc_id: str, tenant: str) -> list[dict]:
        rows = await self._neo4j.run(
            """
            MATCH (c:Chunk {document_id: $doc_id})-[:MENTIONS]->(e:Entity {tenant: $tenant})
            WHERE NOT EXISTS {
                MATCH (other:Chunk)-[:MENTIONS]->(e)
                WHERE other.document_id <> $doc_id
            }
            RETURN DISTINCT e.name AS name, e.type AS type
            """,
            doc_id=doc_id,
            tenant=tenant,
        )
        return [{"name": r["name"], "type": r["type"]} for r in rows]

    async def _affected_edges(self, doc_id: str, tenant: str) -> list[dict]:
        rows = await self._neo4j.run(
            """
            MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
            WHERE r.source_doc_ids IS NOT NULL
              AND $doc_id IN r.source_doc_ids
              AND ($tenant = 'default' OR r.tenant = $tenant)
            RETURN s.name         AS src,
                   s.type         AS src_type,
                   t.name         AS tgt,
                   t.type         AS tgt_type,
                   r.relation     AS relation,
                   r.confidence   AS confidence,
                   size([d IN r.source_doc_ids WHERE d <> $doc_id]) AS other_source_count
            """,
            doc_id=doc_id,
            tenant=tenant,
        )
        return [dict(r) for r in rows]

    async def _resolved_conflicts(self, doc_id: str, tenant: str) -> list[dict]:
        rows = await self._neo4j.run(
            """
            MATCH (c:Conflict {status: 'open'})
            WHERE ($tenant = 'default' OR c.tenant = $tenant)
              AND c.sources CONTAINS $doc_id
            RETURN c.id            AS conflict_id,
                   c.conflict_type AS conflict_type,
                   c.src           AS src,
                   c.tgt           AS tgt,
                   c.relation      AS relation
            """,
            doc_id=doc_id,
            tenant=tenant,
        )
        return [dict(r) for r in rows]

    async def _orphaned_entities(self, doc_id: str, tenant: str) -> list[dict]:
        """Entities that would have no remaining MENTIONS after this doc is removed."""
        rows = await self._neo4j.run(
            """
            MATCH (c:Chunk {document_id: $doc_id})-[:MENTIONS]->(e:Entity {tenant: $tenant})
            WHERE NOT EXISTS {
                MATCH (other:Chunk)-[:MENTIONS]->(e)
                WHERE other.document_id <> $doc_id
            }
            AND (e)-[:RELATES_TO]-()   // only flag entities with edges (pure orphans skipped)
            RETURN DISTINCT e.name AS name, e.type AS type
            """,
            doc_id=doc_id,
            tenant=tenant,
        )
        return [{"name": r["name"], "type": r["type"]} for r in rows]

    async def _inferred_edges_at_risk(
        self,
        removed_triples: list[tuple[str, str, str]],
        tenant: str,
    ) -> list[dict]:
        """Find inferred edges that were derived from (any of) the removed triples."""
        if not removed_triples:
            return []
        rows = await self._neo4j.run(
            """
            MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
            WHERE r.source_type = 'inferred'
              AND ($tenant = 'default' OR r.tenant = $tenant)
              AND r.inferred_by IS NOT NULL
            RETURN s.name AS src, t.name AS tgt, r.relation AS relation,
                   r.inferred_by AS rule, r.confidence AS confidence
            LIMIT 200
            """,
            tenant=tenant,
        )
        # Heuristic: an inferred edge is "at risk" if src or tgt is in removed triples
        removed_names = {t[0] for t in removed_triples} | {t[2] for t in removed_triples}
        at_risk = [
            dict(r) for r in rows
            if r["src"] in removed_names or r["tgt"] in removed_names
        ]
        return at_risk

    async def _community_impact(
        self,
        exclusive_entities: list[dict],
        tenant: str,
    ) -> dict:
        if not exclusive_entities:
            return {"affected_communities": 0, "entity_losses": {}}

        names = [e["name"] for e in exclusive_entities]
        rows = await self._neo4j.run(
            """
            UNWIND $names AS name
            MATCH (e:Entity {name: name, tenant: $tenant})-[:MEMBER_OF]->(c:Community {tenant: $tenant})
            RETURN c.id AS community_id, count(e) AS entities_lost, c.member_count AS total_members
            """,
            names=names,
            tenant=tenant,
        )
        affected = len(rows)
        losses = {r["community_id"]: {"lost": r["entities_lost"], "total": r["total_members"]}
                  for r in rows}
        return {"affected_communities": affected, "entity_losses": losses}

    @staticmethod
    def _compute_score(
        exclusive_entities: int,
        removed_edges: int,
        resolved_conflicts: int,
        orphaned_entities: int,
        baseline: dict,
    ) -> float:
        total_entities  = max(baseline.get("entity_count", 1), 1)
        total_edges     = max(baseline.get("edge_count", 1), 1)
        open_conflicts  = max(baseline.get("open_conflicts", 1), 1)

        score = (
            (exclusive_entities / total_entities)  * 0.30
          + (removed_edges     / total_edges)      * 0.40
          + (resolved_conflicts / open_conflicts)   * 0.20
          + (orphaned_entities  / total_entities)   * 0.10
        )
        return min(1.0, score)

    @staticmethod
    def _recommendation(score: float, resolved_conflicts: list) -> str:
        if score < 0.05:
            return "Safe to remove — minimal graph impact."
        if score < 0.15:
            n = len(resolved_conflicts)
            extra = f" {n} conflict(s) will be resolved." if n else ""
            return f"Low impact — review removed edges before proceeding.{extra}"
        if score < 0.35:
            return "Medium impact — multiple entities exclusive to this document will be lost. Consider re-extracting with a corrected source."
        return "High impact — significant graph restructuring required. Ensure replacement document is ingested first."
