"""Contradiction detector — first-class conflict modeling for incompatible facts.

Problems solved
---------------
1. Silent contradiction accumulation — two documents assert incompatible facts
   (e.g. "Engine A uses Fuel Pump B" vs "Engine A does NOT use Fuel Pump B").
   The Bayesian confidence merge accumulates both without surfacing the conflict.

2. Directional reversals — (A)-[CEO_OF]->(B) and (B)-[CEO_OF]->(A) can
   both exist if two documents disagree on the direction.

3. No resolution tracking — conflicts found by document_authority are ranked
   and silently resolved by authority. There is no audit that a conflict existed,
   was detected, or how it was resolved.

4. Mutually exclusive states — an entity carries contradictory status labels
   from different source documents (e.g. "active" and "deprecated" on the same
   part number).

5. Functional dependency violations — certain relations are functionally
   single-valued: a person can only be CEO_OF one organisation at a time, a
   part can only have one MANUFACTURER at a time.  Multiple targets for such
   relations across different documents signal a conflict.

Architecture
------------
- Conflict nodes are created in Neo4j as first-class objects.
- Each Conflict links the two conflicting edges via HAS_CONFLICT relationships.
- Status: "open" | "resolved_authority" | "resolved_manual" | "false_positive"
- ContradictionDetector.scan() runs post-ingestion or on-demand.
- ContradictionDetector.resolve() marks a conflict with a chosen winner.

Detection strategies are in graphrag.graph.contradiction_strategies (_ConflictStrategies
mixin) — one method per conflict class — to keep this file focused on the public
API, scan orchestration, and resolution queries.

conflict_type values
--------------------
  directional_reversal  — A→B and B→A for the same relation
  exclusive_state       — entity carries mutually exclusive status from 2 docs
  functional_violation  — many-to-one relation has multiple targets across docs
  positive_negative_pair — RELATES_TO and NEGATIVE_RELATES_TO coexist for the same triple

Retired: `multi_source`
----------------------
A fifth type, `multi_source`, fired on "same (src, rel, tgt) from two
non-superseding docs". That is corroboration, not contradiction — an edge is one
triple, so two documents on it assert the same fact — and it produced 94 of
aerospace's 95 and 61 of automotive's 63 open conflicts, hiding whether the four
real strategies work at all. It is now `_record_corroboration()`, which writes
`independent_source_count` to the edge instead of creating Conflict nodes.
Pre-existing `multi_source` nodes are retired to `status: 'false_positive'` by
`scripts/retire_multi_source_conflicts.py`.
"""

from __future__ import annotations

import structlog

from graphrag.core.tenancy import require_tenant
from graphrag.graph.contradiction_strategies import _ConflictStrategies

log = structlog.get_logger(__name__)


class ContradictionDetector(_ConflictStrategies):
    """
    Detect and persist semantic contradictions as first-class Conflict nodes.

    Usage::

        detector = ContradictionDetector(neo4j_client)
        conflicts = await detector.scan(doc_id="doc_abc")
        await detector.resolve(conflict_id, winner_doc_id, resolved_by="admin")
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    # ── Detection ──────────────────────────────────────────────────────────────

    async def scan(
        self,
        doc_id: str | None = None,
        tenant: str = "default",
        scan_limit: int = 500,
    ) -> list[dict]:
        """
        Scan for contradictions and persist new Conflict nodes.

        Parameters
        ----------
        doc_id : narrow the scan to a single recently-ingested document.
        tenant : restrict detection to one tenant's subgraph. Required —
                 passing a falsy value used to scan every tenant, which made
                 an unscoped call silently cross-tenant instead of failing.
        scan_limit : max candidates per detection type. Use a large value
                     (or 0 for unlimited) for exhaustive maintenance scans.

        Detects (via _ConflictStrategies mixin):
        1. Directional reversals: A→B and B→A for the same relation type.
        2. Mutually exclusive entity states from different docs (exclusive_state).
        3. Functional dependency violations — one-to-one relations with multiple
           targets across documents (functional_violation).
        4. Positive/negative triple pairs (positive_negative_pair).

        Also records `independent_source_count` on corroborated edges — see
        `_record_corroboration()`. That is not a conflict and is excluded from
        the return value.

        Returns list of newly created conflict dicts.
        """
        require_tenant(tenant)
        new_conflicts: list[dict] = []

        # Not a conflict strategy — records how many independent documents assert
        # each edge. Kept in scan() because it shares the supersession-aware
        # source analysis, but it contributes no Conflict nodes.
        corroborated = await self._record_corroboration(doc_id, tenant, scan_limit)

        new_conflicts += await self._detect_directional_reversals(doc_id, tenant, scan_limit)
        new_conflicts += await self._detect_exclusive_states(doc_id, tenant, scan_limit)
        new_conflicts += await self._detect_functional_violations(doc_id, tenant, scan_limit)
        new_conflicts += await self._detect_positive_negative_pairs(doc_id, tenant, scan_limit)

        log.info(
            "contradiction_detector.scan_done",
            doc_id=doc_id or "all",
            tenant=tenant,
            new_conflicts=len(new_conflicts),
            corroborated_edges=corroborated,
        )
        return new_conflicts

    # ── Resolution ─────────────────────────────────────────────────────────────

    async def resolve(
        self,
        conflict_id: str,
        resolution: str,    # "resolved_authority" | "resolved_manual" | "false_positive"
        winner_doc_id: str = "",
        resolved_by: str = "system",
        tenant: str = "default",
    ) -> None:
        """Mark a Conflict as resolved with a chosen outcome."""
        await self._neo4j.run(
            """
            MATCH (c:Conflict {id: $id, tenant: $tenant})
            SET c.status        = $resolution,
                c.resolved_at   = datetime(),
                c.resolved_by   = $resolved_by,
                c.winner_doc_id = $winner_doc_id
            """,
            id=conflict_id,
            resolution=resolution,
            resolved_by=resolved_by,
            winner_doc_id=winner_doc_id,
            tenant=tenant,
        )
        log.info(
            "contradiction_detector.resolved",
            conflict_id=conflict_id,
            resolution=resolution,
            winner=winner_doc_id,
        )

    async def suggest_resolution(self, conflict_id: str, tenant: str = "default") -> dict:
        """Rank the conflict's competing sources by trust; suggest, never apply.

        Each source document is assessed (authority, supersession, validity
        window, staleness). A ``winner`` is suggested only when one current
        source clearly dominates; otherwise the conflict stays ``unresolved``
        and every claim is returned with its trust components, so competing
        claims are never collapsed silently (docs/trust-metadata.md).
        """
        import ast

        from graphrag.graph.trust import rank_conflicting_claims
        from graphrag.graph.validity import document_trust_fields

        require_tenant(tenant)
        rows = await self._neo4j.run(
            "MATCH (c:Conflict {id: $id, tenant: $tenant}) "
            "RETURN c.sources AS sources, c.status AS status, c.src AS src, c.tgt AS tgt, "
            "c.relation AS relation",
            id=conflict_id, tenant=tenant,
        )
        if not rows:
            return {"status": "not_found", "winner": None, "claims": []}
        raw = rows[0].get("sources")
        try:
            parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except (ValueError, SyntaxError):
            parsed = []
        values = parsed.values() if isinstance(parsed, dict) else (parsed or [])
        doc_ids: list[str] = []
        for v in values:
            for item in (v if isinstance(v, (list, tuple, set)) else [v]):
                if isinstance(item, str) and item and item not in doc_ids:
                    doc_ids.append(item)
        docs = await self._neo4j.run(
            "UNWIND $ids AS id MATCH (d:Document {tenant: $tenant, id: id}) "
            "RETURN d.id AS claim, " + document_trust_fields("d"),
            ids=doc_ids, tenant=tenant,
        )
        ranked = rank_conflicting_claims([dict(d) for d in docs])
        return {
            "conflict_id": conflict_id,
            "conflict_status": rows[0].get("status"),
            "status": ranked["status"],
            "suggested_winner_doc_id": ranked["winner"]["claim"] if ranked["winner"] else None,
            "claims": [{"document_id": c["claim"], "score": c["score"], "current": c["current"],
                        "trust": c["trust"]} for c in ranked["claims"]],
        }

    async def get_open_conflicts(
        self,
        limit: int = 50,
        tenant: str = "default",
    ) -> list[dict]:
        """Return unresolved conflicts for ``tenant``, ordered by detection time."""
        require_tenant(tenant)
        return await self._neo4j.run(
            """
            MATCH (c:Conflict {status: 'open'})
            WHERE c.tenant = $tenant
            RETURN c.id            AS conflict_id,
                   c.src           AS src,
                   c.tgt           AS tgt,
                   c.relation      AS relation,
                   c.conflict_type AS conflict_type,
                   c.sources       AS sources,
                   c.tenant        AS tenant,
                   c.detected_at   AS detected_at
            ORDER BY c.detected_at DESC
            LIMIT $limit
            """,
            limit=limit,
            tenant=tenant,
        )

    async def get_open_conflicts_for_entities(
        self,
        entity_names: list[str],
        tenant: str = "default",
    ) -> list[dict]:
        """Return open conflicts touching any of the given entity names.

        Used by the retrieval path (HybridRetriever) to warn the LLM when a
        chunk it's about to answer from mentions an entity that's the subject
        of an unresolved contradiction — see ContextBuilder.build()'s
        "conflicts" section. Detection alone (scan()) doesn't protect a live
        answer from stating a disputed fact as settled; this closes that gap.
        """
        if not entity_names:
            return []
        require_tenant(tenant)
        return await self._neo4j.run(
            """
            MATCH (c:Conflict {status: 'open'})
            WHERE (c.src IN $names OR c.tgt IN $names) AND c.tenant = $tenant
            RETURN c.id            AS conflict_id,
                   c.src           AS src,
                   c.tgt           AS tgt,
                   c.relation      AS relation,
                   c.conflict_type AS conflict_type,
                   c.sources       AS sources
            """,
            names=entity_names,
            tenant=tenant,
        )

    async def conflict_rate(self, tenant: str = "default") -> float:
        """Ratio of open conflicts to total RELATES_TO edges — graph quality metric."""
        require_tenant(tenant)
        params: dict = {"tenant": tenant}
        rows = await self._neo4j.run(
            """
            MATCH ()-[r:RELATES_TO]->() WHERE r.tenant = $tenant
            WITH count(r) AS total_edges
            MATCH (c:Conflict {status: 'open'})
            WHERE c.tenant = $tenant
            WITH total_edges, count(c) AS conflicts
            RETURN CASE WHEN total_edges > 0
                        THEN toFloat(conflicts) / total_edges
                        ELSE 0.0
                   END AS rate
            """,
            **params,
        )
        return float(rows[0]["rate"]) if rows else 0.0
