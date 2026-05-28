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

conflict_type values
--------------------
  multi_source          — same (src, rel, tgt) from two non-superseding docs
  directional_reversal  — A→B and B→A for the same relation
  exclusive_state       — entity carries mutually exclusive status from 2 docs
  functional_violation  — many-to-one relation has multiple targets across docs
"""

from __future__ import annotations

from uuid import uuid4

import structlog

log = structlog.get_logger(__name__)


class ContradictionDetector:
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
        tenant: str | None = None,
        scan_limit: int = 500,
    ) -> list[dict]:
        """
        Scan for contradictions and persist new Conflict nodes.

        Parameters
        ----------
        doc_id : narrow the scan to a single recently-ingested document.
        tenant : restrict detection to one tenant's subgraph.
                 Pass None to scan all tenants (single-tenant deployments).
        scan_limit : max candidates per detection type. Use a large value
                     (or 0 for unlimited) for exhaustive maintenance scans.

        Detects:
        1. Same (src, rel, tgt) from two non-superseding docs (multi_source).
        2. Directional reversals: A→B and B→A for the same relation type.
        3. Mutually exclusive entity states from different docs (exclusive_state).
        4. Functional dependency violations — one-to-one relations with multiple
           targets across documents (functional_violation).

        Returns list of newly created conflict dicts.
        """
        new_conflicts: list[dict] = []

        new_conflicts += await self._detect_multi_source_conflicts(doc_id, tenant, scan_limit)
        new_conflicts += await self._detect_directional_reversals(doc_id, tenant, scan_limit)
        new_conflicts += await self._detect_exclusive_states(doc_id, tenant, scan_limit)
        new_conflicts += await self._detect_functional_violations(doc_id, tenant, scan_limit)
        new_conflicts += await self._detect_positive_negative_pairs(doc_id, tenant, scan_limit)

        log.info(
            "contradiction_detector.scan_done",
            doc_id=doc_id or "all",
            tenant=tenant or "all",
            new_conflicts=len(new_conflicts),
        )
        return new_conflicts

    async def _detect_multi_source_conflicts(
        self,
        doc_id: str | None,
        tenant: str | None,
        scan_limit: int,
    ) -> list[dict]:
        """
        Find RELATES_TO edges that carry evidence from 2+ non-superseding
        documents.  The evidence is read from r.source_doc_ids (an accumulated
        list written by merge_relation) rather than a single r.source_doc_id,
        so this detection survives the MERGE-collapse of multiple writes into
        one edge.
        """
        # Build optional filter clauses
        tenant_filter = "AND s.tenant = $tenant AND t.tenant = $tenant" if tenant else ""
        doc_filter    = "AND $doc_id IN r.source_doc_ids" if doc_id else ""
        limit_clause  = f"LIMIT {scan_limit}" if scan_limit > 0 else ""

        params: dict = {}
        if tenant:
            params["tenant"] = tenant
        if doc_id:
            params["doc_id"] = doc_id

        rows = await self._neo4j.run(
            f"""
            MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
            WHERE r.source_doc_ids IS NOT NULL
              AND size(r.source_doc_ids) > 1
              {tenant_filter}
              {doc_filter}
            WITH s.name AS src, t.name AS tgt, r.relation AS rel,
                 r.source_doc_ids AS doc_ids

            // Generate every unordered pair of source documents so that a
            // 3-doc edge (d0 supersedes d1, but d2 is independent) is not
            // silently cleared by checking only the first pair.
            UNWIND range(0, size(doc_ids) - 2) AS i
            UNWIND range(i + 1, size(doc_ids) - 1) AS j
            WITH src, tgt, rel, doc_ids, doc_ids[i] AS d_a, doc_ids[j] AS d_b

            // Check supersession in both directions for this pair
            OPTIONAL MATCH (da:Document {{id: d_a}})-[:SUPERSEDES*]->(db:Document {{id: d_b}})
            OPTIONAL MATCH (db2:Document {{id: d_b}})-[:SUPERSEDES*]->(da2:Document {{id: d_a}})
            WITH src, tgt, rel, doc_ids, d_a, d_b,
                 count(da) AS sup_fwd, count(db2) AS sup_rev
            WHERE sup_fwd = 0 AND sup_rev = 0   // independent pair — real conflict

            // Re-aggregate: conflict exists if any independent pair remains
            WITH src, tgt, rel, doc_ids,
                 collect({{a: d_a, b: d_b}}) AS independent_pairs
            WHERE size(independent_pairs) > 0

            // Skip if already flagged as open
            OPTIONAL MATCH (c:Conflict {{src: src, tgt: tgt, relation: rel,
                                         status: 'open', conflict_type: 'multi_source'}})
            WITH src, tgt, rel, doc_ids, independent_pairs, count(c) AS existing
            WHERE existing = 0
            RETURN src, tgt, rel, doc_ids, independent_pairs
            {limit_clause}
            """,
            **params,
        )

        created: list[dict] = []
        for row in rows:
            conflict_id = str(uuid4())
            await self._neo4j.run(
                """
                CREATE (c:Conflict {
                    id:            $id,
                    src:           $src,
                    tgt:           $tgt,
                    relation:      $rel,
                    conflict_type: 'multi_source',
                    sources:       $sources,
                    tenant:        $tenant,
                    status:        'open',
                    detected_at:   datetime(),
                    resolved_at:   null,
                    resolved_by:   null,
                    winner_doc_id: null
                })
                """,
                id=conflict_id,
                src=row["src"],
                tgt=row["tgt"],
                rel=row["rel"],
                sources=str(row["doc_ids"]),
                tenant=tenant or "",
            )
            created.append({
                "conflict_id": conflict_id,
                "type": "multi_source",
                "src": row["src"],
                "tgt": row["tgt"],
                "relation": row["rel"],
            })
        return created

    async def _detect_directional_reversals(
        self,
        doc_id: str | None,
        tenant: str | None,
        scan_limit: int,
    ) -> list[dict]:
        """
        Find cases where A-[rel]->B AND B-[rel]->A both exist for the same
        relation type — a topological contradiction.
        """
        tenant_filter = "AND a.tenant = $tenant AND b.tenant = $tenant" if tenant else ""
        limit_clause  = f"LIMIT {scan_limit}" if scan_limit > 0 else ""
        params: dict = {}
        if tenant:
            params["tenant"] = tenant

        rows = await self._neo4j.run(
            f"""
            MATCH (a:Entity)-[r1:RELATES_TO]->(b:Entity)
            MATCH (b)-[r2:RELATES_TO]->(a)
            WHERE r1.relation = r2.relation
              AND id(a) < id(b)
              {tenant_filter}
            OPTIONAL MATCH (c:Conflict {{
                src: a.name, tgt: b.name,
                relation: r1.relation,
                conflict_type: 'directional_reversal',
                status: 'open'
            }})
            WITH a.name AS src, b.name AS tgt, r1.relation AS rel,
                 r1.source_doc_id AS doc1, r2.source_doc_id AS doc2,
                 count(c) AS existing
            WHERE existing = 0
            RETURN src, tgt, rel, doc1, doc2
            {limit_clause}
            """,
            **params,
        )

        created: list[dict] = []
        for row in rows:
            conflict_id = str(uuid4())
            await self._neo4j.run(
                """
                CREATE (c:Conflict {
                    id:            $id,
                    src:           $src,
                    tgt:           $tgt,
                    relation:      $rel,
                    conflict_type: 'directional_reversal',
                    sources:       $sources,
                    tenant:        $tenant,
                    status:        'open',
                    detected_at:   datetime(),
                    resolved_at:   null,
                    resolved_by:   null,
                    winner_doc_id: null
                })
                """,
                id=conflict_id,
                src=row["src"],
                tgt=row["tgt"],
                rel=row["rel"],
                sources=str([row["doc1"], row["doc2"]]),
                tenant=tenant or "",
            )
            created.append({
                "conflict_id": conflict_id,
                "type": "directional_reversal",
                "src": row["src"],
                "tgt": row["tgt"],
                "relation": row["rel"],
            })
        return created

    async def _detect_exclusive_states(
        self,
        doc_id: str | None,
        tenant: str | None,
        scan_limit: int,
    ) -> list[dict]:
        """
        Detect entities that carry mutually exclusive status values sourced from
        different documents.  Exclusivity pairs are defined below — if both
        sides of a pair appear on the same entity across different docs a
        Conflict of type 'exclusive_state' is raised.

        Example: Entity has source_doc_id A asserting status='active' and
        source_doc_id B asserting status='deprecated'.
        """
        # We detect via RELATES_TO edges whose relation encodes a state assertion:
        # e.g. IS_ACTIVE, IS_DEPRECATED, IS_APPROVED, IS_REJECTED, etc.
        # Pairs that cannot coexist:
        exclusive_pairs = [
            ("IS_ACTIVE",     "IS_DEPRECATED"),
            ("IS_APPROVED",   "IS_REJECTED"),
            ("IS_CERTIFIED",  "IS_UNCERTIFIED"),
            ("OPERATIONAL",   "DECOMMISSIONED"),
        ]

        tenant_filter = "AND e.tenant = $tenant" if tenant else ""
        limit_clause  = f"LIMIT {scan_limit}" if scan_limit > 0 else ""

        created: list[dict] = []
        for rel_a, rel_b in exclusive_pairs:
            params: dict = {"rel_a": rel_a, "rel_b": rel_b}
            if tenant:
                params["tenant"] = tenant

            rows = await self._neo4j.run(
                f"""
                MATCH (e:Entity)-[r1:RELATES_TO {{relation: $rel_a}}]->(s1)
                MATCH (e)-[r2:RELATES_TO {{relation: $rel_b}}]->(s2)
                WHERE r1.source_doc_id IS NOT NULL
                  AND r2.source_doc_id IS NOT NULL
                  AND r1.source_doc_id <> r2.source_doc_id
                  {tenant_filter}
                OPTIONAL MATCH (c:Conflict {{
                    src: e.name,
                    conflict_type: 'exclusive_state',
                    status: 'open'
                }})
                WHERE c.relation IN [$rel_a, $rel_b]
                WITH e.name AS entity, r1.source_doc_id AS doc_a,
                     r2.source_doc_id AS doc_b, count(c) AS existing
                WHERE existing = 0
                RETURN entity, doc_a, doc_b
                {limit_clause}
                """,
                **params,
            )
            for row in rows:
                conflict_id = str(uuid4())
                label = f"{rel_a} vs {rel_b}"
                await self._neo4j.run(
                    """
                    CREATE (c:Conflict {
                        id:            $id,
                        src:           $entity,
                        tgt:           '',
                        relation:      $label,
                        conflict_type: 'exclusive_state',
                        sources:       $sources,
                        tenant:        $tenant,
                        status:        'open',
                        detected_at:   datetime(),
                        resolved_at:   null,
                        resolved_by:   null,
                        winner_doc_id: null
                    })
                    """,
                    id=conflict_id,
                    entity=row["entity"],
                    label=label,
                    sources=str([row["doc_a"], row["doc_b"]]),
                    tenant=tenant or "",
                )
                created.append({
                    "conflict_id": conflict_id,
                    "type": "exclusive_state",
                    "src": row["entity"],
                    "relation": label,
                })
        return created

    async def _detect_functional_violations(
        self,
        doc_id: str | None,
        tenant: str | None,
        scan_limit: int,
    ) -> list[dict]:
        """
        Detect violations of functional (many-to-one) relation constraints.

        A functional relation must have at most one target per source entity.
        If two documents give different targets for a functional relation the
        graph contains an implicit contradiction about which claim is true.

        Functional relations in this domain:
          - CEO_OF     : a person leads at most one org (per time window)
          - FOUNDED_BY : an org has at most one founding person listed
          - MANUFACTURES: a product is manufactured by one org per doc scope

        Uses r.source_doc_ids (accumulated list) rather than r.source_doc_id
        so multi-document evidence is visible after MERGE.
        """
        functional_relations = ["CEO_OF", "FOUNDED_BY", "MANUFACTURES"]
        tenant_filter = "AND s.tenant = $tenant" if tenant else ""
        limit_clause  = f"LIMIT {scan_limit}" if scan_limit > 0 else ""

        created: list[dict] = []
        for rel in functional_relations:
            params: dict = {"rel": rel}
            if tenant:
                params["tenant"] = tenant

            rows = await self._neo4j.run(
                f"""
                MATCH (s:Entity)-[r:RELATES_TO {{relation: $rel}}]->(t:Entity)
                WHERE r.source_doc_ids IS NOT NULL
                  {tenant_filter}
                // reduce(...) flattens list-of-lists without APOC.
                // Equivalent to APOC's apoc.coll.flatten but pure Cypher 4+.
                WITH s.name AS src, collect(DISTINCT t.name) AS targets,
                     reduce(acc = [], lst IN collect(r.source_doc_ids) | acc + lst) AS all_docs
                WHERE size(targets) > 1
                OPTIONAL MATCH (c:Conflict {{
                    src: src, relation: $rel,
                    conflict_type: 'functional_violation',
                    status: 'open'
                }})
                WITH src, targets, all_docs, count(c) AS existing
                WHERE existing = 0
                RETURN src, targets, all_docs AS docs
                {limit_clause}
                """,
                **params,
            )
            for row in rows:
                conflict_id = str(uuid4())
                await self._neo4j.run(
                    """
                    CREATE (c:Conflict {
                        id:            $id,
                        src:           $src,
                        tgt:           $tgt,
                        relation:      $rel,
                        conflict_type: 'functional_violation',
                        sources:       $sources,
                        tenant:        $tenant,
                        status:        'open',
                        detected_at:   datetime(),
                        resolved_at:   null,
                        resolved_by:   null,
                        winner_doc_id: null
                    })
                    """,
                    id=conflict_id,
                    src=row["src"],
                    tgt=str(row["targets"]),
                    rel=rel,
                    sources=str(row["docs"]),
                    tenant=tenant or "",
                )
                created.append({
                    "conflict_id": conflict_id,
                    "type": "functional_violation",
                    "src": row["src"],
                    "relation": rel,
                    "targets": row["targets"],
                })
        return created

    async def _detect_positive_negative_pairs(
        self,
        doc_id: str | None,
        tenant: str | None,
        scan_limit: int,
    ) -> list[dict]:
        """
        Detect triples where both a RELATES_TO and a NEGATIVE_RELATES_TO edge
        coexist for the same (src, relation, tgt) — an explicit contradiction
        between a positive and a negative knowledge assertion.

        A document saying "A USES B" and another saying "A definitely does NOT
        USE B" represent incompatible epistemic claims that require resolution
        by document authority or manual review.
        """
        tenant_filter = "AND s.tenant = $tenant AND t.tenant = $tenant" if tenant else ""
        doc_filter    = (
            "AND ($doc_id IN pos.source_doc_ids OR $doc_id IN neg.source_doc_ids)"
            if doc_id else ""
        )
        limit_clause  = f"LIMIT {scan_limit}" if scan_limit > 0 else ""
        params: dict  = {}
        if tenant:
            params["tenant"] = tenant
        if doc_id:
            params["doc_id"] = doc_id

        rows = await self._neo4j.run(
            f"""
            MATCH (s:Entity)-[pos:RELATES_TO]->(t:Entity)
            MATCH (s)-[neg:NEGATIVE_RELATES_TO {{relation: pos.relation}}]->(t)
            WHERE true {tenant_filter} {doc_filter}
            OPTIONAL MATCH (c:Conflict {{
                src: s.name, tgt: t.name,
                relation: pos.relation,
                conflict_type: 'positive_negative_pair',
                status: 'open'
            }})
            WITH s.name AS src, t.name AS tgt, pos.relation AS rel,
                 pos.source_doc_ids AS pos_docs, neg.source_doc_ids AS neg_docs,
                 count(c) AS existing
            WHERE existing = 0
            RETURN src, tgt, rel, pos_docs, neg_docs
            {limit_clause}
            """,
            **params,
        )

        created: list[dict] = []
        for row in rows:
            conflict_id = str(uuid4())
            await self._neo4j.run(
                """
                CREATE (c:Conflict {
                    id:            $id,
                    src:           $src,
                    tgt:           $tgt,
                    relation:      $rel,
                    conflict_type: 'positive_negative_pair',
                    sources:       $sources,
                    tenant:        $tenant,
                    status:        'open',
                    detected_at:   datetime(),
                    resolved_at:   null,
                    resolved_by:   null,
                    winner_doc_id: null
                })
                """,
                id=conflict_id,
                src=row["src"],
                tgt=row["tgt"],
                rel=row["rel"],
                sources=str({
                    "positive": row.get("pos_docs") or [],
                    "negative": row.get("neg_docs") or [],
                }),
                tenant=tenant or "",
            )
            created.append({
                "conflict_id": conflict_id,
                "type":        "positive_negative_pair",
                "src":         row["src"],
                "tgt":         row["tgt"],
                "relation":    row["rel"],
                "positive_docs": row.get("pos_docs") or [],
                "negative_docs": row.get("neg_docs") or [],
            })
        return created

    # ── Resolution ─────────────────────────────────────────────────────────────

    async def resolve(
        self,
        conflict_id: str,
        resolution: str,    # "resolved_authority" | "resolved_manual" | "false_positive"
        winner_doc_id: str = "",
        resolved_by: str = "system",
    ) -> None:
        """Mark a Conflict as resolved with a chosen outcome."""
        await self._neo4j.run(
            """
            MATCH (c:Conflict {id: $id})
            SET c.status        = $resolution,
                c.resolved_at   = datetime(),
                c.resolved_by   = $resolved_by,
                c.winner_doc_id = $winner_doc_id
            """,
            id=conflict_id,
            resolution=resolution,
            resolved_by=resolved_by,
            winner_doc_id=winner_doc_id,
        )
        log.info(
            "contradiction_detector.resolved",
            conflict_id=conflict_id,
            resolution=resolution,
            winner=winner_doc_id,
        )

    async def get_open_conflicts(
        self,
        limit: int = 50,
        tenant: str | None = None,
    ) -> list[dict]:
        """Return unresolved conflicts ordered by detection time, optionally tenant-filtered."""
        tenant_filter = "AND c.tenant = $tenant" if tenant else ""
        params: dict = {"limit": limit}
        if tenant:
            params["tenant"] = tenant
        return await self._neo4j.run(
            f"""
            MATCH (c:Conflict {{status: 'open'}})
            WHERE true {tenant_filter}
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
            **params,
        )

    async def conflict_rate(self, tenant: str | None = None) -> float:
        """Ratio of open conflicts to total RELATES_TO edges — graph quality metric.
        Optionally scoped to a single tenant's edges and conflicts."""
        tenant_edge_filter   = "WHERE r.tenant = $tenant" if tenant else ""
        tenant_conflict_filter = "AND c.tenant = $tenant" if tenant else ""
        params: dict = {}
        if tenant:
            params["tenant"] = tenant
        rows = await self._neo4j.run(
            f"""
            MATCH ()-[r:RELATES_TO]->() {tenant_edge_filter}
            WITH count(r) AS total_edges
            MATCH (c:Conflict {{status: 'open'}})
            WHERE true {tenant_conflict_filter}
            WITH total_edges, count(c) AS conflicts
            RETURN CASE WHEN total_edges > 0
                        THEN toFloat(conflicts) / total_edges
                        ELSE 0.0
                   END AS rate
            """,
            **params,
        )
        return float(rows[0]["rate"]) if rows else 0.0
