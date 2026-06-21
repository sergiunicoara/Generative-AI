"""Detection strategies for ContradictionDetector (mixin).

Each method is a private detection strategy that scans for one class of
semantic contradiction and returns a list of newly-created Conflict dicts.
They share a single protocol: they receive (doc_id, tenant, scan_limit) and
use ``self._neo4j`` from the host class.

Import pattern::

    from graphrag.graph.contradiction_strategies import _ConflictStrategies

    class ContradictionDetector(_ConflictStrategies):
        ...
"""

from __future__ import annotations

from uuid import uuid4

import structlog

log = structlog.get_logger(__name__)


class _ConflictStrategies:
    """Mixin — five detection strategies, each surfacing a different conflict class."""

    # ── Strategy 1: Multi-source conflicts ────────────────────────────────────

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

            UNWIND range(0, size(doc_ids) - 2) AS i
            UNWIND range(i + 1, size(doc_ids) - 1) AS j
            WITH src, tgt, rel, doc_ids, doc_ids[i] AS d_a, doc_ids[j] AS d_b

            OPTIONAL MATCH (da:Document {{id: d_a}})-[:SUPERSEDES*]->(db:Document {{id: d_b}})
            OPTIONAL MATCH (db2:Document {{id: d_b}})-[:SUPERSEDES*]->(da2:Document {{id: d_a}})
            WITH src, tgt, rel, doc_ids, d_a, d_b,
                 count(da) AS sup_fwd, count(db2) AS sup_rev
            WHERE sup_fwd = 0 AND sup_rev = 0

            WITH src, tgt, rel, doc_ids,
                 collect({{a: d_a, b: d_b}}) AS independent_pairs
            WHERE size(independent_pairs) > 0

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

    # ── Strategy 2: Directional reversals ────────────────────────────────────

    async def _detect_directional_reversals(
        self,
        doc_id: str | None,
        tenant: str | None,
        scan_limit: int,
    ) -> list[dict]:
        """
        Find cases where A-[rel]->B AND B-[rel]->A both exist for the same
        relation type — a topological contradiction.

        RELATED_TO is excluded: it's the generic fallback relation with no
        domain/range constraints (see ontology_registry.py), so A-RELATED_TO->B
        and B-RELATED_TO->A are just two entities co-mentioned in both
        directions — not a directional claim that can be "reversed".
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
              AND r1.relation <> 'RELATED_TO'
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

    # ── Strategy 3: Exclusive states ─────────────────────────────────────────

    async def _detect_exclusive_states(
        self,
        doc_id: str | None,
        tenant: str | None,
        scan_limit: int,
    ) -> list[dict]:
        """
        Detect entities that carry mutually exclusive status values sourced from
        different documents (e.g. IS_ACTIVE and IS_DEPRECATED on the same entity).
        """
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

    # ── Strategy 4: Functional violations ────────────────────────────────────

    async def _detect_functional_violations(
        self,
        doc_id: str | None,
        tenant: str | None,
        scan_limit: int,
    ) -> list[dict]:
        """
        Detect violations of functional (many-to-one) relation constraints.
        A functional relation must have at most one target per source entity.
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

    # ── Strategy 5: Positive-negative pairs ──────────────────────────────────

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
