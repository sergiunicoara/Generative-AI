"""Negative knowledge — first-class representation of "does not" facts.

Problems solved
---------------
1. Open-world assumption — by default a KG makes no claim about missing edges.
   A graph without "A MANUFACTURES B" may mean (a) we don't know, or (b) A
   definitely does NOT manufacture B.  These are epistemically different.

2. Silent conflict detection — if a positive RELATES_TO and a negative
   NEGATIVE_RELATES_TO edge coexist for the same (src, rel, tgt) that is an
   explicit contradiction that ContradictionDetector should surface.

3. Retrieval quality — negative facts are valid answers.  "Does Engine A use
   Fuel Pump B?" can be answered "No, confirmed in doc X" rather than silence.

Architecture
------------
- NEGATIVE_RELATES_TO edges carry the same provenance model as RELATES_TO:
  source_doc_ids, confidence, valid_from/valid_to, recorded_at, tenant.
- NegativeKnowledgeService exposes assert / retract / query methods.
- ContradictionDetector integration: scan_positive_negative_pairs() detects
  triples where both a positive and a negative edge exist.
- Retrieval integration: get_negative_context() returns confirmed absences for
  a set of entities so the LLM can answer "no" with evidence.

Edge properties (mirrors RELATES_TO)
-------------------------------------
  source_doc_ids  list[str]   accumulated provenance (same accumulation logic)
  confidence      float       certainty that this absence holds
  valid_from      str|null    ISO datetime — when the absence became valid
  valid_to        str|null    ISO datetime — when the absence expired (null=still valid)
  recorded_at     str         ISO datetime — transaction time (when we wrote this)
  tenant          str         tenant scope
"""

from __future__ import annotations

from uuid import uuid4

import structlog

log = structlog.get_logger(__name__)


class NegativeKnowledgeService:
    """
    Assert, retract, and query NEGATIVE_RELATES_TO edges.

    Usage::

        svc = NegativeKnowledgeService(neo4j_client)
        await svc.assert_negative(
            src_name="Engine A", src_type="PRODUCT",
            relation="USES",
            tgt_name="Fuel Pump B", tgt_type="PRODUCT",
            tenant="acme",
            doc_id="ad-2024-001",
            confidence=0.95,
        )
        conflicts = await svc.find_positive_negative_conflicts(tenant="acme")
        context   = await svc.get_negative_context(["Engine A", "Engine B"], tenant="acme")
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    # ── Assert / retract ───────────────────────────────────────────────────────

    async def assert_negative(
        self,
        src_name: str,
        src_type: str,
        relation: str,
        tgt_name: str,
        tgt_type: str,
        tenant: str = "default",
        doc_id: str = "",
        confidence: float = 1.0,
        valid_from: str | None = None,
        valid_to: str | None = None,
    ) -> str:
        """
        Assert that the given relation does NOT hold between src and tgt.

        Creates a NEGATIVE_RELATES_TO edge with the same Bayesian confidence
        accumulation used for positive edges.  If a positive RELATES_TO edge
        already exists for this triple a warning is logged — a
        positive/negative conflict is present and should be resolved.

        Returns the edge's unique surrogate ID (stored on the edge as ``neg_id``).
        """
        neg_id = str(uuid4())

        # MERGE on (src, rel, tgt) — accumulate source_doc_ids just like positive edges
        await self._neo4j.run(
            """
            MATCH (s:Entity {name: $src_name, type: $src_type, tenant: $tenant})
            MATCH (t:Entity {name: $tgt_name, type: $tgt_type, tenant: $tenant})
            MERGE (s)-[r:NEGATIVE_RELATES_TO {relation: $relation}]->(t)
            ON CREATE SET r.neg_id        = $neg_id,
                          r.recorded_at   = datetime(),
                          r.source_doc_ids = [$doc_id]
            ON MATCH SET  r.source_doc_ids = CASE
                              WHEN $doc_id IN r.source_doc_ids THEN r.source_doc_ids
                              ELSE r.source_doc_ids + [$doc_id]
                          END
            SET r.confidence  = CASE
                    WHEN r.confidence IS NULL THEN $confidence
                    ELSE 1.0 - (1.0 - r.confidence) * (1.0 - $confidence)
                END,
                r.valid_from  = coalesce($valid_from, r.valid_from),
                r.valid_to    = coalesce($valid_to,   r.valid_to),
                r.tenant      = $tenant
            """,
            src_name=src_name,
            src_type=src_type,
            tgt_name=tgt_name,
            tgt_type=tgt_type,
            relation=relation,
            tenant=tenant,
            doc_id=doc_id,
            confidence=confidence,
            valid_from=valid_from,
            valid_to=valid_to,
            neg_id=neg_id,
        )

        log.info(
            "negative_knowledge.asserted",
            src=src_name,
            rel=relation,
            tgt=tgt_name,
            tenant=tenant,
            doc_id=doc_id,
            confidence=confidence,
        )

        # Warn immediately if a positive edge also exists (conflict)
        conflict_check = await self._neo4j.run(
            """
            MATCH (s:Entity {name: $src_name, type: $src_type, tenant: $tenant})
                  -[r:RELATES_TO {relation: $relation}]->
                  (t:Entity {name: $tgt_name, type: $tgt_type, tenant: $tenant})
            RETURN count(r) AS n
            """,
            src_name=src_name,
            src_type=src_type,
            tgt_name=tgt_name,
            tgt_type=tgt_type,
            relation=relation,
            tenant=tenant,
        )
        if conflict_check and conflict_check[0].get("n", 0) > 0:
            log.warning(
                "negative_knowledge.positive_conflict_exists",
                src=src_name,
                rel=relation,
                tgt=tgt_name,
                tenant=tenant,
                impact="contradictory positive and negative edges for the same triple",
                fix="run ContradictionDetector.scan_positive_negative_pairs() to surface and resolve",
            )

        return neg_id

    async def retract_negative(
        self,
        src_name: str,
        src_type: str,
        relation: str,
        tgt_name: str,
        tgt_type: str,
        tenant: str = "default",
    ) -> bool:
        """
        Remove a NEGATIVE_RELATES_TO edge (retract the negative assertion).

        Used when a new document overrides a prior negative claim.
        Returns True if an edge was found and deleted.
        """
        rows = await self._neo4j.run(
            """
            MATCH (s:Entity {name: $src_name, type: $src_type, tenant: $tenant})
                  -[r:NEGATIVE_RELATES_TO {relation: $relation}]->
                  (t:Entity {name: $tgt_name, type: $tgt_type, tenant: $tenant})
            DELETE r
            RETURN count(r) AS deleted
            """,
            src_name=src_name,
            src_type=src_type,
            tgt_name=tgt_name,
            tgt_type=tgt_type,
            relation=relation,
            tenant=tenant,
        )
        deleted = bool(rows and rows[0].get("deleted", 0) > 0)
        if deleted:
            log.info(
                "negative_knowledge.retracted",
                src=src_name,
                rel=relation,
                tgt=tgt_name,
                tenant=tenant,
            )
        return deleted

    # ── Query ──────────────────────────────────────────────────────────────────

    async def find_positive_negative_conflicts(
        self,
        tenant: str | None = None,
        scan_limit: int = 200,
    ) -> list[dict]:
        """
        Find triples where both a positive RELATES_TO and a negative
        NEGATIVE_RELATES_TO edge exist for the same (src, rel, tgt).

        These are explicit contradictions: one document says A USES B,
        another says A definitely does NOT USE B.

        Returns list of conflict descriptors — each has enough information to
        create a Conflict node via ContradictionDetector.
        """
        tenant_filter = "AND s.tenant = $tenant AND t.tenant = $tenant" if tenant else ""
        limit_clause  = f"LIMIT {scan_limit}" if scan_limit > 0 else ""
        params: dict  = {}
        if tenant:
            params["tenant"] = tenant

        rows = await self._neo4j.run(
            f"""
            MATCH (s:Entity)-[pos:RELATES_TO]->(t:Entity)
            MATCH (s)-[neg:NEGATIVE_RELATES_TO {{relation: pos.relation}}]->(t)
            WHERE true {tenant_filter}
            RETURN s.name              AS src,
                   s.type              AS src_type,
                   t.name              AS tgt,
                   t.type              AS tgt_type,
                   pos.relation        AS relation,
                   pos.source_doc_ids  AS positive_docs,
                   neg.source_doc_ids  AS negative_docs,
                   pos.confidence      AS positive_confidence,
                   neg.confidence      AS negative_confidence
            {limit_clause}
            """,
            **params,
        )

        conflicts = []
        for row in rows:
            conflicts.append({
                "src":                 row["src"],
                "src_type":            row["src_type"],
                "tgt":                 row["tgt"],
                "tgt_type":            row["tgt_type"],
                "relation":            row["relation"],
                "positive_docs":       row.get("positive_docs") or [],
                "negative_docs":       row.get("negative_docs") or [],
                "positive_confidence": row.get("positive_confidence"),
                "negative_confidence": row.get("negative_confidence"),
                "conflict_type":       "positive_negative_pair",
                "resolution":          (
                    "Consult document authority levels; higher-authority doc wins. "
                    "If negative doc supersedes positive, retract positive edge."
                ),
            })

        log.info(
            "negative_knowledge.conflict_scan_done",
            found=len(conflicts),
            tenant=tenant or "all",
        )
        return conflicts

    async def get_negative_context(
        self,
        entity_names: list[str],
        tenant: str = "default",
        as_of: str | None = None,
    ) -> list[dict]:
        """
        Return NEGATIVE_RELATES_TO edges for a set of entities so that the
        retrieval layer can include confirmed-absent facts in the LLM context.

        Used by local_search to add "the following relations are confirmed NOT
        to exist based on document evidence" to the context window.

        Filters by valid_to if ``as_of`` is provided (temporal validity).
        """
        temporal_filter = (
            "AND (r.valid_from IS NULL OR r.valid_from <= $as_of) "
            "AND (r.valid_to   IS NULL OR r.valid_to   > $as_of)"
            if as_of else ""
        )
        params: dict = {"names": entity_names, "tenant": tenant}
        if as_of:
            params["as_of"] = as_of

        rows = await self._neo4j.run(
            f"""
            UNWIND $names AS name
            MATCH (s:Entity {{name: name, tenant: $tenant}})
                  -[r:NEGATIVE_RELATES_TO]->
                  (t:Entity {{tenant: $tenant}})
            WHERE true {temporal_filter}
            RETURN s.name              AS src,
                   s.type              AS src_type,
                   t.name              AS tgt,
                   t.type              AS tgt_type,
                   r.relation          AS relation,
                   r.confidence        AS confidence,
                   r.source_doc_ids    AS source_doc_ids,
                   r.valid_from        AS valid_from,
                   r.valid_to          AS valid_to
            """,
            **params,
        )

        return [
            {
                "src":            row["src"],
                "src_type":       row["src_type"],
                "tgt":            row["tgt"],
                "tgt_type":       row["tgt_type"],
                "relation":       row["relation"],
                "confidence":     row.get("confidence"),
                "source_doc_ids": row.get("source_doc_ids") or [],
                "valid_from":     row.get("valid_from"),
                "valid_to":       row.get("valid_to"),
                "label":          (
                    f"{row['src']} does NOT {row['relation']} {row['tgt']}"
                ),
            }
            for row in rows
        ]

    async def list_all_negatives(
        self,
        tenant: str = "default",
        limit: int = 100,
    ) -> list[dict]:
        """Return all NEGATIVE_RELATES_TO edges for a tenant — for audit/admin."""
        return await self._neo4j.run(
            """
            MATCH (s:Entity {tenant: $tenant})
                  -[r:NEGATIVE_RELATES_TO]->
                  (t:Entity {tenant: $tenant})
            RETURN s.name            AS src,
                   s.type            AS src_type,
                   t.name            AS tgt,
                   t.type            AS tgt_type,
                   r.relation        AS relation,
                   r.confidence      AS confidence,
                   r.source_doc_ids  AS source_doc_ids,
                   r.recorded_at     AS recorded_at
            ORDER BY r.recorded_at DESC
            LIMIT $limit
            """,
            tenant=tenant,
            limit=limit,
        )
