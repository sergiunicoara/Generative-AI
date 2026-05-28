"""Relation reification — Statement nodes for meta-statements and rich provenance.

Problems solved
---------------
1. Properties are a dead end — RELATES_TO edges carry metadata (confidence,
   source_doc_ids, etc.) as flat properties.  There is no way to make a
   statement *about* one of those properties, or to say "Claim X was contested
   by Claim Y" without creating a separate Conflict node with a free-text link.

2. No provenance chain — you can say "A CEO_OF B came from doc X", but you
   cannot say "The CEO_OF claim was endorsed by Agent Z with confidence 0.7"
   without ad-hoc schema extensions.

3. Quotation and context — "According to the 2019 annual report, Engine A
   uses Fuel Pump B" is a different epistemic object than the bare triple.
   Reification turns the claim itself into a node that can carry context.

What is reification?
--------------------
Reification converts a triple (s)-[r:RELATES_TO {relation: "R"}]->(t) into a
first-class Statement node:

    (s)-[:SUBJECT_OF]->(stmt:Statement {relation:"R"})-[:OBJECT_OF]->(t)

The original RELATES_TO edge is preserved for GNN traversal efficiency.
The Statement node can then participate in higher-order claims:

    (agent)-[:ENDORSED]->(stmt)
    (doc)-[:CITES]->(stmt)
    (stmt2:Statement {relation:"CONTRADICTS"})-[:SUBJECT_OF]->(stmt)

Architecture
------------
- ReificationService creates Statement nodes on demand — not all edges need
  to be reified; only those where meta-statements are required.
- reify_relation()  — create Statement from an existing RELATES_TO edge.
- add_meta()        — attach arbitrary key/value metadata to a Statement.
- endorse()         — add an ENDORSED_BY link (agent, doc, or Statement).
- contradict()      — assert that stmt A contradicts stmt B.
- get_statements()  — retrieve all Statements for an entity.
- as_triples()      — export Statements as plain (s, r, t) dicts for display.

Statement node properties
--------------------------
  id             UUID
  relation       str         e.g. "CEO_OF"
  src_name       str         denormalized for fast reads
  src_type       str
  tgt_name       str
  tgt_type       str
  confidence     float       inherited from the originating edge
  source_doc_ids list[str]   inherited from the originating edge
  tenant         str
  reified_at     datetime
"""

from __future__ import annotations

from uuid import uuid4

import structlog

log = structlog.get_logger(__name__)


class ReificationService:
    """
    Create and query Statement nodes for higher-order claims.

    Usage::

        svc = ReificationService(neo4j_client)

        # Reify a specific relation
        stmt_id = await svc.reify_relation(
            src_name="Elon Musk", src_type="PERSON",
            relation="CEO_OF",
            tgt_name="SpaceX", tgt_type="ORG",
            tenant="default",
        )

        # Add rich context
        await svc.add_meta(stmt_id, key="context",
                           value="As stated in 2022 annual report")
        await svc.endorse(stmt_id, endorser_id="doc_2022_annual_report",
                          endorser_type="Document", confidence=0.99)

        # Meta-statement: this statement was retracted
        stmt2_id = await svc.reify_relation(
            src_name="Elon Musk", src_type="PERSON",
            relation="NO_LONGER_CEO_OF",
            tgt_name="Twitter", tgt_type="ORG",
            tenant="default",
        )
        await svc.contradict(stmt_id, stmt2_id, reason="Leadership change 2024")

        # Query
        stmts = await svc.get_statements("Elon Musk", "PERSON", tenant="default")
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    # ── Reification ────────────────────────────────────────────────────────────

    async def reify_relation(
        self,
        src_name: str,
        src_type: str,
        relation: str,
        tgt_name: str,
        tgt_type: str,
        tenant: str = "default",
        confidence: float | None = None,
        source_doc_ids: list[str] | None = None,
    ) -> str:
        """
        Create a Statement node for the given triple, adding SUBJECT_OF and
        OBJECT_OF edges.  The original RELATES_TO edge is NOT removed — both
        the edge (for GNN traversal) and the Statement node (for meta-claims)
        coexist.

        If a Statement for this exact (src, relation, tgt, tenant) already
        exists, the existing node's ID is returned (idempotent).

        Returns the Statement node ID.
        """
        # Inherit confidence and source_doc_ids from the existing edge
        # unless the caller supplies them explicitly.
        if confidence is None or source_doc_ids is None:
            edge_rows = await self._neo4j.run(
                """
                MATCH (s:Entity {name: $src_name, type: $src_type, tenant: $tenant})
                      -[r:RELATES_TO {relation: $relation}]->
                      (t:Entity {name: $tgt_name, type: $tgt_type, tenant: $tenant})
                RETURN r.confidence    AS confidence,
                       r.source_doc_ids AS source_doc_ids
                """,
                src_name=src_name,
                src_type=src_type,
                tgt_name=tgt_name,
                tgt_type=tgt_type,
                relation=relation,
                tenant=tenant,
            )
            if edge_rows:
                confidence     = confidence     or edge_rows[0].get("confidence")     or 1.0
                source_doc_ids = source_doc_ids or edge_rows[0].get("source_doc_ids") or []

        stmt_id = str(uuid4())

        result = await self._neo4j.run(
            """
            MATCH (s:Entity {name: $src_name, type: $src_type, tenant: $tenant})
            MATCH (t:Entity {name: $tgt_name, type: $tgt_type, tenant: $tenant})
            MERGE (stmt:Statement {
                src_name: $src_name, src_type: $src_type,
                tgt_name: $tgt_name, tgt_type: $tgt_type,
                relation:  $relation, tenant: $tenant
            })
            ON CREATE SET stmt.id             = $stmt_id,
                          stmt.confidence     = $confidence,
                          stmt.source_doc_ids = $source_doc_ids,
                          stmt.reified_at     = datetime()
            MERGE (s)-[:SUBJECT_OF]->(stmt)
            MERGE (stmt)-[:OBJECT_OF]->(t)
            RETURN stmt.id AS id
            """,
            stmt_id=stmt_id,
            src_name=src_name,
            src_type=src_type,
            tgt_name=tgt_name,
            tgt_type=tgt_type,
            relation=relation,
            tenant=tenant,
            confidence=confidence or 1.0,
            source_doc_ids=source_doc_ids or [],
        )

        actual_id = result[0]["id"] if result else stmt_id
        log.info(
            "reification.statement_created",
            stmt_id=actual_id,
            src=src_name,
            rel=relation,
            tgt=tgt_name,
            tenant=tenant,
        )
        return actual_id

    # ── Meta-statement operations ──────────────────────────────────────────────

    async def add_meta(
        self,
        stmt_id: str,
        key: str,
        value: str,
    ) -> None:
        """
        Attach a key/value annotation to a Statement node.

        Examples:
          key="context"   value="As stated in 2022 annual report"
          key="qualifier" value="only applies during fiscal year 2023"
          key="certainty" value="speculative"
        """
        await self._neo4j.run(
            """
            MATCH (stmt:Statement {id: $stmt_id})
            SET stmt[$key] = $value
            """,
            stmt_id=stmt_id,
            key=key,
            value=value,
        )

    async def endorse(
        self,
        stmt_id: str,
        endorser_id: str,
        endorser_type: str = "Document",
        confidence: float = 1.0,
        note: str = "",
    ) -> None:
        """
        Add an ENDORSED_BY link from a Statement to an endorsing entity/document.

        Endorsers can be Document nodes, Entity nodes (experts), or other
        Statement nodes (endorsement chains).
        """
        await self._neo4j.run(
            """
            MATCH (stmt:Statement {id: $stmt_id})
            MERGE (endorser {id: $endorser_id})
            MERGE (stmt)-[e:ENDORSED_BY]->(endorser)
            ON CREATE SET e.confidence = $confidence,
                          e.note       = $note,
                          e.endorsed_at = datetime()
            """,
            stmt_id=stmt_id,
            endorser_id=endorser_id,
            confidence=confidence,
            note=note,
        )

    async def contradict(
        self,
        stmt_a_id: str,
        stmt_b_id: str,
        reason: str = "",
    ) -> None:
        """
        Assert that Statement A contradicts Statement B.

        This is a meta-statement: a CONTRADICTS relationship between two
        Statement nodes.  It is the reification equivalent of Conflict nodes —
        both can coexist (Conflicts are for automated detection, CONTRADICTS
        is for explicit human or automated assertion).
        """
        await self._neo4j.run(
            """
            MATCH (a:Statement {id: $a_id})
            MATCH (b:Statement {id: $b_id})
            MERGE (a)-[c:CONTRADICTS]->(b)
            ON CREATE SET c.reason = $reason,
                          c.asserted_at = datetime()
            """,
            a_id=stmt_a_id,
            b_id=stmt_b_id,
            reason=reason,
        )
        log.info(
            "reification.contradiction_asserted",
            stmt_a=stmt_a_id,
            stmt_b=stmt_b_id,
            reason=reason,
        )

    # ── Query ──────────────────────────────────────────────────────────────────

    async def get_statements(
        self,
        entity_name: str,
        entity_type: str,
        tenant: str = "default",
        role: str = "subject",      # "subject" | "object" | "both"
    ) -> list[dict]:
        """
        Return Statement nodes involving ``entity_name`` as subject, object, or both.

        Each result includes ENDORSED_BY links and CONTRADICTS meta-relations.
        """
        if role == "subject":
            match_clause = """
                MATCH (e:Entity {name: $name, type: $type, tenant: $tenant})
                      -[:SUBJECT_OF]->(stmt:Statement {tenant: $tenant})
            """
        elif role == "object":
            match_clause = """
                MATCH (stmt:Statement {tenant: $tenant})
                      -[:OBJECT_OF]->(e:Entity {name: $name, type: $type, tenant: $tenant})
            """
        else:   # both
            match_clause = """
                MATCH (e:Entity {name: $name, type: $type, tenant: $tenant})
                WHERE (e)-[:SUBJECT_OF]->(:Statement {tenant: $tenant})
                   OR (:Statement {tenant: $tenant})-[:OBJECT_OF]->(e)
                MATCH (stmt:Statement {tenant: $tenant})
                WHERE (e)-[:SUBJECT_OF]->(stmt) OR (stmt)-[:OBJECT_OF]->(e)
            """

        rows = await self._neo4j.run(
            f"""
            {match_clause}
            OPTIONAL MATCH (stmt)-[:ENDORSED_BY]->(endorser)
            OPTIONAL MATCH (stmt)-[c:CONTRADICTS]->(other:Statement)
            RETURN stmt.id             AS stmt_id,
                   stmt.src_name       AS src_name,
                   stmt.src_type       AS src_type,
                   stmt.relation       AS relation,
                   stmt.tgt_name       AS tgt_name,
                   stmt.tgt_type       AS tgt_type,
                   stmt.confidence     AS confidence,
                   stmt.source_doc_ids AS source_doc_ids,
                   stmt.reified_at     AS reified_at,
                   collect(DISTINCT endorser.id) AS endorsers,
                   collect(DISTINCT other.id)    AS contradicts
            """,
            name=entity_name,
            type=entity_type,
            tenant=tenant,
        )
        return [
            {
                "stmt_id":        row["stmt_id"],
                "src":            row["src_name"],
                "src_type":       row["src_type"],
                "relation":       row["relation"],
                "tgt":            row["tgt_name"],
                "tgt_type":       row["tgt_type"],
                "confidence":     row.get("confidence"),
                "source_doc_ids": row.get("source_doc_ids") or [],
                "reified_at":     row.get("reified_at"),
                "endorsers":      [e for e in (row.get("endorsers") or []) if e],
                "contradicts":    [c for c in (row.get("contradicts") or []) if c],
            }
            for row in rows
            if row.get("stmt_id")
        ]

    async def as_triples(
        self,
        tenant: str = "default",
        limit: int = 100,
    ) -> list[dict]:
        """
        Export all Statement nodes as plain (s, r, t) dicts — for display or export.
        """
        return await self._neo4j.run(
            """
            MATCH (stmt:Statement {tenant: $tenant})
            RETURN stmt.src_name   AS src,
                   stmt.relation   AS relation,
                   stmt.tgt_name   AS tgt,
                   stmt.confidence AS confidence,
                   stmt.reified_at AS reified_at
            ORDER BY stmt.reified_at DESC
            LIMIT $limit
            """,
            tenant=tenant,
            limit=limit,
        )
