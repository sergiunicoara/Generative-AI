"""Ingestion validation — guards against graph poisoning.

Problem solved
--------------
Bad data from LLM extraction or corrupt documents can introduce:
  - Orphan nodes (entities with no relationships and no chunk link)
  - Degree anomalies (one entity suddenly connected to 1000 others)
  - Isolated components disconnected from the main graph
  - Self-loop edges (entity relates to itself)
  - Statistically abnormal confidence distributions

Each of these can silently degrade retrieval quality, GNN propagation,
and community detection without raising an obvious error.

Validation runs as a post-write check after every ingestion batch.
Findings are logged as warnings — they don't block ingestion, but they
are surfaced for human review.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)

# Anomaly thresholds
# For sparse domain graphs (< 10k entities), hub entities naturally have
# 20–50× mean degree (e.g. FAA, Boeing in a regulatory corpus). A multiplier
# of 5 would quarantine every important entity. 20 catches genuine poisoning
# (a hallucinated node linked to everything) while leaving real hubs alone.
MAX_DEGREE_MULTIPLIER  = 20.0  # flag if degree > mean * this
MIN_CONFIDENCE         = 0.1   # flag edges with suspiciously low confidence
MAX_ORPHAN_RATE        = 0.10  # flag if > 10% of new entities are orphans
RELATION_RULES: dict[str, set[tuple[str, str]]] = {
    "FOUNDED": {("PERSON", "ORG"), ("PERSON", "PRODUCT")},
    "FOUNDED_BY": {("ORG", "PERSON"), ("PRODUCT", "PERSON")},
    "CEO_OF": {("PERSON", "ORG")},
    "OWNS": {("PERSON", "ORG"), ("ORG", "ORG"), ("ORG", "PRODUCT")},
    "ACQUIRED": {("ORG", "ORG"), ("ORG", "PRODUCT")},
    "MANUFACTURES": {("ORG", "PRODUCT")},
    "LAUNCHED": {("ORG", "PRODUCT"), ("PERSON", "PRODUCT")},
    "WORKS_AT": {("PERSON", "ORG")},
    "LOCATED_IN": {
        ("ORG", "LOCATION"),
        ("PERSON", "LOCATION"),
        ("EVENT", "LOCATION"),
    },
}


class IngestionValidator:
    """
    Post-ingestion graph health checker.

    Usage::

        validator = IngestionValidator(neo4j_client)
        report = await validator.validate(doc_id="doc_abc")
        if report["issues"]:
            log.warning("ingestion_issues", **report)
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    async def validate(self, doc_id: str | None = None) -> dict:
        """
        Run all checks. If doc_id is given, scopes checks to that
        document's entities only. Returns a structured report.
        """
        issues: list[dict] = []

        issues += await self._check_self_loops(doc_id)
        issues += await self._check_orphan_entities(doc_id)
        issues += await self._check_degree_anomalies(doc_id)
        issues += await self._check_low_confidence_edges(doc_id)
        issues += await self._check_relation_schema(doc_id)

        report = {
            "doc_id": doc_id,
            "total_issues": len(issues),
            "issues": issues,
        }

        if issues:
            log.warning(
                "ingestion_validator.issues_found",
                doc_id=doc_id,
                count=len(issues),
                types=list({i["type"] for i in issues}),
            )
        else:
            log.info("ingestion_validator.clean", doc_id=doc_id)

        return report

    # ── Individual checks ──────────────────────────────────────────────────────

    async def _check_self_loops(self, doc_id: str | None) -> list[dict]:
        """Entities that relate to themselves."""
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity)-[r:RELATES_TO]->(e)
            RETURN e.name AS entity, r.relation AS relation
            LIMIT 50
            """
        )
        return [
            {"type": "self_loop", "entity": r["entity"], "relation": r["relation"]}
            for r in rows
        ]

    async def _check_orphan_entities(self, doc_id: str | None) -> list[dict]:
        """Entities with no MENTIONS link to any chunk."""
        scope = "WHERE c.document_id = $doc_id" if doc_id else ""
        params = {"doc_id": doc_id} if doc_id else {}
        query = f"""
            MATCH (e:Entity)
            WHERE NOT (e)<-[:MENTIONS]-(:Chunk)
            {scope.replace('WHERE', 'AND') if scope else ''}
            RETURN e.name AS entity, e.type AS type
            LIMIT 100
        """
        # For scoped check: find entities from this doc's chunks only
        if doc_id:
            rows = await self._neo4j.run(
                """
                MATCH (c:Chunk {document_id: $doc_id})-[:MENTIONS]->(e:Entity)
                WHERE NOT EXISTS {
                    MATCH (e)<-[:MENTIONS]-(:Chunk)
                }
                RETURN e.name AS entity, e.type AS type
                LIMIT 100
                """,
                doc_id=doc_id,
            )
        else:
            rows = await self._neo4j.run(
                """
                MATCH (e:Entity)
                WHERE NOT (e)<-[:MENTIONS]-(:Chunk)
                RETURN e.name AS entity, e.type AS type
                LIMIT 100
                """
            )
        return [
            {"type": "orphan_entity", "entity": r["entity"], "entity_type": r["type"]}
            for r in rows
        ]

    async def _check_degree_anomalies(self, doc_id: str | None) -> list[dict]:
        """Entities with degree far above the graph mean (potential hallucinated hubs)."""
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity)-[r:RELATES_TO]-()
            WITH e.name AS entity, count(r) AS degree
            WITH collect({entity: entity, degree: degree}) AS all_nodes,
                 avg(toFloat(degree))                      AS mean_degree
            UNWIND all_nodes AS node
            WITH node, mean_degree
            WHERE node.degree > mean_degree * $multiplier
            RETURN node.entity AS entity, node.degree AS degree, mean_degree
            LIMIT 20
            """,
            multiplier=MAX_DEGREE_MULTIPLIER,
        )
        return [
            {
                "type": "degree_anomaly",
                "entity": r["entity"],
                "degree": r["degree"],
                "mean_degree": round(r["mean_degree"], 1),
            }
            for r in rows
        ]

    async def _check_low_confidence_edges(self, doc_id: str | None) -> list[dict]:
        """Edges with suspiciously low confidence that may be hallucinations."""
        scope_clause = (
            "AND r.source_doc_id = $doc_id" if doc_id else ""
        )
        params: dict = {"threshold": MIN_CONFIDENCE}
        if doc_id:
            params["doc_id"] = doc_id

        rows = await self._neo4j.run(
            f"""
            MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
            WHERE r.confidence < $threshold {scope_clause}
            RETURN s.name AS src, t.name AS tgt,
                   r.relation AS relation, r.confidence AS confidence
            LIMIT 50
            """,
            **params,
        )
        return [
            {
                "type": "low_confidence_edge",
                "src": r["src"],
                "tgt": r["tgt"],
                "relation": r["relation"],
                "confidence": round(r["confidence"], 3),
            }
            for r in rows
        ]

    async def _check_relation_schema(self, doc_id: str | None) -> list[dict]:
        """Relations that violate the current ontology's allowed type pairs."""
        scope_clause = "AND r.source_doc_id = $doc_id" if doc_id else ""
        params: dict = {}
        if doc_id:
            params["doc_id"] = doc_id

        rows = await self._neo4j.run(
            f"""
            MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
            WHERE r.relation <> 'RELATED_TO' {scope_clause}
            RETURN s.name AS src, s.type AS src_type,
                   t.name AS tgt, t.type AS tgt_type,
                   r.relation AS relation
            LIMIT 200
            """,
            **params,
        )
        issues: list[dict] = []
        for row in rows:
            allowed_pairs = RELATION_RULES.get(row["relation"], set())
            if allowed_pairs and (row["src_type"], row["tgt_type"]) not in allowed_pairs:
                issues.append(
                    {
                        "type": "relation_schema_violation",
                        "src": row["src"],
                        "src_type": row["src_type"],
                        "tgt": row["tgt"],
                        "tgt_type": row["tgt_type"],
                        "relation": row["relation"],
                    }
                )
        return issues

    async def remove_self_loops(self) -> int:
        """Delete self-referencing edges. Returns count removed."""
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity)-[r:RELATES_TO]->(e)
            DELETE r
            RETURN count(r) AS removed
            """
        )
        removed = rows[0]["removed"] if rows else 0
        if removed:
            log.warning("ingestion_validator.self_loops_removed", count=removed)
        return removed
