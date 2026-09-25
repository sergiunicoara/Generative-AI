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

from graphrag.core.tenancy import require_tenant
from graphrag.graph.ontology_registry import _RELATION_RULES as RELATION_RULES

log = structlog.get_logger(__name__)

# Anomaly thresholds
# For sparse domain graphs (< 10k entities), hub entities naturally have
# 20–50× mean degree (e.g. FAA, Boeing in a regulatory corpus). A multiplier
# of 5 would quarantine every important entity. 20 catches genuine poisoning
# (a hallucinated node linked to everything) while leaving real hubs alone.
MAX_DEGREE_MULTIPLIER  = 20.0  # flag if degree > mean * this
MIN_CONFIDENCE         = 0.1   # flag edges with suspiciously low confidence
MAX_ORPHAN_RATE        = 0.10  # flag if > 10% of new entities are orphans
class IngestionValidator:
    """
    Post-ingestion graph health checker.

    Usage::

        validator = IngestionValidator(neo4j_client)
        report = await validator.validate(tenant="acme", doc_id="doc_abc")
        if report["issues"]:
            log.warning("ingestion_issues", **report)
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    async def validate(self, tenant: str, doc_id: str | None = None) -> dict:
        """
        Run all checks within `tenant`. If doc_id is given, scopes checks
        further to that document's entities only. Returns a structured report.

        `tenant` was previously not a parameter anywhere in this module --
        every check ran across the whole graph regardless of caller, so a
        poisoning signal in one tenant's data could be reported (and, for
        _check_degree_anomalies, computed) using every other tenant's graph
        mixed in.
        """
        tenant = require_tenant(tenant)
        issues: list[dict] = []

        issues += await self._check_self_loops(tenant, doc_id)
        issues += await self._check_orphan_entities(tenant, doc_id)
        issues += await self._check_degree_anomalies(tenant, doc_id)
        issues += await self._check_low_confidence_edges(tenant, doc_id)
        issues += await self._check_relation_schema(tenant, doc_id)

        report = {
            "doc_id": doc_id,
            "tenant": tenant,
            "total_issues": len(issues),
            "issues": issues,
        }

        if issues:
            log.warning(
                "ingestion_validator.issues_found",
                doc_id=doc_id,
                tenant=tenant,
                count=len(issues),
                types=list({i["type"] for i in issues}),
            )
        else:
            log.info("ingestion_validator.clean", doc_id=doc_id, tenant=tenant)

        return report

    # ── Individual checks ──────────────────────────────────────────────────────

    async def _check_self_loops(self, tenant: str, doc_id: str | None) -> list[dict]:
        """Entities that relate to themselves."""
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity {tenant: $tenant})-[r:RELATES_TO]->(e)
            RETURN e.name AS entity, r.relation AS relation
            LIMIT 50
            """,
            tenant=tenant,
        )
        return [
            {"type": "self_loop", "entity": r["entity"], "relation": r["relation"]}
            for r in rows
        ]

    async def _check_orphan_entities(self, tenant: str, doc_id: str | None) -> list[dict]:
        """Entities with no MENTIONS link to any chunk."""
        # The scoped and unscoped cases need structurally different queries, so
        # both are written out below. An earlier f-string `query` was built here
        # from a `scope`/`params` pair and then never executed — dead scaffolding
        # that read as if it were the query actually being run.
        if doc_id:
            rows = await self._neo4j.run(
                """
                MATCH (c:Chunk {document_id: $doc_id, tenant: $tenant})-[:MENTIONS]->(e:Entity {tenant: $tenant})
                WHERE NOT EXISTS {
                    MATCH (e)<-[:MENTIONS]-(:Chunk)
                }
                RETURN e.name AS entity, e.type AS type
                LIMIT 100
                """,
                doc_id=doc_id,
                tenant=tenant,
            )
        else:
            rows = await self._neo4j.run(
                """
                MATCH (e:Entity {tenant: $tenant})
                WHERE NOT (e)<-[:MENTIONS]-(:Chunk)
                RETURN e.name AS entity, e.type AS type
                LIMIT 100
                """,
                tenant=tenant,
            )
        return [
            {"type": "orphan_entity", "entity": r["entity"], "entity_type": r["type"]}
            for r in rows
        ]

    async def _check_degree_anomalies(self, tenant: str, doc_id: str | None) -> list[dict]:
        """Entities with degree far above the graph mean (potential hallucinated hubs).

        Was unscoped -- both the anomaly candidates AND the mean_degree
        baseline they're compared against were computed across every
        tenant's graph combined, so a small tenant sharing a deployment
        with a much larger one could have its hub entities flagged (or its
        real anomalies masked) by a baseline that has nothing to do with
        its own graph.
        """
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity {tenant: $tenant})-[r:RELATES_TO {tenant: $tenant}]-()
            WITH e.name AS entity, e.type AS entity_type, count(r) AS degree
            WITH collect({entity: entity, entity_type: entity_type, degree: degree}) AS all_nodes,
                 avg(toFloat(degree))                      AS mean_degree
            UNWIND all_nodes AS node
            WITH node, mean_degree
            WHERE node.degree > mean_degree * $multiplier
            RETURN node.entity AS entity, node.entity_type AS entity_type, node.degree AS degree, mean_degree
            LIMIT 20
            """,
            tenant=tenant,
            multiplier=MAX_DEGREE_MULTIPLIER,
        )
        return [
            {
                "type": "degree_anomaly",
                "entity": r["entity"],
                "entity_type": r["entity_type"],
                "degree": r["degree"],
                "mean_degree": round(r["mean_degree"], 1),
            }
            for r in rows
        ]

    async def _check_low_confidence_edges(self, tenant: str, doc_id: str | None) -> list[dict]:
        """Edges with suspiciously low confidence that may be hallucinations."""
        scope_clause = (
            "AND r.source_doc_id = $doc_id" if doc_id else ""
        )
        params: dict = {"threshold": MIN_CONFIDENCE, "tenant": tenant}
        if doc_id:
            params["doc_id"] = doc_id

        rows = await self._neo4j.run(
            f"""
            MATCH (s:Entity {{tenant: $tenant}})-[r:RELATES_TO {{tenant: $tenant}}]->(t:Entity {{tenant: $tenant}})
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

    async def _check_relation_schema(self, tenant: str, doc_id: str | None) -> list[dict]:
        """Relations that violate the current ontology's allowed type pairs."""
        scope_clause = "AND r.source_doc_id = $doc_id" if doc_id else ""
        params: dict = {"tenant": tenant}
        if doc_id:
            params["doc_id"] = doc_id

        rows = await self._neo4j.run(
            f"""
            MATCH (s:Entity {{tenant: $tenant}})-[r:RELATES_TO {{tenant: $tenant}}]->(t:Entity {{tenant: $tenant}})
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

    async def remove_self_loops(self, tenant: str) -> int:
        """Delete self-referencing edges within `tenant` only.

        Was unscoped -- every ingestion run's cleanup pass deleted
        self-loop RELATES_TO edges across EVERY tenant's graph, not just
        the one being ingested.
        """
        tenant = require_tenant(tenant)
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity {tenant: $tenant})-[r:RELATES_TO {tenant: $tenant}]->(e)
            DELETE r
            RETURN count(r) AS removed
            """,
            tenant=tenant,
        )
        removed = rows[0]["removed"] if rows else 0
        if removed:
            log.warning("ingestion_validator.self_loops_removed", count=removed, tenant=tenant)
        return removed
