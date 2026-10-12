"""Offline graph data-completeness report (read-only, no LLM, no cost).

Distinct from ``correctness.py`` (is what is stored valid?) and from answer-quality evals (does the model answer
well?): this asks whether the stored graph is *populated* enough to serve retrieval: documents chunked and embedded,
chunks linked to entities, entities described / embedded / typed / connected, relations sourced and dated.

Each measure is a share with its numerator and denominator. A ratio is ``None`` when its denominator is zero. Scope is
one tenant, optionally one ``source_id``; quarantined entities and deleted documents are excluded.

    python scripts/graph_completeness.py --tenant aerospace [--source-id S]
"""
from __future__ import annotations

from typing import Any

from graphrag.graph.validation.graph_checks import run_read_only
from graphrag.graph.validity import edge_is_current

_CURRENT = edge_is_current("r")

DOCUMENTS = """
MATCH (d:Document {tenant: $tenant})
WHERE coalesce(d.is_deleted, false) = false AND ($source_id IS NULL OR d.source_id = $source_id)
WITH d, EXISTS { MATCH (c:Chunk {tenant: $tenant})-[:PART_OF]->(d) } AS has_chunks,
     EXISTS { MATCH (c:Chunk {tenant: $tenant})-[:PART_OF]->(d) MATCH (c)-[:MENTIONS]->(:Entity) } AS has_entities
RETURN count(d) AS documents,
       sum(CASE WHEN d.status = 'done' THEN 1 ELSE 0 END) AS ingested,
       sum(CASE WHEN has_chunks THEN 1 ELSE 0 END) AS with_chunks,
       sum(CASE WHEN has_entities THEN 1 ELSE 0 END) AS with_entities,
       sum(CASE WHEN coalesce(d.external_id, '') <> '' THEN 1 ELSE 0 END) AS with_external_id,
       sum(CASE WHEN coalesce(d.source_version, '') <> '' THEN 1 ELSE 0 END) AS with_version,
       sum(CASE WHEN d.ingested_at IS NOT NULL THEN 1 ELSE 0 END) AS with_ingested_at
"""

CHUNKS = """
MATCH (c:Chunk {tenant: $tenant})-[:PART_OF]->(d:Document {tenant: $tenant})
WHERE coalesce(d.is_deleted, false) = false AND ($source_id IS NULL OR d.source_id = $source_id)
RETURN count(c) AS chunks,
       sum(CASE WHEN c.embedding IS NOT NULL AND size(c.embedding) > 0 THEN 1 ELSE 0 END) AS embedded,
       sum(CASE WHEN coalesce(c.text, '') <> '' THEN 1 ELSE 0 END) AS with_text,
       sum(CASE WHEN EXISTS { MATCH (c)-[:MENTIONS]->(:Entity) } THEN 1 ELSE 0 END) AS with_entities
"""

ENTITIES = """
MATCH (e:Entity {tenant: $tenant})
WHERE coalesce(e.quarantined, false) = false
RETURN count(e) AS entities,
       sum(CASE WHEN coalesce(e.description, '') <> '' THEN 1 ELSE 0 END) AS with_description,
       sum(CASE WHEN e.embedding IS NOT NULL AND size(e.embedding) > 0 THEN 1 ELSE 0 END) AS embedded,
       sum(CASE WHEN coalesce(e.type, '') <> '' THEN 1 ELSE 0 END) AS typed,
       sum(CASE WHEN EXISTS { MATCH (:Chunk {tenant: $tenant})-[:MENTIONS]->(e) } THEN 1 ELSE 0 END) AS with_evidence,
       sum(CASE WHEN EXISTS { MATCH (e)-[:RELATES_TO]-(:Entity {tenant: $tenant}) } THEN 1 ELSE 0 END) AS connected
"""

RELATIONS = f"""
MATCH (:Entity {{tenant: $tenant}})-[r:RELATES_TO {{tenant: $tenant}}]->(:Entity {{tenant: $tenant}})
WHERE {_CURRENT}
RETURN count(r) AS relations,
       sum(CASE WHEN coalesce(r.relation, '') <> '' THEN 1 ELSE 0 END) AS with_relation,
       sum(CASE WHEN r.confidence IS NOT NULL THEN 1 ELSE 0 END) AS with_confidence,
       sum(CASE WHEN r.source_doc_id IS NOT NULL OR size(coalesce(r.source_doc_ids, [])) > 0
                THEN 1 ELSE 0 END) AS with_source,
       sum(CASE WHEN r.valid_from IS NOT NULL THEN 1 ELSE 0 END) AS with_valid_from
"""

QUERIES = {"documents": DOCUMENTS, "chunks": CHUNKS, "entities": ENTITIES, "relations": RELATIONS}

# (section, measure name, numerator key, denominator key)
MEASURES = (
    ("documents", "documents_ingested", "ingested", "documents"),
    ("documents", "documents_chunked", "with_chunks", "documents"),
    ("documents", "documents_with_entities", "with_entities", "documents"),
    ("documents", "documents_with_external_id", "with_external_id", "documents"),
    ("documents", "documents_with_version", "with_version", "documents"),
    ("chunks", "chunks_embedded", "embedded", "chunks"),
    ("chunks", "chunks_with_text", "with_text", "chunks"),
    ("chunks", "chunks_with_entities", "with_entities", "chunks"),
    ("entities", "entities_described", "with_description", "entities"),
    ("entities", "entities_embedded", "embedded", "entities"),
    ("entities", "entities_typed", "typed", "entities"),
    ("entities", "entities_with_evidence", "with_evidence", "entities"),
    ("entities", "entities_connected", "connected", "entities"),
    ("relations", "relations_named", "with_relation", "relations"),
    ("relations", "relations_with_confidence", "with_confidence", "relations"),
    ("relations", "relations_sourced", "with_source", "relations"),
    ("relations", "relations_dated", "with_valid_from", "relations"),
)


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def summarize(counts: dict[str, dict[str, Any]], *, incomplete_below: float = 0.95) -> dict[str, Any]:
    """Turn the four count rows into measures. ``incomplete`` lists measures under ``incomplete_below``, lowest first."""
    measures: dict[str, dict[str, Any]] = {}
    for section, name, num_key, den_key in MEASURES:
        row = counts.get(section) or {}
        num, den = int(row.get(num_key) or 0), int(row.get(den_key) or 0)
        measures[name] = {"share": _ratio(num, den), "numerator": num, "denominator": den}
    incomplete = sorted(
        ((m["share"], n) for n, m in measures.items() if m["share"] is not None and m["share"] < incomplete_below))
    return {
        "sizes": {s: int((counts.get(s) or {}).get(k) or 0)
                  for s, k in (("documents", "documents"), ("chunks", "chunks"),
                               ("entities", "entities"), ("relations", "relations"))},
        "measures": measures,
        "incomplete": [{"measure": n, "share": s} for s, n in incomplete],
        "threshold": incomplete_below,
        "note": "Population/completeness of the stored graph, not answer quality and not correctness. No LLM was used. "
                "A low share is a prompt to look: some fields are legitimately absent for some sources.",
    }


async def graph_completeness_report(neo4j, tenant: str, source_id: str | None = None) -> dict[str, Any]:
    counts: dict[str, dict[str, Any]] = {}
    for section, cypher in QUERIES.items():
        rows = await run_read_only(neo4j, cypher, tenant=tenant, source_id=source_id, as_of=None)
        counts[section] = dict(rows[0]) if rows else {}
    report = summarize(counts)
    report.update(tenant=tenant, source_id=source_id)
    return report


__all__ = ["MEASURES", "QUERIES", "graph_completeness_report", "summarize"]
