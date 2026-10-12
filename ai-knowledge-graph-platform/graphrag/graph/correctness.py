"""Offline graph-correctness report (docs/IMPLEMENTATION_AUDIT.md "graph correctness metrics").

Read-only structural checks over one tenant's published relations. These are *evaluation* measures, not
production metrics: the retrieval queries already exclude retracted / expired / quarantined facts by
construction, so a live gauge of them would be trivially 1.0. What this report catches is damage in the
stored graph itself (inverted validity windows, edges that look current but cite a superseded document,
inferred edges with no recorded premises, document edges with no source).

No LLM, no network beyond Neo4j, no writes: both queries pass ``assert_read_only`` and run in a READ session
when the client offers ``run_read``.

    python scripts/graph_correctness.py --tenant aerospace
"""
from __future__ import annotations

from typing import Any

from graphrag.graph.validation.graph_checks import run_read_only
from graphrag.graph.validity import edge_is_current

_CURRENT = edge_is_current("r")

EDGE_COUNTS = f"""
MATCH (s:Entity {{tenant: $tenant}})-[r:RELATES_TO {{tenant: $tenant}}]->(t:Entity {{tenant: $tenant}})
WITH r, s, t, ({_CURRENT}) AS is_current,
     (coalesce(r.origin, '') = 'INFERRED' OR coalesce(r.source_type, '') = 'inferred') AS is_inferred
RETURN count(r) AS edges,
       sum(CASE WHEN is_current THEN 1 ELSE 0 END) AS current_edges,
       sum(CASE WHEN r.valid_from IS NOT NULL AND r.valid_to IS NOT NULL AND r.valid_to < r.valid_from
                THEN 1 ELSE 0 END) AS inverted_validity,
       sum(CASE WHEN is_current AND (coalesce(s.quarantined, false) OR coalesce(t.quarantined, false))
                THEN 1 ELSE 0 END) AS current_edges_on_quarantined,
       sum(CASE WHEN is_inferred THEN 1 ELSE 0 END) AS inferred_edges,
       sum(CASE WHEN is_inferred AND size(coalesce(r.premise_keys, [])) = 0 THEN 1 ELSE 0 END)
           AS inferred_without_premises,
       sum(CASE WHEN NOT is_inferred AND coalesce(r.source_type, 'document') = 'document'
                     AND r.source_doc_id IS NULL AND size(coalesce(r.source_doc_ids, [])) = 0
                THEN 1 ELSE 0 END) AS document_edges_without_source
"""

SUPERSEDED_SOURCE_EDGES = f"""
MATCH (:Entity {{tenant: $tenant}})-[r:RELATES_TO {{tenant: $tenant}}]->(:Entity {{tenant: $tenant}})
WHERE ({_CURRENT}) AND r.source_doc_id IS NOT NULL
  AND EXISTS {{ MATCH (d:Document {{tenant: $tenant, id: r.source_doc_id}}) WHERE d.superseded_by IS NOT NULL }}
RETURN count(r) AS superseded_source_edges
"""


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize(counts: dict[str, int], superseded_source_edges: int = 0) -> dict[str, Any]:
    """Turn raw counts into the report. A ratio is ``None`` when its denominator is zero."""
    edges = int(counts.get("edges") or 0)
    current = int(counts.get("current_edges") or 0)
    inferred = int(counts.get("inferred_edges") or 0)
    inverted = int(counts.get("inverted_validity") or 0)
    no_premises = int(counts.get("inferred_without_premises") or 0)
    no_source = int(counts.get("document_edges_without_source") or 0)
    on_quarantined = int(counts.get("current_edges_on_quarantined") or 0)
    superseded = int(superseded_source_edges or 0)
    return {
        "edges": edges,
        "current_edges": current,
        "measures": {
            # share of stored edges that a retrieval at "now" may use
            "current_edge_share": _ratio(current, edges),
            # share of edges whose validity window is not inverted (valid_to before valid_from)
            "temporal_consistency": _ratio(edges - inverted, edges),
            # share of current edges that do NOT cite a superseded document (superseded-fact usage = 1 - this)
            "current_edges_from_current_documents": _ratio(current - superseded, current),
            # share of inferred edges with recorded premises (derivation can be replayed)
            "inference_traceability": _ratio(inferred - no_premises, inferred),
            # share of document-derived edges that record a source document
            "provenance_coverage": _ratio(edges - no_source, edges),
        },
        "violations": {
            "inverted_validity": inverted,
            "superseded_source_edges": superseded,
            "current_edges_on_quarantined_entities": on_quarantined,
            "inferred_without_premises": no_premises,
            "document_edges_without_source": no_source,
        },
        "note": "Structural checks on the stored graph; not answer-quality measures. No LLM was used.",
    }


async def graph_correctness_report(neo4j, tenant: str) -> dict[str, Any]:
    rows = await run_read_only(neo4j, EDGE_COUNTS, tenant=tenant, as_of=None)
    superseded = await run_read_only(neo4j, SUPERSEDED_SOURCE_EDGES, tenant=tenant, as_of=None)
    report = summarize(dict(rows[0]) if rows else {},
                       (superseded[0].get("superseded_source_edges") if superseded else 0) or 0)
    report["tenant"] = tenant
    return report


__all__ = ["EDGE_COUNTS", "SUPERSEDED_SOURCE_EDGES", "graph_correctness_report", "summarize"]
