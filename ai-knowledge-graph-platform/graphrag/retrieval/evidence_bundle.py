"""Compact, serialisable evidence summary for retrieval and policy decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceBundle:
    chunk_ids: list[str]
    citation_ids: list[str]
    entity_ids: list[str]
    graph_edges: list[str]
    source_count: int
    path_count: int
    valid_at: str | None = None
    transaction_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_evidence_bundle(
    *,
    local_results: dict,
    global_results: dict,
    citations: list[str],
    valid_at: str | None = None,
    transaction_at: str | None = None,
) -> EvidenceBundle:
    """Summarise only identifiers and provenance already retrieved for a query."""
    chunks = local_results.get("chunks", [])
    chunk_ids = list(dict.fromkeys(str(c["chunk_id"]) for c in chunks if c.get("chunk_id")))
    entity_ids = list(dict.fromkeys(str(e) for e in local_results.get("referenced_entities", []) if e))
    graph_edges = []
    for edge in local_results.get("entity_edges", []) + local_results.get("document_link_edges", []):
        source, relation, target = edge.get("src"), edge.get("relation"), edge.get("tgt")
        if source and relation and target:
            graph_edges.append(f"{source}|{relation}|{target}")
    for community in global_results.get("communities", []):
        identifier = community.get("community_id") or community.get("id") or community.get("title")
        if identifier:
            graph_edges.append(f"community|contains|{identifier}")
    sources = {
        str(chunk.get("source") or chunk.get("_doc_name") or "")
        for chunk in chunks
        if chunk.get("source") or chunk.get("_doc_name")
    }
    return EvidenceBundle(
        chunk_ids=chunk_ids,
        citation_ids=list(dict.fromkeys(str(c) for c in citations if c)),
        entity_ids=entity_ids,
        graph_edges=list(dict.fromkeys(graph_edges)),
        source_count=len(sources) if sources else len(set(citations)),
        path_count=sum(1 for chunk in chunks if chunk.get("path_confidence") is not None),
        valid_at=valid_at,
        transaction_at=transaction_at,
    )


__all__ = ["EvidenceBundle", "build_evidence_bundle"]
