"""Configurable text, graph, path, and provenance fusion for evidence chunks."""

from __future__ import annotations

from dataclasses import dataclass

from graphrag.retrieval.query_planner import classify_query


@dataclass(frozen=True)
class EvidenceFusionWeights:
    text: float
    graph: float
    path: float
    provenance: float

    def normalized(self) -> "EvidenceFusionWeights":
        values = [max(0.0, self.text), max(0.0, self.graph), max(0.0, self.path), max(0.0, self.provenance)]
        total = sum(values)
        if total <= 0:
            return EvidenceFusionWeights(1.0, 0.0, 0.0, 0.0)
        return EvidenceFusionWeights(*(value / total for value in values))


_DEFAULTS = {
    "factoid": EvidenceFusionWeights(0.70, 0.10, 0.05, 0.15),
    "relational": EvidenceFusionWeights(0.35, 0.30, 0.25, 0.10),
    "contradiction": EvidenceFusionWeights(0.35, 0.20, 0.20, 0.25),
    "multi_hop": EvidenceFusionWeights(0.25, 0.25, 0.40, 0.10),
    "negative": EvidenceFusionWeights(0.75, 0.05, 0.05, 0.15),
}


def fusion_weights(question: str, overrides: dict | None = None) -> EvidenceFusionWeights:
    """Return normalised per-query-class weights, optionally calibrated by config."""
    query_class = classify_query(question)
    values = dict((overrides or {}).get(query_class, {}))
    base = _DEFAULTS[query_class]
    return EvidenceFusionWeights(
        text=float(values.get("text", base.text)),
        graph=float(values.get("graph", base.graph)),
        path=float(values.get("path", base.path)),
        provenance=float(values.get("provenance", base.provenance)),
    ).normalized()


def apply_evidence_fusion(chunks: list[dict], weights: EvidenceFusionWeights) -> list[dict]:
    """Score retrieved chunks using only already-computed, bounded evidence signals."""
    weights = weights.normalized()
    for chunk in chunks:
        text = _unit_score(chunk.get("text_score", chunk.get("final_score", chunk.get("score", 0.0))))
        graph = _unit_score(chunk.get("gnn_score", 0.0))
        path = _unit_score(chunk.get("path_confidence", 0.0))
        provenance = _provenance_score(chunk)
        chunk["fusion_components"] = {
            "text": text, "graph": graph, "path": path, "provenance": provenance,
        }
        chunk["final_score"] = (
            weights.text * text + weights.graph * graph
            + weights.path * path + weights.provenance * provenance
        )
    chunks.sort(key=lambda item: item["final_score"], reverse=True)
    return chunks


def _unit_score(value: object) -> float:
    try:
        return min(1.0, max(0.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _provenance_score(chunk: dict) -> float:
    if chunk.get("source_type") == "inferred":
        return 0.60
    if chunk.get("document_link"):
        return 0.80
    return _unit_score(chunk.get("authority_weight", 1.0))


__all__ = ["EvidenceFusionWeights", "apply_evidence_fusion", "fusion_weights"]
