"""Low-overhead construction helpers for retrieval trajectory telemetry."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graphrag.core.models import RetrievalStep, RetrievalTrajectory


def _unique(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value not in (None, "")))


def evidence_ids(results: dict) -> list[str]:
    """Return stable chunk/evidence IDs from a local retrieval result."""
    referenced = results.get("referenced_chunks") or []
    if referenced:
        return _unique(referenced)
    return _unique(chunk.get("chunk_id") for chunk in results.get("chunks", []))


def graph_edge_ids(results: dict) -> list[str]:
    """Canonicalize graph edges without retaining source text or tenant data."""
    edge_ids: list[str] = []
    for edge in results.get("entity_edges", []):
        source = edge.get("src") or edge.get("source")
        target = edge.get("tgt") or edge.get("target")
        relation = edge.get("relation") or "RELATES_TO"
        if source and target:
            edge_ids.append(f"{source}|{relation}|{target}")
    return _unique(edge_ids)


def surfaces_for_mode(
    mode: str,
    *,
    has_global: bool = False,
    text_enabled: bool = True,
    vector_enabled: bool = True,
    graph_enabled: bool = True,
) -> list[str]:
    """Map runtime retrieval modes to benchmarkable evidence surfaces."""
    surfaces: list[str] = []
    if mode in {"local", "hybrid", "agentic"}:
        if text_enabled:
            surfaces.append("text")
        if vector_enabled:
            surfaces.append("vector")
        if graph_enabled:
            surfaces.append("graph")
    if mode in {"global", "hybrid"} or has_global:
        surfaces.append("community")
    return _unique(surfaces)


def trajectory_from_steps(
    *,
    query_class: str,
    planned_mode: str,
    routing_reason: str,
    steps: list[RetrievalStep],
    completed_by: str,
) -> RetrievalTrajectory:
    """Aggregate ordered steps into a bounded query-level trajectory."""
    return RetrievalTrajectory(
        query_class=query_class,
        planned_mode=planned_mode,
        routing_reason=routing_reason,
        steps=steps,
        selected_surfaces=_unique(surface for step in steps for surface in step.surfaces),
        evidence_ids=_unique(item for step in steps for item in step.evidence_ids),
        graph_edges=_unique(item for step in steps for item in step.graph_edges),
        tool_calls=sum(step.action in {"search", "sub_search", "global_search"} for step in steps),
        completed_by=completed_by,
    )
