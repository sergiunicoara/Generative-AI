"""Deterministic evaluation for multi-surface retrieval trajectories.

The aggregate follows WorkSurface-Bench's published category weights while
adapting its route/evidence concepts to this project's observable trace model.
It is intentionally independent of an LLM judge: answer quality is supplied by
the existing evaluation pipeline and structural correctness is scored here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from graphrag.core.models import RetrievalTrajectory


class RetrievalTrajectoryExpectation(BaseModel):
    expected_surfaces: list[str] = Field(default_factory=list)
    expected_evidence_ids: list[str] = Field(default_factory=list)
    expected_graph_edges: list[str] = Field(default_factory=list)
    tool_budget: int = Field(default=1, ge=0)


class RetrievalTrajectoryScore(BaseModel):
    answer_score: float
    route_precision: float
    route_recall: float
    route_f1: float
    evidence_precision: float
    evidence_recall: float
    evidence_f1: float
    graph_edge_recall: float
    structural_evidence_score: float
    efficiency: float
    aggregate: float


class RetrievalTrajectoryEvaluationRequest(BaseModel):
    trajectory: RetrievalTrajectory
    expected: RetrievalTrajectoryExpectation
    answer_score: float = Field(ge=0.0, le=1.0)


def _prf(observed: set[str], expected: set[str]) -> tuple[float, float, float]:
    if not expected:
        # This dimension was not specified by the golden case, so it is
        # outside the case's contract rather than a false positive.
        return 1.0, 1.0, 1.0
    true_positive = len(observed & expected)
    precision = true_positive / len(observed) if observed else 0.0
    recall = true_positive / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_retrieval_trajectory(
    trajectory: RetrievalTrajectory,
    expected: RetrievalTrajectoryExpectation,
    *,
    answer_score: float,
) -> RetrievalTrajectoryScore:
    """Score route selection, evidence capture and tool-call efficiency."""
    route_p, route_r, route_f1 = _prf(
        set(trajectory.selected_surfaces), set(expected.expected_surfaces),
    )
    evidence_p, evidence_r, evidence_f1 = _prf(
        set(trajectory.evidence_ids), set(expected.expected_evidence_ids),
    )
    expected_edges = set(expected.expected_graph_edges)
    graph_recall = (
        len(set(trajectory.graph_edges) & expected_edges) / len(expected_edges)
        if expected_edges else 1.0
    )
    structural_evidence = (
        (evidence_f1 + graph_recall) / 2 if expected_edges else evidence_f1
    )
    calls = trajectory.tool_calls
    budget = expected.tool_budget
    efficiency = 1.0 if calls <= budget else (budget / calls if calls else 1.0)
    aggregate = round((
        0.35 * answer_score
        + 0.30 * structural_evidence
        + 0.25 * route_f1
        + 0.10 * efficiency
    ), 6)
    return RetrievalTrajectoryScore(
        answer_score=answer_score,
        route_precision=route_p,
        route_recall=route_r,
        route_f1=route_f1,
        evidence_precision=evidence_p,
        evidence_recall=evidence_r,
        evidence_f1=evidence_f1,
        graph_edge_recall=graph_recall,
        structural_evidence_score=structural_evidence,
        efficiency=efficiency,
        aggregate=aggregate,
    )
