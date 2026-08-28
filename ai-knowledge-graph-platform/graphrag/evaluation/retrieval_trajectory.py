"""Deterministic evaluation for multi-surface retrieval trajectories.

The aggregate follows WorkSurface-Bench's published category weights while
adapting its route/evidence concepts to this project's observable trace model.
It is intentionally independent of an LLM judge: answer quality is supplied by
the existing evaluation pipeline and structural correctness is scored here.
"""

from __future__ import annotations

from enum import StrEnum

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


class StageFailureCategory(StrEnum):
    ENTITY_RESOLUTION = "entity_resolution_failure"
    CANDIDATE_GENERATION = "candidate_generation_failure"
    RERANKING = "reranking_failure"
    GRAPH_TRAVERSAL = "graph_traversal_failure"
    FRESHNESS = "freshness_failure"
    CONTEXT_SELECTION = "context_selection_failure"
    GENERATION = "generation_failure"
    CITATION = "citation_failure"
    EVALUATOR_UNCERTAINTY = "evaluator_uncertainty"


class RetrievalStageExpectation(BaseModel):
    """Optional gold expectations for independently observable pipeline stages."""

    expected_entity_ids: list[str] = Field(default_factory=list)
    expected_candidate_ids: list[str] = Field(default_factory=list)
    expected_reranked_ids: list[str] = Field(default_factory=list)
    expected_graph_edges: list[str] = Field(default_factory=list)
    expected_fresh_evidence_ids: list[str] = Field(default_factory=list)
    expected_evidence_ids: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    minimum_answer_score: float = Field(default=0.8, ge=0.0, le=1.0)


class RetrievalStageMetric(BaseModel):
    stage: str
    status: str  # passed | failed | unobserved | not_evaluated
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    f1: float | None = Field(default=None, ge=0.0, le=1.0)
    observed_count: int = 0
    expected_count: int = 0


class RetrievalStageEvaluationResult(BaseModel):
    metrics: list[RetrievalStageMetric] = Field(default_factory=list)
    failure_category: StageFailureCategory | None = None
    failure_reason: str = ""


class RetrievalStageEvaluationRequest(BaseModel):
    trajectory: RetrievalTrajectory
    expected: RetrievalStageExpectation
    citations: list[str] = Field(default_factory=list)
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


def _stage_metric(stage: str, observed: list[str], expected: list[str], *, observed_stage: bool = True) -> RetrievalStageMetric:
    if not expected:
        return RetrievalStageMetric(stage=stage, status="not_evaluated", observed_count=len(observed))
    if not observed_stage:
        return RetrievalStageMetric(stage=stage, status="unobserved", observed_count=len(observed), expected_count=len(expected))
    precision, recall, f1 = _prf(set(observed), set(expected))
    return RetrievalStageMetric(
        stage=stage, status="passed" if recall == 1.0 else "failed",
        precision=precision, recall=recall, f1=f1,
        observed_count=len(observed), expected_count=len(expected),
    )


def _step_values(trajectory: RetrievalTrajectory, actions: set[str], field: str) -> tuple[list[str], bool]:
    matching = [step for step in trajectory.steps if step.action.casefold() in actions]
    values = [str(value) for step in matching for value in getattr(step, field) if value]
    return list(dict.fromkeys(values)), bool(matching)


def evaluate_retrieval_stages(
    trajectory: RetrievalTrajectory,
    expected: RetrievalStageExpectation,
    *, citations: list[str], answer_score: float,
) -> RetrievalStageEvaluationResult:
    """Score each stage without inferring an unrecorded stage's outcome."""
    entities, entity_observed = _step_values(trajectory, {"entity_resolution", "entity_resolve"}, "evidence_ids")
    candidates, candidate_observed = _step_values(trajectory, {"search", "sub_search", "global_search", "candidate_generation"}, "evidence_ids")
    reranked, rerank_observed = _step_values(trajectory, {"rerank", "cross_encoder"}, "evidence_ids")
    fresh, freshness_observed = _step_values(trajectory, {"freshness", "freshness_filter"}, "evidence_ids")
    metrics = [
        _stage_metric("entity_resolution", entities, expected.expected_entity_ids, observed_stage=entity_observed),
        _stage_metric("candidate_generation", candidates, expected.expected_candidate_ids, observed_stage=candidate_observed),
        _stage_metric("reranking", reranked, expected.expected_reranked_ids, observed_stage=rerank_observed),
        _stage_metric("graph_traversal", trajectory.graph_edges, expected.expected_graph_edges, observed_stage=bool(trajectory.steps)),
        _stage_metric("freshness_filtering", fresh, expected.expected_fresh_evidence_ids, observed_stage=freshness_observed),
        _stage_metric("evidence_selection", trajectory.evidence_ids, expected.expected_evidence_ids, observed_stage=bool(trajectory.steps)),
        _stage_metric("citations", citations, expected.expected_citations),
        RetrievalStageMetric(
            stage="generation", status="passed" if answer_score >= expected.minimum_answer_score else "failed",
            precision=answer_score, recall=answer_score, f1=answer_score, observed_count=1, expected_count=1,
        ),
    ]
    stage_categories = {
        "entity_resolution": StageFailureCategory.ENTITY_RESOLUTION,
        "candidate_generation": StageFailureCategory.CANDIDATE_GENERATION,
        "reranking": StageFailureCategory.RERANKING,
        "graph_traversal": StageFailureCategory.GRAPH_TRAVERSAL,
        "freshness_filtering": StageFailureCategory.FRESHNESS,
        "evidence_selection": StageFailureCategory.CONTEXT_SELECTION,
        "generation": StageFailureCategory.GENERATION,
        "citations": StageFailureCategory.CITATION,
    }
    for metric in metrics:
        if metric.status == "unobserved":
            return RetrievalStageEvaluationResult(
                metrics=metrics, failure_category=StageFailureCategory.EVALUATOR_UNCERTAINTY,
                failure_reason=f"{metric.stage} was expected but not captured in the trajectory",
            )
        if metric.status == "failed":
            return RetrievalStageEvaluationResult(
                metrics=metrics, failure_category=stage_categories[metric.stage],
                failure_reason=f"{metric.stage} did not meet its expectation",
            )
    return RetrievalStageEvaluationResult(metrics=metrics)


def assess_runtime_trajectory(
    trajectory: RetrievalTrajectory | None,
    *, citations: list[str], faithfulness: float, temporal_query: bool,
    answer_support_threshold: float = 0.8,
) -> RetrievalStageEvaluationResult:
    """Produce durable diagnostics for normal evaluations that lack gold stage IDs."""
    if trajectory is None:
        return RetrievalStageEvaluationResult(
            failure_category=StageFailureCategory.EVALUATOR_UNCERTAINTY,
            failure_reason="retrieval trajectory was not captured",
        )
    expected = RetrievalStageExpectation(
        expected_candidate_ids=list(trajectory.evidence_ids) or ["__evidence_required__"],
        expected_evidence_ids=list(trajectory.evidence_ids) or ["__evidence_required__"],
        expected_fresh_evidence_ids=["__freshness_trace_required__"] if temporal_query else [],
        expected_citations=list(citations) or ["__citation_required__"],
        minimum_answer_score=answer_support_threshold,
    )
    if temporal_query:
        fresh, captured = _step_values(trajectory, {"freshness", "freshness_filter"}, "evidence_ids")
        if captured and fresh:
            expected.expected_fresh_evidence_ids = fresh
    return evaluate_retrieval_stages(trajectory, expected, citations=citations, answer_score=faithfulness)


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
