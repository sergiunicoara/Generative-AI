"""Structural route/evidence scoring and runtime trace integration tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.routes.evaluation import score_retrieval_stages, score_retrieval_trajectory
from graphrag.core.models import RetrievalStep, RetrievalTrajectory
from graphrag.evaluation.domain_eval import validate_dataset
from graphrag.evaluation.retrieval_trajectory import (
    RetrievalTrajectoryEvaluationRequest,
    RetrievalTrajectoryExpectation,
    RetrievalStageEvaluationRequest,
    RetrievalStageExpectation,
    StageFailureCategory,
    assess_runtime_trajectory,
    evaluate_retrieval_stages,
    evaluate_retrieval_trajectory,
)
from graphrag.retrieval.agentic_retriever import AgenticRetriever


def test_scores_route_evidence_graph_and_efficiency() -> None:
    trajectory = RetrievalTrajectory(
        selected_surfaces=["text", "graph"],
        evidence_ids=["chunk-a", "chunk-extra"],
        graph_edges=["Elon Musk|FOUNDED|SpaceX"],
        tool_calls=2,
    )
    expected = RetrievalTrajectoryExpectation(
        expected_surfaces=["text", "graph"],
        expected_evidence_ids=["chunk-a"],
        expected_graph_edges=["Elon Musk|FOUNDED|SpaceX"],
        tool_budget=1,
    )

    score = evaluate_retrieval_trajectory(trajectory, expected, answer_score=0.8)

    assert score.route_f1 == 1.0
    assert score.evidence_precision == 0.5
    assert score.evidence_recall == 1.0
    assert score.graph_edge_recall == 1.0
    assert score.efficiency == 0.5
    assert score.aggregate == pytest.approx(0.83)


def test_unspecified_structural_dimensions_are_neutral() -> None:
    trajectory = RetrievalTrajectory(
        selected_surfaces=["vector"], evidence_ids=["chunk-a"], tool_calls=1,
    )
    expected = RetrievalTrajectoryExpectation(tool_budget=1)

    score = evaluate_retrieval_trajectory(trajectory, expected, answer_score=1.0)

    assert score.route_f1 == 1.0
    assert score.evidence_f1 == 1.0
    assert score.aggregate == 1.0


async def test_scoring_endpoint_uses_the_same_deterministic_contract() -> None:
    body = RetrievalTrajectoryEvaluationRequest(
        trajectory=RetrievalTrajectory(
            selected_surfaces=["graph"], evidence_ids=["chunk-a"], tool_calls=1,
        ),
        expected=RetrievalTrajectoryExpectation(
            expected_surfaces=["graph"], expected_evidence_ids=["chunk-a"], tool_budget=1,
        ),
        answer_score=1.0,
    )

    score = await score_retrieval_trajectory(body, _tenant="tenant-a")

    assert score.aggregate == 1.0


def test_stage_evaluator_attributes_an_observed_graph_failure() -> None:
    trajectory = RetrievalTrajectory(
        steps=[
            RetrievalStep(step=1, action="search", query="q", evidence_ids=["chunk-a"]),
            RetrievalStep(step=2, action="rerank", query="q", evidence_ids=["chunk-a"]),
        ],
        evidence_ids=["chunk-a"], graph_edges=["a|REL|b"],
    )
    result = evaluate_retrieval_stages(
        trajectory,
        RetrievalStageExpectation(
            expected_candidate_ids=["chunk-a"], expected_reranked_ids=["chunk-a"],
            expected_graph_edges=["a|REL|c"], expected_evidence_ids=["chunk-a"],
        ),
        citations=[], answer_score=1.0,
    )
    assert result.failure_category == StageFailureCategory.GRAPH_TRAVERSAL
    assert next(metric for metric in result.metrics if metric.stage == "graph_traversal").status == "failed"


def test_stage_evaluator_reports_uncertainty_for_untraced_expected_rerank() -> None:
    trajectory = RetrievalTrajectory(
        steps=[RetrievalStep(step=1, action="search", query="q", evidence_ids=["chunk-a"])],
        evidence_ids=["chunk-a"],
    )
    result = evaluate_retrieval_stages(
        trajectory, RetrievalStageExpectation(expected_reranked_ids=["chunk-a"]),
        citations=[], answer_score=1.0,
    )
    assert result.failure_category == StageFailureCategory.EVALUATOR_UNCERTAINTY


async def test_stage_scoring_endpoint_uses_same_contract() -> None:
    body = RetrievalStageEvaluationRequest(
        trajectory=RetrievalTrajectory(
            steps=[RetrievalStep(step=1, action="search", query="q", evidence_ids=["chunk-a"])],
            evidence_ids=["chunk-a"],
        ),
        expected=RetrievalStageExpectation(expected_candidate_ids=["chunk-a"]),
        answer_score=1.0,
    )
    result = await score_retrieval_stages(body, _tenant="tenant-a")
    assert result.failure_category is None


def test_runtime_stage_assessment_reports_missing_trace_as_uncertain() -> None:
    result = assess_runtime_trajectory(None, citations=[], faithfulness=1.0, temporal_query=False)
    assert result.failure_category == StageFailureCategory.EVALUATOR_UNCERTAINTY


def test_domain_eval_accepts_structural_expectations_and_rejects_bad_shapes() -> None:
    valid = {"tenant": "a", "questions": [{
        "id": "Q1",
        "type": "multi_hop",
        "question": "q",
        "expected_citations": [],
        "expected_surfaces": ["text", "graph"],
        "expected_evidence_ids": ["chunk-a"],
        "expected_graph_edges": ["a|REL|b"],
        "expected_candidate_ids": ["chunk-a"],
        "minimum_answer_score": 0.9,
        "tool_budget": 2,
    }]}
    assert validate_dataset(valid)["valid"] is True

    invalid = {**valid, "questions": [{**valid["questions"][0], "tool_budget": -1}]}
    result = validate_dataset(invalid)
    assert result["valid"] is False
    assert "tool_budget must be a non-negative integer" in result["errors"][0]


async def test_agentic_retriever_captures_seed_and_subsearch_evidence() -> None:
    retriever = AgenticRetriever.__new__(AgenticRetriever)
    retriever._local = AsyncMock()
    retriever._local.search = AsyncMock(side_effect=[
        {
            "chunks": [{"chunk_id": "seed", "text": "seed text"}],
            "referenced_chunks": ["seed"],
            "entities": [],
            "entity_edges": [{"src": "Elon Musk", "relation": "FOUNDED", "tgt": "SpaceX"}],
        },
        {
            "chunks": [{"chunk_id": "falcon", "text": "Falcon 9"}],
            "referenced_chunks": ["falcon"],
            "entities": [],
            "entity_edges": [{"src": "SpaceX", "relation": "LAUNCHED", "tgt": "Falcon 9"}],
        },
    ])
    retriever._ctx_builder = MagicMock()
    retriever._ctx_builder.build.return_value = ("context", ["doc-a"])
    retriever._max_steps = 2
    retriever._verifier = AsyncMock()
    retriever._reason = AsyncMock(side_effect=[
        "SEARCH: rockets launched by SpaceX",
        "ANSWER: SpaceX launched Falcon 9.",
    ])

    neo4j = MagicMock()
    neo4j.get_document_filenames = AsyncMock(return_value=[])
    with patch("graphrag.retrieval.agentic_retriever.get_neo4j", return_value=neo4j):
        result = await retriever.retrieve_and_answer("What rockets did Elon Musk's company launch?")

    trajectory = result.retrieval_trajectory
    assert trajectory is not None
    assert [step.action for step in trajectory.steps] == ["search", "sub_search", "answer"]
    assert trajectory.evidence_ids == ["seed", "falcon"]
    assert trajectory.graph_edges == [
        "Elon Musk|FOUNDED|SpaceX",
        "SpaceX|LAUNCHED|Falcon 9",
    ]
    assert trajectory.tool_calls == 2
