"""Provenance and metering behavior around queued evaluation jobs."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag.agents.evaluation_agent import EvaluationAgent
from graphrag.core.models import EvalJob, EvalResult, QueryResult


async def test_evaluation_preserves_job_tenant_correlation_and_stable_kpi_event_id():
    agent = EvaluationAgent.__new__(EvaluationAgent)
    agent._evaluator = MagicMock()
    agent._evaluator.evaluate_single = AsyncMock(return_value=EvalResult(
        job_id="query-7", query_id="query-7", faithfulness=0.9, answer_relevancy=0.8,
        context_precision=0.7, context_recall=0.6,
    ))
    agent._tracker = MagicMock()
    agent._tracker.record = AsyncMock()
    job = EvalJob(
        job_id="evaluation-7", tenant="automotive", correlation_id="corr-evaluation-7",
        ground_truth="a",
        query_result=QueryResult(query_id="query-7", question="q", answer="a", contexts=["ctx"], latency_ms=12.0),
    )
    calibration = MagicMock()
    calibration.add_sample = AsyncMock()

    neo4j = MagicMock()
    neo4j.run = AsyncMock(return_value=[])
    with (
        patch("graphrag.agents.evaluation_agent.record_evaluation_job") as record_metric,
        patch("graphrag.agents.evaluation_agent.record_rubric_results") as record_rubrics,
        patch("graphrag.graph.neo4j_client.get_neo4j", return_value=neo4j),
        patch("graphrag.graph.confidence_calibration.CalibrationService", return_value=calibration),
    ):
        result = await agent.run(job)

    assert result.job_id == "evaluation-7"
    kpi = agent._tracker.record.await_args.args[0]
    assert kpi.event_id == "evaluation-7"
    calibration.add_sample.assert_not_awaited()
    assert record_metric.call_args.kwargs["outcome"] == "completed"
    assert record_metric.call_args.kwargs["tenant"] == "automotive"
    assert record_rubrics.call_args.kwargs["tenant"] == "automotive"
    assert neo4j.run.await_count == 7
    kpi = agent._tracker.record.await_args.args[0]
    assert kpi.judge_decision == "accept"
    assert kpi.evaluation_source == "reference_judge"
    assert result.failure_category == "evaluator_uncertainty"
    assert result.failure_reason == "retrieval trajectory was not captured"


@pytest.mark.asyncio
async def test_evaluation_policy_retrieves_ragas_then_accepts_finite_score():
    agent = EvaluationAgent.__new__(EvaluationAgent)
    agent._evaluator = MagicMock()
    agent._evaluator.evaluate_single = AsyncMock(return_value=EvalResult(
        job_id="q-8", query_id="q-8", faithfulness=0.86,
    ))
    job = EvalJob(
        job_id="evaluation-8", tenant="aerospace", ground_truth="",
        query_result=QueryResult(query_id="q-8", question="q", answer="answer", contexts=["ctx"]),
    )

    result, policy = await agent._evaluate_with_policy(job)

    assert policy is not None
    assert policy.retrieval_used is True
    assert result.judge_decision == "accept"
    assert result.judge_confidence == 0.86
    assert result.evaluation_source == "ragas"
    assert result.retrieval_used is True
    agent._evaluator.evaluate_single.assert_awaited_once()


@pytest.mark.asyncio
async def test_evaluation_policy_abstains_on_refusal_without_ragas_call():
    agent = EvaluationAgent.__new__(EvaluationAgent)
    agent._evaluator = MagicMock()
    agent._evaluator.evaluate_single = AsyncMock()
    job = EvalJob(
        job_id="evaluation-9", tenant="aerospace", ground_truth="",
        query_result=QueryResult(
            query_id="q-9", question="q", answer="I cannot answer from the available context.",
        ),
    )

    result, policy = await agent._evaluate_with_policy(job)

    assert policy is not None
    assert result.judge_decision == "abstain"
    assert result.evaluation_source == "judge"
    assert result.retrieval_used is False
    agent._evaluator.evaluate_single.assert_not_awaited()


async def test_explicit_authorization_denial_overrides_semantic_acceptance():
    agent = EvaluationAgent.__new__(EvaluationAgent)
    job = EvalJob(
        job_id="evaluation-denied", tenant="aerospace",
        query_result=QueryResult(
            query_id="q-denied", question="q", answer="supported answer",
            citations=["source"], contexts=["source"], policy_result="escalate",
            policy_reason_code="no_authorized_evidence",
        ),
    )
    result = EvalResult(job_id="evaluation-denied", query_id="q-denied", faithfulness=1.0,
                        judge_decision="accept")

    gated = agent._apply_deterministic_rubrics(job, result)

    assert gated.rubric_hard_failed
    assert gated.judge_decision == "abstain"
    assert gated.abstention_reason == "mandatory_rubric_failed"
