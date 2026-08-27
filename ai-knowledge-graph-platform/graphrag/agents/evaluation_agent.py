"""Agent that runs RAGAS evaluation and logs KPIs.

After each evaluation the (context_precision → faithfulness) pair is written
to the CalibrationService so the dashboard Calibration tab reflects real data
without a separate manual step.
"""

from __future__ import annotations

import asyncio
import math

import time

import structlog

from graphrag.agents.base_agent import BaseGraphRAGAgent
from graphrag.business_matrix.kpi_tracker import KPITracker
from graphrag.core.config import get_settings
from graphrag.core.models import EvalJob, EvalResult, KPIEvent
from graphrag.evaluation.factory import build_evaluator
from graphrag.evaluation.judge_retrieve_abstain import (
    CalibrationThresholds,
    JudgeDecision,
    JudgeRetrieveAbstainResult,
    BigQueryJudgeMetricsSink,
    finalize_after_retrieval,
    judge_without_retrieval,
)
from graphrag.evidence.claim_graph import (
    build_claim_evidence_graph,
    persist_claim_evidence_graph,
)
from graphrag.observability.agent_telemetry import record_evaluation_job, record_evaluation_quality
from graphrag.observability.correlation import correlation_context
from graphrag.observability.tracing import trace_span

log = structlog.get_logger(__name__)


class EvaluationAgent(BaseGraphRAGAgent):
    def __init__(self):
        self._evaluator = build_evaluator()
        self._tracker = KPITracker()
        super().__init__("evaluation_agent")

    def _model(self) -> str:
        return get_settings().groq_model  # Groq: provenance stamping for ADK scaffold

    def _instruction(self) -> str:
        return (
            "You are an evaluation agent. Given a completed query turn, "
            "run configured evidence metrics (RAGAS when available, otherwise the "
            "deterministic reference evaluator) and log all KPIs to the business matrix store."
        )

    def _judge_thresholds(self) -> CalibrationThresholds:
        cfg = get_settings().evaluation.get("judge_retrieve_abstain", {})
        return CalibrationThresholds(
            accept_threshold=float(cfg.get("accept_threshold", 0.90)),
            retrieve_threshold=float(cfg.get("retrieve_threshold", 0.55)),
            target_fdr=float(cfg.get("target_fdr", 0.05)),
        )

    async def _evaluate_with_policy(
        self, job: EvalJob,
    ) -> tuple[EvalResult, JudgeRetrieveAbstainResult | None]:
        """Run the cheap judge first, then pay for RAGAS only when needed."""
        cfg = get_settings().evaluation.get("judge_retrieve_abstain", {})
        thresholds = self._judge_thresholds()
        qr = job.query_result
        if not cfg.get("enabled", True):
            return await self._evaluator.evaluate_single(
                query_id=qr.query_id, question=qr.question, answer=qr.answer,
                contexts=qr.contexts, ground_truth=job.ground_truth,
            ), None

        initial = judge_without_retrieval(
            answer=qr.answer, reference=job.ground_truth, thresholds=thresholds,
        )
        if initial.decision == JudgeDecision.ABSTAIN:
            result = EvalResult(
                job_id=qr.query_id, query_id=qr.query_id,
                judge_decision=initial.decision.value,
                judge_confidence=initial.confidence,
                judge_accept_threshold=thresholds.accept_threshold,
                judge_retrieve_threshold=thresholds.retrieve_threshold,
                judge_target_fdr=thresholds.target_fdr,
                retrieval_used=False, abstention_reason=initial.abstention_reason,
                evaluation_source="judge",
            )
            return result, initial
        if initial.decision == JudgeDecision.ACCEPT:
            result = EvalResult(
                job_id=qr.query_id, query_id=qr.query_id,
                faithfulness=initial.confidence,
                judge_decision=initial.decision.value,
                judge_confidence=initial.confidence,
                judge_accept_threshold=thresholds.accept_threshold,
                judge_retrieve_threshold=thresholds.retrieve_threshold,
                judge_target_fdr=thresholds.target_fdr,
                retrieval_used=False, evaluation_source="reference_judge",
            )
            return result, initial

        ragas_result = await self._evaluator.evaluate_single(
            query_id=qr.query_id, question=qr.question, answer=qr.answer,
            contexts=qr.contexts, ground_truth=job.ground_truth,
        )
        retrieval_threshold = float(cfg.get("retrieval_accept_threshold", 0.80))
        retrieval_policy = CalibrationThresholds(
            accept_threshold=retrieval_threshold,
            retrieve_threshold=thresholds.retrieve_threshold,
            target_fdr=thresholds.target_fdr,
        )
        final = finalize_after_retrieval(
            initial, ragas_result.faithfulness,
            accept_threshold=retrieval_policy.accept_threshold,
        )
        result = ragas_result.model_copy(update={
            "judge_decision": final.decision.value,
            "judge_confidence": final.confidence,
            "judge_accept_threshold": retrieval_policy.accept_threshold,
            "judge_retrieve_threshold": retrieval_policy.retrieve_threshold,
            "judge_target_fdr": retrieval_policy.target_fdr,
            "retrieval_used": True,
            "abstention_reason": final.abstention_reason,
            "evaluation_source": ragas_result.evaluation_source,
        })
        return result, final

    async def run(self, job: EvalJob) -> EvalResult:
        started_at = time.monotonic()
        outcome = "failed"
        log.info("evaluation_agent.start", job_id=job.job_id, tenant=job.tenant)

        try:
            with correlation_context(job.correlation_id), trace_span(
                "evaluation.run", job_id=job.job_id, query_id=job.query_result.query_id,
                tenant=job.tenant,
                correlation_id=job.correlation_id or job.query_result.correlation_id,
            ) as evaluation_span:
                qr = job.query_result
                eval_result, policy_result = await self._evaluate_with_policy(job)
                # Evaluators naturally key a score by query; preserve the
                # durable queue job ID as well so retries and traces have one
                # unambiguous evaluation identity.
                eval_result.job_id = job.job_id

                # A stable event ID makes a redelivered job visible and gives
                # the KPI backend a deterministic key for deduplication.
                kpi = KPIEvent(
                    event_id=job.job_id,
                    query_id=qr.query_id,
                    tenant=job.tenant,
                    latency_ms=qr.latency_ms,
                    faithfulness=eval_result.faithfulness,
                    answer_relevancy=eval_result.answer_relevancy,
                    context_precision=eval_result.context_precision,
                    context_recall=eval_result.context_recall,
                    retrieval_mode=qr.retrieval_mode,
                    model_version=qr.model_version,
                    judge_decision=eval_result.judge_decision,
                    judge_confidence=eval_result.judge_confidence,
                    judge_accept_threshold=eval_result.judge_accept_threshold,
                    judge_retrieve_threshold=eval_result.judge_retrieve_threshold,
                    judge_target_fdr=eval_result.judge_target_fdr,
                    retrieval_used=eval_result.retrieval_used,
                    abstention_reason=eval_result.abstention_reason,
                    evaluation_source=eval_result.evaluation_source,
                )
                await self._tracker.record(kpi)
                record_evaluation_quality(
                    faithfulness=eval_result.faithfulness,
                    source=eval_result.evaluation_source,
                )

                # Persist an auditable claim/artifact/action/check subgraph.
                # Evaluation must remain available if Neo4j or an optional
                # metrics sink is temporarily unavailable.
                try:
                    graph = build_claim_evidence_graph(qr, eval_result, tenant=job.tenant)
                    from graphrag.graph.neo4j_client import get_neo4j
                    await persist_claim_evidence_graph(get_neo4j(), graph)
                    if evaluation_span is not None:
                        evaluation_span.set_attribute("claim_count", len(graph.claims))
                        evaluation_span.set_attribute("artifact_count", len(graph.artifacts))
                        evaluation_span.set_attribute("judge.decision", eval_result.judge_decision)
                        evaluation_span.set_attribute("judge.retrieval_used", eval_result.retrieval_used)
                        if qr.source_trace_id:
                            evaluation_span.set_attribute("source_trace_id", qr.source_trace_id)
                except Exception as exc:
                    log.warning("evaluation.claim_graph_persist_failed", error=str(exc)[:200])

                metrics_table = get_settings().evaluation.get(
                    "judge_retrieve_abstain", {}
                ).get("bigquery_metrics_table", "")
                if metrics_table and policy_result is not None:
                    try:
                        row = policy_result.metrics(tenant=job.tenant, query_id=qr.query_id)
                        await asyncio.to_thread(BigQueryJudgeMetricsSink(metrics_table).write, row)
                    except Exception as exc:
                        log.warning("evaluation.judge_metrics_sink_failed", error=str(exc)[:200])

        # ── Wire calibration sample ────────────────────────────────────────────
        # predicted_confidence = context_precision (how confident the retrieval was)
        # actual_outcome       = faithfulness      (how correct the answer was)
        # This populates the dashboard Calibration tab automatically after each run.
                try:
                    if eval_result.evaluation_source not in {"ragas", "reference"}:
                        raise ValueError("calibration samples require retrieval-backed evaluation")
                    if not all(
                        isinstance(value, (int, float)) and math.isfinite(value)
                        for value in (eval_result.context_precision, eval_result.faithfulness)
                    ):
                        raise ValueError("calibration metrics are unavailable or non-finite")
                    from graphrag.graph.confidence_calibration import CalibrationService
                    from graphrag.graph.neo4j_client import get_neo4j
                    cal_svc = CalibrationService(get_neo4j())
                    await cal_svc.add_sample(
                        predicted_confidence=eval_result.context_precision,
                        actual_outcome=eval_result.faithfulness,
                        relation=qr.retrieval_mode,
                        source_doc_id=qr.query_id,
                        prompt_version=qr.model_version,
                        tenant=job.tenant,
                        verified_by=eval_result.evaluation_source,
                    )
                    log.debug("evaluation_agent.calibration_sample_added", tenant=job.tenant)
                except Exception as exc:
                    # Calibration is downstream learning data, never a reason
                    # to drop an otherwise measured evaluation result.
                    log.warning("evaluation_agent.calibration_sample_failed", error=str(exc))

            outcome = "completed"
            log.info(
                "evaluation_agent.done",
                job_id=job.job_id,
                faithfulness=round(eval_result.faithfulness, 3),
                answer_relevancy=round(eval_result.answer_relevancy, 3),
            )
            return eval_result
        finally:
            record_evaluation_job(
                outcome=outcome, tenant=job.tenant, job_id=job.job_id, started_at=started_at,
            )
