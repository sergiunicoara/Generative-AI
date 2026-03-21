"""Google ADK agent that runs RAGAS evaluation and logs KPIs."""

from __future__ import annotations

import structlog

from graphrag.agents.base_agent import BaseGraphRAGAgent
from graphrag.business_matrix.kpi_tracker import KPITracker
from graphrag.core.config import get_settings
from graphrag.core.models import EvalJob, EvalResult, KPIEvent
from graphrag.evaluation.ragas_evaluator import RagasEvaluator

log = structlog.get_logger(__name__)


class EvaluationAgent(BaseGraphRAGAgent):
    def __init__(self):
        self._evaluator = RagasEvaluator()
        self._tracker = KPITracker()
        super().__init__("evaluation_agent")

    def _model(self) -> str:
        return get_settings().gemini_ingest_model  # flash is fine for eval

    def _instruction(self) -> str:
        return (
            "You are an evaluation agent. Given a completed query turn, "
            "run RAGAS metrics (faithfulness, answer_relevancy, context_precision, "
            "context_recall) and log all KPIs to the business matrix store."
        )

    def _tools(self) -> list:
        try:
            from google.adk.tools import FunctionTool
            from graphrag.agents.tools.evaluation_tools import score_answer, log_kpi
            return [FunctionTool(score_answer), FunctionTool(log_kpi)]
        except ImportError:
            return []

    async def run(self, job: EvalJob) -> EvalResult:
        log.info("evaluation_agent.start", job_id=job.job_id)

        qr = job.query_result
        eval_result = await self._evaluator.evaluate_single(
            query_id=qr.query_id,
            question=qr.question,
            answer=qr.answer,
            contexts=qr.contexts,
            ground_truth=job.ground_truth,
        )

        kpi = KPIEvent(
            query_id=qr.query_id,
            latency_ms=qr.latency_ms,
            faithfulness=eval_result.faithfulness,
            answer_relevancy=eval_result.answer_relevancy,
            context_precision=eval_result.context_precision,
            context_recall=eval_result.context_recall,
            retrieval_mode=qr.retrieval_mode,
            model_version=qr.model_version,
        )
        await self._tracker.record(kpi)

        log.info(
            "evaluation_agent.done",
            job_id=job.job_id,
            faithfulness=round(eval_result.faithfulness, 3),
            answer_relevancy=round(eval_result.answer_relevancy, 3),
        )
        return eval_result
