"""RAGAS evaluation wrapper — faithfulness, answer_relevancy, context_precision, context_recall."""

from __future__ import annotations

import asyncio

import structlog
from datasets import Dataset

from graphrag.core.config import get_settings
from graphrag.core.exceptions import EvaluationError
from graphrag.core.models import EvalResult

log = structlog.get_logger(__name__)


class RagasEvaluator:
    def __init__(self):
        cfg = get_settings()
        self._metrics_cfg = cfg.evaluation.get(
            "ragas_metrics",
            ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        )
        self._llm = self._build_llm(cfg.gemini_ingest_model)

    def _build_llm(self, model_name: str):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from graphrag.core.config import get_settings
            cfg = get_settings()
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=cfg.google_api_key,
            )
        except ImportError:
            log.warning("ragas_evaluator.langchain_genai_missing")
            return None

    def _build_metrics(self):
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        return [metric_map[m] for m in self._metrics_cfg if m in metric_map]

    async def evaluate_single(
        self,
        query_id: str,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = "",
    ) -> EvalResult:
        if not contexts:
            log.warning("ragas_evaluator.no_contexts", query_id=query_id)
            return EvalResult(job_id=query_id, query_id=query_id)

        dataset = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth or ""],
            }
        )

        metrics = self._build_metrics()

        loop = asyncio.get_event_loop()
        try:
            from ragas import evaluate

            scores = await loop.run_in_executor(
                None,
                lambda: evaluate(dataset, metrics=metrics, llm=self._llm),
            )
            result_dict = scores.to_pandas().iloc[0].to_dict()
        except Exception as exc:
            log.error("ragas_evaluator.error", error=str(exc), query_id=query_id)
            raise EvaluationError(str(exc)) from exc

        return EvalResult(
            job_id=query_id,
            query_id=query_id,
            faithfulness=float(result_dict.get("faithfulness", 0.0)),
            answer_relevancy=float(result_dict.get("answer_relevancy", 0.0)),
            context_precision=float(result_dict.get("context_precision", 0.0)),
            context_recall=float(result_dict.get("context_recall", 0.0)),
        )
