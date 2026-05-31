"""RAGAS evaluation wrapper — faithfulness, answer_relevancy, context_precision, context_recall.

The RAGAS judge LLM is resolved in priority order:
  1. Groq (``langchain-groq``) — consistent with the generation pipeline, free-tier quota
  2. Google Generative AI (``langchain-google-genai``) — fallback if langchain-groq absent
  3. None — RAGAS will use its internal default LLM (may fail without API keys)

Install Groq support: ``pip install langchain-groq``
"""

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
        self._llm = self._build_llm()

    def _build_llm(self):
        """Build the LLM judge for RAGAS. Tries Groq first, falls back to Gemini."""
        cfg = get_settings()

        # 1st choice: Groq — consistent with generation pipeline, avoids Gemini quota
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model=cfg.groq_model,
                api_key=cfg.groq_api_key,
                temperature=0.0,
            )
            log.info("ragas_evaluator.llm_groq", model=cfg.groq_model)
            return llm
        except ImportError:
            log.debug("ragas_evaluator.langchain_groq_missing — trying Gemini fallback")
        except Exception as exc:
            log.warning("ragas_evaluator.groq_init_failed", error=str(exc))

        # 2nd choice: Gemini via langchain-google-genai
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=cfg.gemini_ingest_model,
                google_api_key=cfg.google_api_key,
            )
            log.info("ragas_evaluator.llm_gemini", model=cfg.gemini_ingest_model)
            return llm
        except ImportError:
            log.warning(
                "ragas_evaluator.no_langchain_llm",
                note="install langchain-groq or langchain-google-genai for RAGAS evaluation",
            )

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

        loop = asyncio.get_running_loop()
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
