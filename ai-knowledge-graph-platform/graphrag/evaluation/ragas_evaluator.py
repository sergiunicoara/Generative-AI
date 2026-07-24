"""RAGAS evaluation wrapper — faithfulness, answer_relevancy, context_precision, context_recall.

The RAGAS judge LLM is resolved in priority order:
  1. DeepSeek-V3 (``langchain-openai`` + DeepSeek base URL) — generous rate limits, fast failover
  2. Groq (``langchain-groq``) — consistent with the generation pipeline, free-tier quota
  3. Gemini (``langchain-google-genai``) — last resort if both above are unavailable
  4. None — RAGAS will use its internal default LLM (may fail without API keys)

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
        """Build the LLM judge for RAGAS.

        Priority: Groq → DeepSeek → Gemini.

        Groq is the primary judge because its 100k TPD resets daily at midnight UTC
        and the faithfulness eval (~35k tokens) fits well within that budget.
        DeepSeek is the fallback when Groq is rate-limited; Gemini is the last resort.
        """
        cfg = get_settings()

        # 1st choice: DeepSeek — no daily token cap (paid subscription)
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model="deepseek-v4-pro",   # was "deepseek-chat" — deprecated, API rejects it
                api_key=cfg.deepseek_api_key,
                base_url="https://api.deepseek.com",
                temperature=0.0,
            )
            log.info("ragas_evaluator.llm_deepseek", model="deepseek-v4-pro")
            return llm
        except ImportError:
            log.debug("ragas_evaluator.langchain_openai_missing — trying Groq")
        except Exception as exc:
            log.warning("ragas_evaluator.deepseek_init_failed", error=str(exc)[:120])

        # 2nd choice: Groq — 100k TPD, resets daily at midnight UTC
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
            log.debug("ragas_evaluator.langchain_groq_missing — trying Gemini")
        except Exception as exc:
            log.warning("ragas_evaluator.groq_init_failed", error=str(exc)[:120])

        # 3rd choice: Gemini via langchain-google-genai
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
                note="install langchain-openai, langchain-groq, or langchain-google-genai",
            )
        except Exception as exc:
            log.warning("ragas_evaluator.gemini_init_failed", error=str(exc)[:120])

        return None

    @staticmethod
    def _patch_langchain_vertexai_stub():
        """Stub out langchain_community's VertexAI classes.

        ragas==0.4.x unconditionally imports ChatVertexAI/VertexAI from
        langchain_community, but langchain-community>=0.4 dropped that
        submodule entirely. This project never configures VertexAI, so a
        minimal stub satisfies ragas's import without pulling in a
        conflicting langchain-community version.
        """
        import sys
        import types

        try:
            import langchain_community.chat_models.vertexai  # noqa: F401

            return
        except ModuleNotFoundError:
            pass

        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.language_models.llms import LLM

        class ChatVertexAI(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "vertexai-stub"

            def _generate(self, *args, **kwargs):
                raise NotImplementedError("VertexAI is not configured for this project")

        class VertexAI(LLM):
            @property
            def _llm_type(self) -> str:
                return "vertexai-stub"

            def _call(self, *args, **kwargs):
                raise NotImplementedError("VertexAI is not configured for this project")

        vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")
        vertexai_module.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = vertexai_module

        import langchain_community.llms as llms_module

        if not hasattr(llms_module, "VertexAI"):
            llms_module.VertexAI = VertexAI

    def _build_metrics(self):
        self._patch_langchain_vertexai_stub()
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

            # Only pass FakeEmbeddings when no metric needs real embeddings.
            # answer_relevancy computes cosine similarity between generated and original
            # question — it needs real embeddings, not random vectors.
            _embedding_metrics = {"answer_relevancy"}
            _needs_real_embeddings = any(
                m in self._metrics_cfg for m in _embedding_metrics
            )
            if _needs_real_embeddings:
                _eval_embeddings = None  # let RAGAS use its default (OpenAI)
            else:
                try:
                    from langchain_community.embeddings import FakeEmbeddings
                    _eval_embeddings = FakeEmbeddings(size=1)
                except Exception:
                    _eval_embeddings = None

            scores = await loop.run_in_executor(
                None,
                lambda: evaluate(
                    dataset,
                    metrics=metrics,
                    llm=self._llm,
                    embeddings=_eval_embeddings,
                ),
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
