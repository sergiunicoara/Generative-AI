"""Combine local + global search results with configurable weights and re-ranking."""

from __future__ import annotations

import asyncio
import time

from google import genai
import structlog

from graphrag.core.config import get_settings
from graphrag.core.models import QueryResult
from graphrag.retrieval.local_search import LocalSearch
from graphrag.retrieval.global_search import GlobalSearch
from graphrag.retrieval.context_builder import ContextBuilder
from graphrag.retrieval.agentic_retriever import AgenticRetriever, _is_low_confidence

log = structlog.get_logger(__name__)

_ANSWER_PROMPT = """\
You are an expert assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, say so explicitly.
Always cite your sources by chunk ID.

Context:
{context}

Question: {question}

Answer:"""


class HybridRetriever:
    def __init__(self):
        cfg = get_settings()
        self._client = genai.Client(api_key=cfg.google_api_key)
        self._model_name = cfg.gemini_query_model
        self._cfg = cfg.retrieval
        self._local = LocalSearch()
        self._global = GlobalSearch()
        self._context_builder = ContextBuilder()
        self._model_version = cfg.gemini_query_model
        self._agentic = AgenticRetriever(
            max_steps=self._cfg.get("agentic_max_steps", 4)
        )

    async def retrieve_and_answer(
        self,
        question: str,
        mode: str = "hybrid",
    ) -> QueryResult:
        t0 = time.monotonic()

        local_results = {}
        global_results = {}

        if mode in ("local", "hybrid"):
            local_results = await self._local.search(question)

        if mode in ("global", "hybrid"):
            global_results = await self._global.search(question)

        context, citations = self._context_builder.build(
            local_results=local_results,
            global_results=global_results,
            weights=(
                self._cfg.get("hybrid_weight_local", 0.6),
                self._cfg.get("hybrid_weight_global", 0.4),
            ),
            top_k=self._cfg.get("rerank_top_k", 5),
        )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=self._model_name,
                contents=_ANSWER_PROMPT.format(context=context, question=question),
            ),
        )
        answer = response.text.strip()
        latency_ms = (time.monotonic() - t0) * 1000

        # ── Agentic fallback ───────────────────────────────────────────────────
        # If the hybrid answer is low-confidence, hand off to the iterative
        # agent which re-searches sub-questions until it accumulates enough
        # context to answer confidently (solves multi-document reasoning).
        agentic_enabled = self._cfg.get("agentic_fallback", True)
        if agentic_enabled and _is_low_confidence(answer, citations):
            log.info(
                "hybrid_retriever.low_confidence",
                answer_preview=answer[:80],
                triggering="agentic_fallback",
            )
            result = await self._agentic.retrieve_and_answer(
                question=question,
                initial_context=context,
                initial_citations=citations,
            )
            result.latency_ms += latency_ms
            return result

        log.info("hybrid_retriever.done", mode=mode, latency_ms=round(latency_ms, 1))

        return QueryResult(
            question=question,
            answer=answer,
            contexts=[c["text"] for c in local_results.get("chunks", [])],
            citations=citations,
            latency_ms=latency_ms,
            retrieval_mode=mode,
            model_version=self._model_version,
        )
