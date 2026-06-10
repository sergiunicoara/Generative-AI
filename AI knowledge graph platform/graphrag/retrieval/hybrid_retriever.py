"""Combine local + global search results with configurable weights and re-ranking."""

from __future__ import annotations

import asyncio
import time

import structlog

from graphrag.core.config import get_settings
from graphrag.core.llm_client import get_llm
from graphrag.core.models import QueryResult
from graphrag.retrieval.local_search import LocalSearch
from graphrag.retrieval.global_search import GlobalSearch
from graphrag.retrieval.context_builder import ContextBuilder
from graphrag.retrieval.agentic_retriever import AgenticRetriever, _is_low_confidence
from graphrag.retrieval.claim_verifier import ClaimVerifier
from graphrag.retrieval.session_context import get_session_context
from graphrag.core.llm_utils import safe_response_text

log = structlog.get_logger(__name__)

_ANSWER_PROMPT = """\
You are a regulatory knowledge assistant. Answer using ONLY the information in the context below.
Rules:
- Use ONLY facts stated in the context. Do NOT add information from your training data.
- If a fact is not in the context, do not include it in your answer.
- If the context does not contain enough information to answer, say so explicitly.
- Be concise: 3-5 sentences unless the question requires more.
- State facts directly. Do NOT preface your answer with phrases like "Based on the context", \
"Based solely on the context", "According to the provided context", or similar.

Context:
{context}

Question: {question}

Answer:"""


class HybridRetriever:
    def __init__(self):
        cfg = get_settings()
        self._model_name = cfg.groq_model
        self._cfg = cfg.retrieval
        self._local = LocalSearch()
        self._global = GlobalSearch()
        self._context_builder = ContextBuilder()
        self._model_version = cfg.groq_model
        self._agentic = AgenticRetriever(
            max_steps=self._cfg.get("agentic_max_steps", 4)
        )
        self._verifier = ClaimVerifier()
        self._use_session_ctx = self._cfg.get("session_context_enabled", True)
        self._session_ctx = get_session_context() if self._use_session_ctx else None

    async def retrieve_and_answer(
        self,
        question: str,
        mode: str = "hybrid",
        tenant: str = "default",
        session_id: str = "",
    ) -> QueryResult:
        t0 = time.monotonic()

        local_results = {}
        global_results = {}

        if mode in ("local", "hybrid"):
            local_results = await self._local.search(
                question,
                session_id=session_id,
                tenant=tenant,
            )

        if mode in ("global", "hybrid"):
            global_results = await self._global.search(question, tenant=tenant)

        context, citations = self._context_builder.build(
            local_results=local_results,
            global_results=global_results,
            weights=(
                self._cfg.get("hybrid_weight_local", 0.6),
                self._cfg.get("hybrid_weight_global", 0.4),
            ),
            top_k=self._cfg.get("rerank_top_k", 5),
        )

        answer = await get_llm().generate(
            _ANSWER_PROMPT.format(context=context, question=question),
        ) or "Insufficient context to answer this question."

        # ── Claim verification — strip ungrounded sentences ────────────────────
        if self._cfg.get("claim_verification", False):
            answer, n_removed = await self._verifier.verify(answer, context)
            if n_removed:
                log.info("hybrid_retriever.claims_stripped", n_removed=n_removed)

        latency_ms = (time.monotonic() - t0) * 1000

        # ── Record session turn with the real answer ───────────────────────────
        # Done here (not in local_search) so the stored turn always reflects the
        # actual answer shown to the user, making follow-up enrichment faithful.
        if self._use_session_ctx and self._session_ctx and session_id and local_results:
            await self._session_ctx.record_turn(
                session_id=session_id,
                question=question,
                answer=answer,
                referenced_entities=local_results.get("referenced_entities", []),
                referenced_chunks=local_results.get("referenced_chunks", []),
            )

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
                tenant=tenant,
                session_id=session_id,
            )
            result.latency_ms += latency_ms
            return result

        log.info("hybrid_retriever.done", mode=mode, latency_ms=round(latency_ms, 1))

        return QueryResult(
            question=question,
            answer=answer,
            # `context` is the full string fed to the synthesis LLM (local chunks +
            # entity context + global community knowledge). Using only local chunks
            # here caused RAGAS to judge claims grounded in "Community knowledge"
            # as unsupported (faithfulness=0.0 false negatives, e.g. AUT-03).
            contexts=[context] if context else [],
            citations=citations,
            latency_ms=latency_ms,
            retrieval_mode=mode,
            model_version=self._model_version,
        )
