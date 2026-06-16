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
- When a document or procedure has a revision/version number (e.g. "rev.2", "revizia 2", "v.2"), \
state it using the compact document-ID form by removing the dot/space, e.g. "rev.2" -> "rev2". \
Apply this compact form to EVERY revision number you mention, every time you mention it — \
including when you restate the same number later in the answer.
- If a question asks which revision is REFERENCED by one document and whether it matches the \
CURRENT/IN-FORCE revision of another, your answer MUST explicitly state BOTH revision numbers \
in compact form (e.g. "rev2" and "rev4") and explicitly say whether they match or not — do not \
describe the mismatch only in words ("an older revision") without naming both numbers.
- A "=== METADATA ===" block contains a "doc_id" line identifying that chunk's source document \
and revision (e.g. "doc_id: IL-INS-03-rev4"). Treat this as a fact about which revision exists \
when the question concerns document revisions.
- A chunk header may include "Source: <filename>" identifying which document that chunk came \
from. This is for attribution and revision-comparison only — do NOT refuse to use a fact merely \
because its chunk's Source differs from a document named in the question. Use all relevant facts \
from the context to answer fully.
- If the question names specific documents (e.g. "conform X și Y"), and the context contains a \
fact from a chunk whose Source is NOT one of those documents, prefer a fact from a chunk whose \
Source IS one of the named documents when the two conflict.
- A "Community knowledge:" section is a coarse, lower-precision summary. If it conflicts with a \
specific fact stated in a numbered "[Chunk ...]" section above it, the chunk-level fact is more \
reliable — prefer it.

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
        query_id: str = "",
    ) -> QueryResult:
        t0 = time.monotonic()

        from graphrag.retrieval.result_store import get_result_store
        _store = get_result_store() if query_id else None

        async def _step(msg: str):
            if _store and query_id:
                await _store.push_progress(query_id, msg)

        local_results = {}
        global_results = {}

        if mode in ("local", "hybrid"):
            await _step("🔍 Căutare BM25 + vector în graf...")
            await _step("🕸️ Scorare GNN — traversare 2 hop-uri...")
            local_results = await self._local.search(
                question,
                session_id=session_id,
                tenant=tenant,
            )
            n_reranked = self._cfg.get("rerank_top_k", 5)
            await _step(f"📊 Reranking cross-encoder → top {n_reranked} fragmente")

        if mode in ("global", "hybrid"):
            await _step("🕸️ Expansiune graf (comunități Leiden)...")
            global_results = await self._global.search(question, tenant=tenant)

        await _step("✍️ Sinteză răspuns cu LLM...")
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
