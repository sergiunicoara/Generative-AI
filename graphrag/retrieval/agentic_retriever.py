"""
Agentic Retriever — iterative IRCoT-style fallback.

Flow:
  1. Initial hybrid search
  2. LLM reasons: "What entity/concept do I still need?"
  3. Re-search on that entity
  4. Repeat up to `max_steps`
  5. Synthesize final answer from all accumulated context
"""

from __future__ import annotations

import time

import structlog

from graphrag.core.config import get_settings
from graphrag.core.llm_client import get_fast_llm, get_llm
from graphrag.core.llm_utils import normalize_dashes
from graphrag.core.models import QueryResult
from graphrag.retrieval.local_search import LocalSearch
from graphrag.retrieval.context_builder import ContextBuilder
from graphrag.retrieval.claim_verifier import ClaimVerifier
from graphrag.retrieval.fallback_policy import is_low_confidence as _is_low_confidence  # noqa: F401

log = structlog.get_logger(__name__)

_REASONING_PROMPT = """\
You are a research assistant doing iterative retrieval.

Question: {question}

Context gathered so far:
{context}

Based on this context, answer one of:
A) If you can already answer the question fully, respond with:
   ANSWER: <your complete answer with citations>

B) If you need more information, respond with:
   SEARCH: <specific entity, concept, or sub-question to look up next>

Be concise. Do not explain."""

_FINAL_PROMPT = """\
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
If the context also mentions a different revision of the same document (e.g. "rev.4"), name that \
revision too (e.g. "rev4") and state whether it matches the one referenced.
- A "=== METADATA ===" block contains a "doc_id" line identifying that chunk's source document \
and revision (e.g. "doc_id: IL-INS-03-rev4"). Treat this as a fact about which revision exists \
when the question concerns document revisions.

Context:
{context}

Question: {question}

Answer:"""

class AgenticRetriever:
    """
    Iterative retrieval agent — searches, reasons, re-searches until confident.
    Used as fallback when HybridRetriever returns a low-confidence answer.

    Two-model design for latency:
    - Reasoning steps (SEARCH/ANSWER decisions): llama-3.1-8b-instant (~0.2s each)
    - Final synthesis: llama-3.3-70b-versatile (~1.5s, full quality)

    With max_steps=2 this yields: ~0.5s retrieval + 2×0.2s reasoning + ~1.5s synthesis
    = ~2.4s total vs the previous ~6s with 70B for every step.
    """

    def __init__(self, max_steps: int = 2):
        self._local = LocalSearch()
        # NOTE: unlike HybridRetriever, this class doesn't thread per-tenant
        # config through to ContextBuilder.build() calls below (both are
        # hardcoded top_k), so context_hop_reserved_slots (see
        # context_builder.py) has no effect on this agentic-fallback path --
        # only on the primary hybrid/local path. Defaults to 0 (no-op)
        # either way, so this is a known scope gap, not a behavior change.
        self._ctx_builder = ContextBuilder()
        self._max_steps = max_steps
        self._verifier = ClaimVerifier()

    async def _reason(self, prompt: str) -> str:
        """Fast 8B model for cheap SEARCH/ANSWER routing decisions."""
        return await get_fast_llm().generate(prompt)

    async def _synthesize(self, prompt: str) -> str:
        """Full 70B model for final answer synthesis."""
        # See llm_utils.normalize_dashes — same fix as HybridRetriever's
        # main synthesis path, applied here too since this is also a
        # final, user-facing/graded answer, not an intermediate step.
        return normalize_dashes(await get_llm().generate(prompt))

    async def retrieve_and_answer(
        self,
        question: str,
        initial_context: str = "",
        initial_citations: list[str] | None = None,
        tenant: str = "default",
        session_id: str = "",
    ) -> QueryResult:
        t0 = time.monotonic()

        all_chunks: list[dict] = []
        all_citations: list[str] = list(initial_citations or [])
        context_sections: list[str] = []

        if initial_context:
            context_sections.append(initial_context)

        # Initial search on the original question
        seed_results = await self._local.search(
            question,
            session_id=session_id,
            tenant=tenant,
        )
        seed_chunks = seed_results.get("chunks", [])
        all_chunks.extend(seed_chunks)

        ctx, cits = self._ctx_builder.build(
            local_results=seed_results,
            global_results={},
            top_k=5,
        )
        if ctx:
            context_sections.append(ctx)
            all_citations.extend(cits)

        log.info("agentic_retriever.start", question=question, max_steps=self._max_steps)

        # Iterative reasoning loop — 8B fast model for SEARCH/ANSWER decisions
        for step in range(self._max_steps):
            current_context = "\n\n---\n\n".join(context_sections)

            reasoning = await self._reason(
                _REASONING_PROMPT.format(
                    question=question,
                    context=current_context or "(no context yet)",
                )
            )

            log.info(
                "agentic_retriever.step",
                step=step + 1,
                reasoning_prefix=reasoning[:120],
            )

            if reasoning.upper().startswith("ANSWER:"):
                # LLM is confident — extract and verify the answer
                # normalize_dashes here too: this early-exit shortcut uses
                # _reason()/get_fast_llm(), a separate model call from
                # _synthesize()/get_llm() below — missed on the first pass
                # of this fix (2026-08-17) because it's a distinct code path
                # that live-verification against AGT-02/CON-01/AUT-01/AUT-03/
                # PRE-02 caught: all five were taking THIS shortcut, not the
                # _synthesize() path already fixed.
                answer = normalize_dashes(reasoning[7:].strip())
                current_context = "\n\n---\n\n".join(context_sections)
                cfg = get_settings().retrieval
                if cfg.get("claim_verification", False):
                    answer, n_removed = await self._verifier.verify(answer, current_context)
                    if n_removed:
                        log.info("agentic_retriever.claims_stripped", n_removed=n_removed)
                latency_ms = (time.monotonic() - t0) * 1000
                log.info(
                    "agentic_retriever.done",
                    steps=step + 1,
                    latency_ms=round(latency_ms, 1),
                    mode="agentic",
                )
                return QueryResult(
                    question=question,
                    answer=answer,
                    contexts=[c.get("text", "") for c in all_chunks],
                    citations=list(dict.fromkeys(all_citations)),
                    latency_ms=latency_ms,
                    retrieval_mode="agentic",
                    model_version=get_settings().groq_model,  # final synthesis model
                )

            elif reasoning.upper().startswith("SEARCH:"):
                # LLM wants more info — re-search on sub-query
                sub_query = reasoning[7:].strip()
                log.info("agentic_retriever.sub_search", query=sub_query)

                sub_results = await self._local.search(
                    sub_query,
                    session_id=session_id,
                    tenant=tenant,
                )
                sub_chunks = sub_results.get("chunks", [])

                # Only add chunks not already seen
                seen_ids = {c.get("chunk_id") for c in all_chunks}
                new_chunks = [c for c in sub_chunks if c.get("chunk_id") not in seen_ids]
                all_chunks.extend(new_chunks)

                if new_chunks:
                    sub_ctx, sub_cits = self._ctx_builder.build(
                        local_results={"chunks": new_chunks, "entities": sub_results.get("entities", [])},
                        global_results={},
                        top_k=3,
                    )
                    if sub_ctx:
                        context_sections.append(f"[Search: {sub_query}]\n{sub_ctx}")
                        all_citations.extend(sub_cits)
                else:
                    log.info("agentic_retriever.no_new_chunks", sub_query=sub_query)
                    break
            else:
                # Unexpected format — treat as final answer
                break

        # Max steps reached — synthesize with full 70B model for quality
        final_context = "\n\n---\n\n".join(context_sections)
        final_answer = await self._synthesize(
            _FINAL_PROMPT.format(context=final_context, question=question)
        )

        # ── Claim verification — strip ungrounded sentences ────────────────────
        if get_settings().retrieval.get("claim_verification", False):
            final_answer, n_removed = await self._verifier.verify(final_answer, final_context)
            if n_removed:
                log.info("agentic_retriever.claims_stripped", n_removed=n_removed)

        latency_ms = (time.monotonic() - t0) * 1000
        log.info(
            "agentic_retriever.done",
            steps=self._max_steps,
            latency_ms=round(latency_ms, 1),
            mode="agentic_fallback",
        )

        return QueryResult(
            question=question,
            answer=final_answer.strip(),
            contexts=[c.get("text", "") for c in all_chunks],
            citations=list(dict.fromkeys(all_citations)),
            latency_ms=latency_ms,
            retrieval_mode="agentic",
            model_version=get_settings().groq_model,
        )
