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

import asyncio
import time
from typing import Optional

import structlog
from google import genai

from graphrag.core.config import get_settings
from graphrag.core.models import QueryResult
from graphrag.retrieval.local_search import LocalSearch
from graphrag.retrieval.context_builder import ContextBuilder

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
You are an expert assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, say so explicitly.
Always cite sources by chunk ID.

Context:
{context}

Question: {question}

Answer:"""

_LOW_CONFIDENCE_SIGNALS = (
    "i don't know",
    "i do not know",
    "not enough information",
    "cannot answer",
    "insufficient",
    "no information",
    "context does not",
    "not mentioned",
    "not provided",
    "no relevant",
)


def _is_low_confidence(answer: str, citations: list[str]) -> bool:
    """Heuristic: answer is weak if it hedges or has no citations."""
    lower = answer.lower()
    hedges = any(sig in lower for sig in _LOW_CONFIDENCE_SIGNALS)
    return hedges or len(citations) == 0


class AgenticRetriever:
    """
    Iterative retrieval agent — searches, reasons, re-searches until confident.
    Used as fallback when HybridRetriever returns a low-confidence answer.
    """

    def __init__(self, max_steps: int = 4):
        cfg = get_settings()
        self._client = genai.Client(api_key=cfg.google_api_key)
        self._model = cfg.gemini_query_model
        self._local = LocalSearch()
        self._ctx_builder = ContextBuilder()
        self._max_steps = max_steps

    def _llm(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        resp = loop.run_until_complete(
            loop.run_in_executor(
                None,
                lambda: self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                ),
            )
        )
        return resp.text.strip()

    async def _llm_async(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=prompt,
            ),
        )
        return resp.text.strip()

    async def retrieve_and_answer(
        self,
        question: str,
        initial_context: str = "",
        initial_citations: list[str] | None = None,
    ) -> QueryResult:
        t0 = time.monotonic()

        all_chunks: list[dict] = []
        all_citations: list[str] = list(initial_citations or [])
        context_sections: list[str] = []

        if initial_context:
            context_sections.append(initial_context)

        # Initial search on the original question
        seed_results = await self._local.search(question)
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

        # Iterative reasoning loop
        for step in range(self._max_steps):
            current_context = "\n\n---\n\n".join(context_sections)

            reasoning = await self._llm_async(
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
                # LLM is confident — extract the answer
                answer = reasoning[7:].strip()
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
                    model_version=self._model,
                )

            elif reasoning.upper().startswith("SEARCH:"):
                # LLM wants more info — re-search on sub-query
                sub_query = reasoning[7:].strip()
                log.info("agentic_retriever.sub_search", query=sub_query)

                sub_results = await self._local.search(sub_query)
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

        # Max steps reached — synthesize with all accumulated context
        final_context = "\n\n---\n\n".join(context_sections)
        final_answer = await self._llm_async(
            _FINAL_PROMPT.format(context=final_context, question=question)
        )

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
            model_version=self._model,
        )
