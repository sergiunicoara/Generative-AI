"""Claim-level grounding verifier.

Splits an LLM answer into atomic sentences, checks each against the retrieved
context using the fast 8B model, and removes sentences that aren't supported.

This is a post-synthesis step — it runs after the main answer is generated and
before the QueryResult is returned to the caller.

Usage:
    verifier = ClaimVerifier()
    clean_answer, n_removed = await verifier.verify(answer, context)
"""

from __future__ import annotations

import re

import structlog

from graphrag.core.llm_client import get_fast_llm

log = structlog.get_logger(__name__)

# Maximum context characters to pass per claim check (~1500 tokens at 4 chars/token)
_MAX_CONTEXT_CHARS = 6000

# Fallback message when all claims are stripped
_FALLBACK = (
    "The retrieved documents do not contain sufficient information to answer this."
)

_VERIFY_PROMPT = """\
Context:
{context}

Claim: "{claim}"

Is this claim directly supported by the context above?
Reply with YES or NO only. Do not explain."""


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation, preserve the punctuation."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _truncate_context(context: str) -> str:
    """Trim context to _MAX_CONTEXT_CHARS to keep prompts fast."""
    if len(context) <= _MAX_CONTEXT_CHARS:
        return context
    return context[:_MAX_CONTEXT_CHARS] + "\n[context truncated]"


class ClaimVerifier:
    """Post-synthesis claim grounding checker using the fast 8B model.

    For each sentence in the answer, asks the 8B model whether the claim is
    directly supported by the retrieved context. Sentences that fail are
    removed. If all sentences fail, returns the standard insufficient-context
    fallback message.

    Design notes:
    - Uses get_fast_llm() (8B, ~0.2s/call) not get_llm() (70B, ~1.5s/call).
    - Sentences are verified concurrently (asyncio.gather) to minimise latency
      overhead (typically adds ~0.3–0.5s for a 3-sentence answer).
    - Empty answers and the fallback message itself are passed through unchanged.
    """

    async def _check_claim(self, claim: str, context: str) -> bool:
        """Return True if the fast model judges the claim as grounded."""
        prompt = _VERIFY_PROMPT.format(
            context=_truncate_context(context),
            claim=claim,
        )
        raw = await get_fast_llm().generate(prompt) or ""
        # Accept YES, YES., YES!, etc. — raw is already a plain string from Groq/DeepSeek
        return raw.strip().upper().startswith("YES")

    async def verify(self, answer: str, context: str) -> tuple[str, int]:
        """Verify each sentence in *answer* against *context*.

        Returns
        -------
        verified_answer : str
            Answer with ungrounded sentences removed.  Falls back to
            _FALLBACK if all sentences are removed.
        n_removed : int
            Number of sentences that were stripped.
        """
        if not answer or not answer.strip():
            return answer, 0

        # Pass through if the answer is already the fallback / refusal message
        if _FALLBACK in answer:
            return answer, 0

        sentences = _split_sentences(answer)
        if not sentences:
            return answer, 0

        import asyncio
        results = await asyncio.gather(
            *[self._check_claim(s, context) for s in sentences],
            return_exceptions=False,
        )

        kept = [s for s, ok in zip(sentences, results) if ok]
        n_removed = len(sentences) - len(kept)

        if n_removed:
            log.info(
                "claim_verifier.stripped",
                total=len(sentences),
                removed=n_removed,
                kept=len(kept),
            )

        if not kept:
            return _FALLBACK, n_removed

        return " ".join(kept), n_removed
