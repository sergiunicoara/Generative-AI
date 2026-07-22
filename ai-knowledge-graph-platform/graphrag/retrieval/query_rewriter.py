"""Stage-1 query rewriter — expand/normalize a query before first-pass retrieval.

Sits in front of the hybrid retriever. Transforms the raw user question into a
retrieval-optimized query (acronym expansion, revision-phrasing normalization,
a synonym variant or two) to improve first-pass recall from BM25 + vector +
graph search.

Scope is deliberately narrow — this does NOT decompose multi-part questions or
do multi-hop reasoning; that already lives in ``AgenticRetriever`` (IRCoT). It
also never touches the string used for answer synthesis or evaluation: only the
*search* query is rewritten, so RAGAS faithfulness is measured against the
user's real question.

Fails open: any error, timeout, or empty model response returns the original
question unchanged. The rewriter is a recall booster, never a gate.
"""

from __future__ import annotations

import structlog

from graphrag.core.config import get_settings
from graphrag.core.llm_client import get_fast_llm

log = structlog.get_logger(__name__)

_REWRITE_PROMPT = """\
You rewrite a search query to improve keyword-based document retrieval. Do NOT answer it.

The query will be matched against document chunks by keyword search, where
generic phrasing dilutes the specific terms that actually distinguish the
right chunk from every other chunk. Rewrite the question into a single
search query that:
- Keeps the 2-5 most distinctive terms verbatim: named entities, document
  IDs, technical terms, exact numbers/figures, acronyms.
- Drops generic filler that adds no retrieval value: phrases like "for X
  campaigns", "per the Y", "according to", "in the context of", and other
  wrapping that just restates the corpus/tenant name.
- If — and only if — an acronym is literally present in the question, you
  may expand it inline (acronym plus its expansion). Never introduce an
  acronym, synonym, expansion, or abbreviation that is not already present
  or directly implied by the question's own words. If the question has no
  acronyms or version/revision phrasing, do not add any — leave that part
  of the rewrite untouched.
- Preserves every entity, document ID, and numeric constraint from the
  original question verbatim.

Output rules:
- Plain search terms only — no boolean operators (AND, OR), no quotes,
  no placeholders (e.g. "XXXX", "document ID:"), no explanations.
- Return ONLY the rewritten query on a single line. No preamble.

Question: {question}

Rewritten query:"""

# A rewrite that balloons past this multiple of the original length is almost
# always the model ignoring instructions and answering / rambling — discard it.
_MAX_EXPANSION_RATIO = 6

# Malformed-output markers: the model occasionally emits boolean-query syntax
# or literal placeholder text instead of a plain keyword string. BM25 doesn't
# parse boolean operators as such (they're indexed as ordinary tokens), so a
# rewrite containing these actively hurts retrieval rather than helping it —
# reject and fall back to the original question.
_MALFORMED_MARKERS = (" AND ", " OR ", "XXXX", "document id:", "```")


class QueryRewriter:
    """Expand and normalize a query for first-pass retrieval (fast 8B model)."""

    def __init__(self) -> None:
        self._cfg = get_settings().retrieval

    async def rewrite(self, question: str, tenant: str = "default") -> str:
        """Return a retrieval-optimized query, or the original on any failure."""
        try:
            raw = await get_fast_llm().generate(
                _REWRITE_PROMPT.format(question=question)
            )
        except Exception as exc:  # fail open — never block a query on the rewriter
            log.warning("query_rewriter.failed", error=str(exc))
            return question

        rewritten = (raw or "").strip().splitlines()[0].strip() if raw else ""
        # Guard against empty, echoed, or runaway (model-answered-instead) output.
        if (
            not rewritten
            or rewritten.lower() == question.lower()
            or len(rewritten) > len(question) * _MAX_EXPANSION_RATIO
        ):
            return question
        # Guard against malformed output (boolean syntax, literal placeholders)
        # that BM25 can't use as intended — see _MALFORMED_MARKERS above.
        rewritten_upper = rewritten.upper()
        if any(marker.upper() in rewritten_upper for marker in _MALFORMED_MARKERS):
            log.warning("query_rewriter.malformed_output", rewritten=rewritten[:120])
            return question

        log.info("query_rewriter.rewritten", original=question[:80], rewritten=rewritten[:80])
        return rewritten
