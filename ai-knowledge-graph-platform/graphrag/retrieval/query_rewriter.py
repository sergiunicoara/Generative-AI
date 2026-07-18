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
You rewrite a search query to improve document retrieval. Do NOT answer it.

Rewrite the question into a single search query that:
- Expands domain acronyms inline (e.g. "APQP" -> "APQP Advanced Product Quality Planning").
- Normalizes any revision/version phrasing to compact form (e.g. "revision 2" -> "rev2", "rev.4" -> "rev4").
- Adds at most two synonym or paraphrase terms that aid keyword matching.
- Preserves every entity, document ID, and constraint from the original.

Return ONLY the rewritten query on a single line. No preamble, no quotes, no explanation.

Question: {question}

Rewritten query:"""

# A rewrite that balloons past this multiple of the original length is almost
# always the model ignoring instructions and answering / rambling — discard it.
_MAX_EXPANSION_RATIO = 6


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

        log.info("query_rewriter.rewritten", original=question[:80], rewritten=rewritten[:80])
        return rewritten
