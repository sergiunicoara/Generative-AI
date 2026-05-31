"""Global search: community summary map-reduce for broad questions."""

from __future__ import annotations

import asyncio

import structlog

from graphrag.core.config import get_settings
from graphrag.core.llm_client import get_llm
from graphrag.graph.neo4j_client import get_neo4j
from graphrag.ingestion.embedder import Embedder

log = structlog.get_logger(__name__)

_MAP_PROMPT = """\
You are answering a question using the following community summary.
Extract any relevant information, or respond with "Not relevant."

Question: {question}
Community summary: {summary}

Relevant information:"""

_REDUCE_PROMPT = """\
You are synthesizing partial answers from multiple community summaries to answer a question.

Question: {question}

Partial answers:
{partial_answers}

Final comprehensive answer:"""


class GlobalSearch:
    def __init__(self):
        cfg = get_settings()
        self._cfg = cfg.retrieval
        self._neo4j = get_neo4j()
        self._embedder = Embedder()

    async def search(self, question: str, tenant: str = "default") -> dict:
        embedding = await self._embedder.embed_text(question)

        top_k = self._cfg.get("global_top_communities", 5)
        communities = await self._neo4j.vector_search_communities(
            embedding,
            top_k=top_k,
            tenant=tenant,
        )

        if not communities:
            log.warning(
                "global_search.no_communities",
                tenant=tenant,
                hint=(
                    "No Community nodes found. Run scripts/community_rebuild.py "
                    "or enable graph.auto_rebuild_communities in settings."
                ),
            )
            return {"communities": [], "synthesized_answer": ""}

        # Warn if the top communities are connected-components fallbacks —
        # this signals graspologic is missing and global quality is degraded.
        fallback_communities = [
            c for c in communities
            if str(c.get("summary", "")).startswith("[fallback:")
        ]
        if fallback_communities:
            log.error(
                "global_search.degraded_communities",
                fallback_count=len(fallback_communities),
                total=len(communities),
                tenant=tenant,
                impact="answers based on connected-components, not Leiden hierarchy",
            )

        # Map: extract relevant info from each community summary
        llm = get_llm()
        map_tasks = [
            llm.generate(_MAP_PROMPT.format(question=question, summary=c["summary"]))
            for c in communities
        ]
        map_texts = await asyncio.gather(*map_tasks)

        partial_answers = []
        for community, text in zip(communities, map_texts):
            if text and "not relevant" not in text.lower():
                partial_answers.append(f"[Level {community['level']}] {text}")

        if not partial_answers:
            return {"communities": communities, "synthesized_answer": ""}

        # Reduce: synthesize all partial answers
        synthesized = await llm.generate(
            _REDUCE_PROMPT.format(
                question=question,
                partial_answers="\n\n".join(partial_answers),
            )
        )

        log.info(
            "global_search.done",
            communities=len(communities),
            partial_answers=len(partial_answers),
        )
        return {
            "communities": communities,
            "synthesized_answer": synthesized or "",
        }
