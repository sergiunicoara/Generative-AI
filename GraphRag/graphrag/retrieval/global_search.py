"""Global search: community summary map-reduce for broad questions."""

from __future__ import annotations

import asyncio

from google import genai
import structlog

from graphrag.core.config import get_settings
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
        self._client = genai.Client(api_key=cfg.google_api_key)
        self._model_name = cfg.gemini_query_model
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
        loop = asyncio.get_event_loop()
        map_tasks = [
            loop.run_in_executor(
                None,
                lambda c=c: self._client.models.generate_content(
                    model=self._model_name,
                    contents=_MAP_PROMPT.format(
                        question=question,
                        summary=c["summary"],
                    ),
                ),
            )
            for c in communities
        ]
        map_responses = await asyncio.gather(*map_tasks)

        partial_answers = []
        for community, response in zip(communities, map_responses):
            text = response.text.strip()
            if "not relevant" not in text.lower():
                partial_answers.append(f"[Level {community['level']}] {text}")

        if not partial_answers:
            return {"communities": communities, "synthesized_answer": ""}

        # Reduce: synthesize all partial answers
        reduce_response = await loop.run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=self._model_name,
                contents=_REDUCE_PROMPT.format(
                    question=question,
                    partial_answers="\n\n".join(partial_answers),
                ),
            ),
        )

        log.info(
            "global_search.done",
            communities=len(communities),
            partial_answers=len(partial_answers),
        )
        return {
            "communities": communities,
            "synthesized_answer": reduce_response.text.strip(),
        }
