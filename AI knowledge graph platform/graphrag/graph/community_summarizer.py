"""Generate and embed community summaries via Groq (LLM) + Gemini (embeddings)."""

from __future__ import annotations

import asyncio

import structlog

from graphrag.core.llm_client import get_llm, get_embedder
from graphrag.core.models import Community
from graphrag.graph.neo4j_client import get_neo4j

log = structlog.get_logger(__name__)


class CommunitySummarizer:
    def __init__(self):
        self._neo4j = get_neo4j()

    async def summarize_all(self, communities: list[Community]) -> list[Community]:
        tasks = [self._summarize_one(c) for c in communities]
        return await asyncio.gather(*tasks)

    async def _summarize_one(self, community: Community) -> Community:
        # Fetch entity names for this community
        entity_names = await self._neo4j.run(
            """
            UNWIND $ids AS eid
            MATCH (e:Entity {id: eid})
            RETURN e.name AS name, e.type AS type, e.description AS description
            """,
            ids=community.member_entity_ids,
        )

        entity_text = "\n".join(
            f"- {e['name']} ({e['type']}): {e['description']}" for e in entity_names
        )

        prompt = (
            "You are summarizing a community of related entities from a knowledge graph.\n"
            "Write a concise 2-3 sentence summary of what this group of entities represents, "
            "their relationships, and why they are grouped together.\n\n"
            f"Entities:\n{entity_text}\n\nSummary:"
        )

        community.summary = await get_llm().generate(prompt) or f"[fallback] Community {community.id}"

        # Embed the summary
        embeddings = await get_embedder().embed([community.summary])
        community.embedding = embeddings[0]

        log.info(
            "community_summarizer.done",
            community_id=community.id,
            level=community.level,
        )
        return community
