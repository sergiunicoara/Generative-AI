"""Generate and embed Gemini summaries for each community."""

from __future__ import annotations

import asyncio

from google import genai
from google.genai import types as genai_types
import structlog

from graphrag.core.config import get_settings
from graphrag.core.models import Community
from graphrag.graph.neo4j_client import get_neo4j

log = structlog.get_logger(__name__)


class CommunitySummarizer:
    def __init__(self):
        cfg = get_settings()
        self._client = genai.Client(api_key=cfg.google_api_key)
        self._model_name = cfg.gemini_ingest_model
        self._embed_model = cfg.gemini_embed_model
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

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
            ),
        )
        community.summary = response.text.strip()

        # Embed the summary
        embed_response = await loop.run_in_executor(
            None,
            lambda: self._client.models.embed_content(
                model=self._embed_model,
                contents=community.summary,
                config=genai_types.EmbedContentConfig(task_type="retrieval_document"),
            ),
        )
        community.embedding = embed_response.embeddings[0].values

        log.info(
            "community_summarizer.done",
            community_id=community.id,
            level=community.level,
        )
        return community
