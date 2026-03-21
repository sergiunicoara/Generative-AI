"""Local search: vector similarity over chunks → entity graph expansion."""

from __future__ import annotations

import structlog

from graphrag.core.config import get_settings
from graphrag.graph.neo4j_client import get_neo4j
from graphrag.ingestion.embedder import Embedder

log = structlog.get_logger(__name__)


class LocalSearch:
    def __init__(self):
        self._cfg = get_settings().retrieval
        self._neo4j = get_neo4j()
        self._embedder = Embedder()

    async def search(self, question: str) -> dict:
        embedding = await self._embedder.embed_text(question)

        top_k = self._cfg.get("local_top_k", 10)
        hops  = self._cfg.get("multihop_depth", 2)

        # Step 1 — vector similarity: find seed chunks
        seed_chunks = await self._neo4j.vector_search_chunks(embedding, top_k=top_k)
        seed_ids    = [c["chunk_id"] for c in seed_chunks]

        # Step 2 — graph traversal: follow entity relations N hops and
        #           pull back the chunks those neighbors appear in
        hop_chunks  = await self._neo4j.get_multihop_chunks(seed_ids, hops=hops)

        # Merge — seed chunks first (higher relevance), then graph-expanded
        seen: set[str] = set(seed_ids)
        extra_chunks = [c for c in hop_chunks if c["chunk_id"] not in seen]
        all_chunks   = seed_chunks + extra_chunks

        # Step 3 — entity context from all retrieved chunk ids
        all_ids  = [c["chunk_id"] for c in all_chunks]
        entities = await self._neo4j.get_entity_neighbors(all_ids)

        log.info(
            "local_search.done",
            seed_chunks=len(seed_chunks),
            hop_chunks=len(extra_chunks),
            entities=len(entities),
            hops=hops,
        )
        return {"chunks": all_chunks, "entities": entities}
