"""Local search: vector ANN + BM25 fulltext → RRF fusion → cross-encoder rerank → graph expansion."""

from __future__ import annotations

import structlog

from graphrag.core.config import get_settings
from graphrag.graph.neo4j_client import get_neo4j
from graphrag.ingestion.embedder import Embedder
from graphrag.retrieval.bm25_search import HybridBM25Search
from graphrag.retrieval.reranker import CrossEncoderReranker

log = structlog.get_logger(__name__)


class LocalSearch:
    def __init__(self):
        self._cfg = get_settings().retrieval
        self._neo4j = get_neo4j()
        self._embedder = Embedder()
        self._bm25 = HybridBM25Search()
        self._reranker = CrossEncoderReranker(
            top_k=self._cfg.get("rerank_top_k", 5)
        )

    async def search(self, question: str) -> dict:
        top_k    = self._cfg.get("local_top_k", 10)
        hops     = self._cfg.get("multihop_depth", 2)
        use_bm25 = self._cfg.get("bm25_enabled", True)
        use_rerank = self._cfg.get("reranker_enabled", True)

        # Step 1 — vector ANN: semantic similarity
        embedding     = await self._embedder.embed_text(question)
        vector_chunks = await self._neo4j.vector_search_chunks(embedding, top_k=top_k)

        # Step 2 — BM25 + RRF fusion (hybrid search)
        if use_bm25:
            fused_chunks = await self._bm25.search(
                query=question,
                vector_chunks=vector_chunks,
                top_k=top_k,
            )
        else:
            fused_chunks = vector_chunks

        # Step 3 — cross-encoder reranking: deep pairwise query-chunk scoring
        #           narrows from top_k RRF candidates down to rerank_top_k
        if use_rerank and fused_chunks:
            seed_chunks = await self._reranker.rerank(question, fused_chunks)
        else:
            seed_chunks = fused_chunks

        seed_ids = [c["chunk_id"] for c in seed_chunks]

        # Step 4 — multi-hop graph traversal: follow entity relations N hops
        #           and pull back the chunks those neighbors appear in
        hop_chunks = await self._neo4j.get_multihop_chunks(seed_ids, hops=hops)

        # Merge: seed chunks first (reranked), then graph-expanded
        seen: set[str] = set(seed_ids)
        extra_chunks = [c for c in hop_chunks if c["chunk_id"] not in seen]
        all_chunks   = seed_chunks + extra_chunks

        # Step 5 — entity context from all chunk IDs
        all_ids  = [c["chunk_id"] for c in all_chunks]
        entities = await self._neo4j.get_entity_neighbors(all_ids)

        log.info(
            "local_search.done",
            vector_chunks=len(vector_chunks),
            fused_chunks=len(fused_chunks),
            reranked_chunks=len(seed_chunks),
            hop_chunks=len(extra_chunks),
            entities=len(entities),
            hops=hops,
            bm25=use_bm25,
            reranker=use_rerank,
        )
        return {"chunks": all_chunks, "entities": entities}
