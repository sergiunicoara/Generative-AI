"""Local search: vector ANN + BM25 → RRF → cross-encoder rerank → multi-hop → GNN scorer.

Full 6-step retrieval pipeline
-------------------------------
1. Vector ANN          — semantic similarity via Neo4j vector index
2. BM25 + RRF fusion   — lexical hybrid search; merges both ranked lists
3. Cross-encoder       — deep pairwise (query, chunk) relevance scoring
4. Multi-hop traversal — follow RELATES_TO edges to find bridging chunks
5. GNN scoring         — graph-propagated entity embeddings (GCN / GAT)
6. Entity context      — gather entity descriptions for LLM prompt
"""

from __future__ import annotations

import asyncio

import structlog

from graphrag.core.config import get_settings
from graphrag.graph.gnn_scorer import GNNScorer
from graphrag.graph.neo4j_client import get_neo4j
from graphrag.ingestion.embedder import Embedder
from graphrag.retrieval.bm25_search import HybridBM25Search
from graphrag.retrieval.reranker import CrossEncoderReranker

log = structlog.get_logger(__name__)


class LocalSearch:
    def __init__(self):
        self._cfg      = get_settings().retrieval
        self._neo4j    = get_neo4j()
        self._embedder = Embedder()
        self._bm25     = HybridBM25Search()
        self._reranker = CrossEncoderReranker(
            top_k=self._cfg.get("rerank_top_k", 5)
        )
        self._gnn = GNNScorer(
            gnn_type   = self._cfg.get("gnn_type", "gat"),
            num_layers = self._cfg.get("gnn_layers", 2),
            alpha      = self._cfg.get("gnn_alpha", 0.6),
            beta       = self._cfg.get("gnn_beta",  0.4),
        )

    async def search(self, question: str) -> dict:
        top_k      = self._cfg.get("local_top_k", 10)
        hops       = self._cfg.get("multihop_depth", 2)
        use_bm25   = self._cfg.get("bm25_enabled", True)
        use_rerank = self._cfg.get("reranker_enabled", True)
        use_gnn    = self._cfg.get("gnn_enabled", True)

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

        # Step 4 — multi-hop graph traversal: follow RELATES_TO edges N hops
        #           and pull back the chunks those neighbor entities appear in
        hop_chunks = await self._neo4j.get_multihop_chunks(seed_ids, hops=hops)

        # Merge: reranked seeds first, then graph-expanded extras
        seen: set[str] = set(seed_ids)
        extra_chunks = [c for c in hop_chunks if c["chunk_id"] not in seen]
        all_chunks   = seed_chunks + extra_chunks
        all_ids      = [c["chunk_id"] for c in all_chunks]

        # Step 5 — GNN scoring: propagate entity embeddings over the RELATES_TO
        #           subgraph (GCN or GAT), then blend structural score with
        #           cross-encoder score for a final graph-aware ranking.
        if use_gnn and all_chunks:
            # Fetch entity embeddings + intra-subgraph edges in parallel
            chunk_entities, entity_edges = await asyncio.gather(
                self._neo4j.get_chunk_entity_embeddings(all_ids),
                _fetch_subgraph_edges(self._neo4j, all_ids),
            )
            loop = asyncio.get_event_loop()
            all_chunks = await loop.run_in_executor(
                None,
                lambda: self._gnn.score(
                    query_vec      = embedding,
                    chunks         = all_chunks,
                    chunk_entities = chunk_entities,
                    entity_edges   = entity_edges,
                ),
            )

        # Step 6 — entity context: neighbor names + descriptions for LLM prompt
        entities = await self._neo4j.get_entity_neighbors(all_ids)

        log.info(
            "local_search.done",
            vector_chunks=len(vector_chunks),
            fused_chunks=len(fused_chunks),
            reranked_chunks=len(seed_chunks),
            hop_chunks=len(extra_chunks),
            total_chunks=len(all_chunks),
            entities=len(entities),
            hops=hops,
            bm25=use_bm25,
            reranker=use_rerank,
            gnn=use_gnn,
        )
        return {"chunks": all_chunks, "entities": entities}


# ── helpers ───────────────────────────────────────────────────────────────────

async def _fetch_subgraph_edges(neo4j, chunk_ids: list[str]) -> list[dict]:
    """Helper: get entity names from chunks then fetch the connecting edges."""
    # Re-use the entity embedding query to get names cheaply
    rows = await neo4j.get_chunk_entity_embeddings(chunk_ids)
    names = list({r["entity_name"] for r in rows})
    if not names:
        return []
    return await neo4j.get_entity_relations_subgraph(names)
