"""Local search: vector ANN + BM25 → RRF → cross-encoder rerank → multi-hop → GNN scorer.

Full 6-step retrieval pipeline
-------------------------------
1. Vector ANN          — semantic similarity via Neo4j vector index
2. BM25 + RRF fusion   — lexical hybrid search; merges both ranked lists
3. Cross-encoder       — deep pairwise (query, chunk) relevance scoring
4. Multi-hop traversal — follow RELATES_TO edges to find bridging chunks
5. GNN scoring         — graph-propagated entity embeddings (GCN / GAT)
6. Entity context      — gather entity descriptions for LLM prompt

Additional features
-------------------
- Session context    — resolves ambiguous follow-up queries across turns
- Authority weights  — edges from superseded documents get confidence penalty
- Source type filter — inferred/LLM-generated edges scored separately
"""

from __future__ import annotations

import asyncio

import structlog

from graphrag.core.config import get_settings
from graphrag.graph.document_authority import DocumentAuthorityService
from graphrag.graph.gnn_scorer import GNNScorer
from graphrag.graph.neo4j_client import get_neo4j
from graphrag.ingestion.embedder import Embedder
from graphrag.retrieval.bm25_search import HybridBM25Search
from graphrag.retrieval.reranker import CrossEncoderReranker
from graphrag.retrieval.session_context import get_session_context

log = structlog.get_logger(__name__)

# ── Query-adaptive weight routing ─────────────────────────────────────────────
_RELATIONAL_SIGNALS = [
    "relationship", "how did", "caused by", "connected", "led to",
    "between", "link", "related", "chain", "through", "via",
    "owned by", "founded by", "acquired", "impact of", "effect of",
]


def _adaptive_weights(question: str, alpha_default: float, beta_default: float):
    q = question.lower()
    if any(s in q for s in _RELATIONAL_SIGNALS):
        return 0.5, 0.5
    return alpha_default, beta_default


class LocalSearch:
    def __init__(self):
        self._cfg            = get_settings().retrieval
        self._neo4j          = get_neo4j()
        self._embedder       = Embedder()
        self._bm25           = HybridBM25Search()
        self._reranker       = CrossEncoderReranker(
            top_k=self._cfg.get("rerank_top_k", 5)
        )
        self._gnn = GNNScorer(
            gnn_type                  = self._cfg.get("gnn_type", "gat"),
            num_layers                = self._cfg.get("gnn_layers", 2),
            alpha                     = self._cfg.get("gnn_alpha", 0.9),
            beta                      = self._cfg.get("gnn_beta",  0.1),
            edge_confidence_threshold = self._cfg.get("gnn_edge_confidence_threshold", 0.7),
            confidence_half_life_days = self._cfg.get("gnn_confidence_half_life_days", 0),
        )
        self._adaptive_weights   = self._cfg.get("gnn_adaptive_weights", True)
        self._authority_svc      = DocumentAuthorityService(self._neo4j)
        self._session_ctx        = get_session_context()

    async def search(
        self,
        question: str,
        session_id: str = "",
    ) -> dict:
        top_k      = self._cfg.get("local_top_k", 10)
        hops       = self._cfg.get("multihop_depth", 2)
        use_bm25   = self._cfg.get("bm25_enabled", True)
        use_rerank = self._cfg.get("reranker_enabled", True)
        use_gnn    = self._cfg.get("gnn_enabled", True)

        # ── Session context: resolve ambiguous follow-up queries ───────────────
        enriched_question = self._session_ctx.enrich_query(session_id, question)
        if enriched_question != question:
            log.info("local_search.query_enriched", session_id=session_id)

        # Step 1 — vector ANN
        embedding     = await self._embedder.embed_text(enriched_question)
        vector_chunks = await self._neo4j.vector_search_chunks(embedding, top_k=top_k)

        # Step 2 — BM25 + RRF fusion
        if use_bm25:
            fused_chunks = await self._bm25.search(
                query=enriched_question,
                vector_chunks=vector_chunks,
                top_k=top_k,
            )
        else:
            fused_chunks = vector_chunks

        # Step 3 — cross-encoder reranking
        if use_rerank and fused_chunks:
            seed_chunks = await self._reranker.rerank(enriched_question, fused_chunks)
        else:
            seed_chunks = fused_chunks

        seed_ids = [c["chunk_id"] for c in seed_chunks]

        # Step 4 — multi-hop graph traversal
        hop_chunks = await self._neo4j.get_multihop_chunks(seed_ids, hops=hops)

        seen: set[str] = set(seed_ids)
        extra_chunks = [c for c in hop_chunks if c["chunk_id"] not in seen]
        all_chunks   = seed_chunks + extra_chunks
        all_ids      = [c["chunk_id"] for c in all_chunks]

        # Step 5 — GNN scoring with authority-weighted edges
        if use_gnn and all_chunks:
            alpha, beta = _adaptive_weights(
                enriched_question,
                self._cfg.get("gnn_alpha", 0.9),
                self._cfg.get("gnn_beta",  0.1),
            ) if self._adaptive_weights else (
                self._cfg.get("gnn_alpha", 0.9),
                self._cfg.get("gnn_beta",  0.1),
            )

            chunk_entities, entity_edges = await asyncio.gather(
                self._neo4j.get_chunk_entity_embeddings(all_ids),
                _fetch_subgraph_edges(self._neo4j, all_ids),
            )

            # Apply document authority weights to edge confidence
            entity_edges = await self._authority_svc.apply_authority_weights(
                entity_edges
            )

            loop = asyncio.get_event_loop()
            all_chunks = await loop.run_in_executor(
                None,
                lambda: self._gnn.score(
                    query_vec      = embedding,
                    chunks         = all_chunks,
                    chunk_entities = chunk_entities,
                    entity_edges   = entity_edges,
                    alpha          = alpha,
                    beta           = beta,
                ),
            )

        # Step 6 — entity context
        entities = await self._neo4j.get_entity_neighbors(all_ids)

        # ── Record turn in session context ────────────────────────────────────
        referenced_entities = list({
            e.get("entity", "") for e in entities if e.get("entity")
        })
        self._session_ctx.record_turn(
            session_id=session_id,
            question=question,
            answer="",   # answer not available yet at this stage
            referenced_entities=referenced_entities,
            referenced_chunks=all_ids,
        )

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
            session_id=session_id or "none",
        )
        return {"chunks": all_chunks, "entities": entities}


# ── helpers ───────────────────────────────────────────────────────────────────

async def _fetch_subgraph_edges(neo4j, chunk_ids: list[str]) -> list[dict]:
    """Helper: get entity names from chunks then fetch the connecting edges."""
    rows = await neo4j.get_chunk_entity_embeddings(chunk_ids)
    names = list({r["entity_name"] for r in rows})
    if not names:
        return []
    return await neo4j.get_entity_relations_subgraph(names)
