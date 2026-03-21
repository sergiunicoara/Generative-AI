"""
BM25 fulltext search + Reciprocal Rank Fusion (RRF) with vector results.

True hybrid search:
  BM25 (keyword)  ──┐
                    ├── RRF merge ──► unified ranked chunk list
  Vector (ANN)    ──┘

RRF formula: score(d) = Σ 1 / (k + rank(d))
where k=60 is a smoothing constant (standard value from the original paper).
"""

from __future__ import annotations

import structlog

from graphrag.graph.neo4j_client import get_neo4j

log = structlog.get_logger(__name__)

RRF_K = 60  # standard smoothing constant


def _reciprocal_rank_fusion(
    *ranked_lists: list[dict],
    id_key: str = "chunk_id",
    top_k: int = 10,
) -> list[dict]:
    """
    Merge N ranked lists using Reciprocal Rank Fusion.

    Each list is a list of dicts with at least `id_key` and `text`.
    Returns a merged list sorted by RRF score, deduplicated.
    """
    rrf_scores: dict[str, float] = {}
    chunk_store: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            doc_id = item.get(id_key)
            if not doc_id:
                continue
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            if doc_id not in chunk_store:
                chunk_store[doc_id] = item

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {**chunk_store[doc_id], "score": rrf_score}
        for doc_id, rrf_score in merged
        if doc_id in chunk_store
    ]


class HybridBM25Search:
    """
    True hybrid search: BM25 fulltext + vector ANN → RRF fusion.

    Drop-in companion to LocalSearch — call after vector search to get
    a fused ranked list.
    """

    def __init__(self):
        self._neo4j = get_neo4j()

    async def search(
        self,
        query: str,
        vector_chunks: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        """
        Args:
            query:         Raw question string (for BM25)
            vector_chunks: Already-retrieved vector ANN results
            top_k:         Final number of chunks to return after fusion

        Returns:
            RRF-fused list of chunks, each with a `score` and `retrieval` tag.
        """
        # BM25 on chunk text
        bm25_chunks = await self._neo4j.bm25_search_chunks(query, top_k=top_k)

        # BM25 on entity names/descriptions → back to chunks
        entity_chunks = await self._neo4j.bm25_search_entities(query, top_k=top_k)

        # Merge entity BM25 into chunk BM25 (same RRF pass)
        bm25_combined = _reciprocal_rank_fusion(
            bm25_chunks, entity_chunks, top_k=top_k * 2
        )

        # Final RRF: vector results vs BM25 results
        fused = _reciprocal_rank_fusion(
            vector_chunks,
            bm25_combined,
            top_k=top_k,
        )

        # Tag each chunk with which retrieval method(s) found it
        vector_ids = {c["chunk_id"] for c in vector_chunks}
        bm25_ids   = {c["chunk_id"] for c in bm25_combined}
        for chunk in fused:
            cid = chunk["chunk_id"]
            sources = []
            if cid in vector_ids:
                sources.append("vector")
            if cid in bm25_ids:
                sources.append("bm25")
            chunk["retrieval"] = "+".join(sources) if sources else "rrf"

        log.info(
            "hybrid_bm25.done",
            vector=len(vector_chunks),
            bm25=len(bm25_combined),
            fused=len(fused),
        )
        return fused
