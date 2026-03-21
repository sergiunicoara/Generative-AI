"""Cross-encoder reranker: re-scores RRF-fused chunks against the query.

Pipeline position:
    Vector ANN + BM25 → RRF fusion (top_k) → CrossEncoderReranker → top rerank_k → LLM

Uses sentence-transformers cross-encoder/ms-marco-MiniLM-L-6-v2:
  - Lightweight (22M params), fast on CPU (~5ms per chunk)
  - Trained on MS MARCO passage ranking (query-passage relevance)
  - Returns a relevance logit per (query, passage) pair
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

import structlog
from sentence_transformers import CrossEncoder

log = structlog.get_logger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _get_cross_encoder() -> CrossEncoder:
    """Load once, reuse across requests (lazy singleton)."""
    log.info("reranker.loading", model=_MODEL_NAME)
    return CrossEncoder(_MODEL_NAME, max_length=512)


class CrossEncoderReranker:
    """Re-ranks a list of chunks against the query using a cross-encoder model.

    Args:
        top_k: Number of chunks to return after reranking.
    """

    def __init__(self, top_k: int = 5):
        self._top_k = top_k

    async def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """Rerank chunks in a thread executor (model is CPU-bound).

        Args:
            query:  The user question.
            chunks: List of chunk dicts with at least a 'text' key.

        Returns:
            Top-k chunks sorted by cross-encoder score (descending),
            with 'rerank_score' field added to each dict.
        """
        if not chunks:
            return chunks

        loop = asyncio.get_event_loop()
        reranked = await loop.run_in_executor(
            None,
            lambda: self._score(query, chunks),
        )
        top = reranked[: self._top_k]
        log.info(
            "reranker.done",
            input_chunks=len(chunks),
            output_chunks=len(top),
            top_score=round(top[0]["rerank_score"], 4) if top else 0,
        )
        return top

    def _score(self, query: str, chunks: list[dict]) -> list[dict]:
        model = _get_cross_encoder()
        pairs = [(query, c["text"]) for c in chunks]
        scores = model.predict(pairs)          # numpy array of floats
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
        return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
