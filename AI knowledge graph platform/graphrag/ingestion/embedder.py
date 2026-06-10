"""Batch embeddings via OpenAI text-embedding-3-large (3072d)."""

from __future__ import annotations

from itertools import islice

import structlog

from graphrag.core.config import get_settings
from graphrag.core.llm_client import get_embedder
from graphrag.core.models import Chunk

log = structlog.get_logger(__name__)


def _batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


class Embedder:
    def __init__(self):
        cfg = get_settings()
        self._batch_size = cfg.ingestion.get("embedding_batch_size", 100)

    async def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        all_texts = [c.text for c in chunks]
        embedder = get_embedder()

        embeddings: list[list[float]] = []
        for batch_texts in _batched(all_texts, self._batch_size):
            batch_embeddings = await embedder.embed(batch_texts)
            embeddings.extend(batch_embeddings)
            log.info("embedder.batch_done", count=len(batch_texts))

        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedder count mismatch: expected {len(chunks)} embeddings "
                f"but received {len(embeddings)}."
            )

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        return chunks

    async def embed_text(self, text: str, task_type: str = "retrieval_query") -> list[float]:
        embeddings = await get_embedder().embed([text])
        return embeddings[0]
