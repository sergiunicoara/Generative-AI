"""Batch embeddings via Gemini text-embedding-004."""

from __future__ import annotations

import asyncio
from itertools import islice

from google import genai
from google.genai import types as genai_types
import structlog

from graphrag.core.config import get_settings
from graphrag.core.models import Chunk

log = structlog.get_logger(__name__)


def _batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


class Embedder:
    def __init__(self):
        cfg = get_settings()
        self._client = genai.Client(api_key=cfg.google_api_key)
        self._model = cfg.gemini_embed_model
        self._batch_size = cfg.ingestion.get("embedding_batch_size", 100)

    async def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        loop = asyncio.get_event_loop()
        all_texts = [c.text for c in chunks]

        embeddings: list[list[float]] = []
        for batch_texts in _batched(all_texts, self._batch_size):
            result = await loop.run_in_executor(
                None,
                lambda t=batch_texts: self._client.models.embed_content(
                    model=self._model,
                    contents=t,
                    config=genai_types.EmbedContentConfig(task_type="retrieval_document"),
                ),
            )
            embeddings.extend([e.values for e in result.embeddings])
            log.info("embedder.batch_done", count=len(batch_texts))

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        return chunks

    async def embed_text(self, text: str, task_type: str = "retrieval_query") -> list[float]:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._client.models.embed_content(
                model=self._model,
                contents=text,
                config=genai_types.EmbedContentConfig(task_type=task_type),
            ),
        )
        return result.embeddings[0].values
