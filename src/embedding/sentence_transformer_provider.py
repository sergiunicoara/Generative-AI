"""Local embedding provider — sentence-transformers, no API key, fully
offline. Chosen over a hosted API (OpenAI/Cohere/...) because no
EMBEDDING_API_KEY is configured anywhere in this environment, and a real key
is not something to fabricate or request through chat (see repo-wide safety
rules on credential handling). A local model keeps entity-resolution semantic
scoring genuinely real — not a stub returning a fixed vector — and fully
reproducible without network access or per-call cost. Swap for a hosted
provider later by implementing the same EmbeddingProvider protocol; nothing
else in src/resolution/ changes.

Model: all-MiniLM-L6-v2 (384-dim, ~80MB, CPU-fast) — a standard, well-known
small sentence-embedding model, not a bespoke choice.
"""

from __future__ import annotations

import asyncio
import functools

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_DIMENSION = 384


class SentenceTransformerEmbeddingProvider:
    model_name = _MODEL_NAME
    dimension = _DIMENSION

    def __init__(self):
        # Loaded once per process — model loading (~1-2s) happens at
        # construction, not per embed() call.
        self._model = SentenceTransformer(_MODEL_NAME)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Batched — one model.encode() call for the whole list, never one
        call per text (the same N+1-avoidance principle as everywhere else in
        this repo). sentence-transformers' encode() is synchronous/CPU-bound,
        so it runs in a thread pool executor to avoid blocking the event loop.
        """
        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(
            None, functools.partial(self._model.encode, texts, normalize_embeddings=True)
        )
        return [vector.tolist() for vector in vectors]
