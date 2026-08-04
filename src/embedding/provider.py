"""Pluggable batch embedding provider (docs/plan.md's Stack list). Structurally
identical in spirit to src/extraction/provider.py's ExtractionProvider: one
Protocol, swap implementations without touching callers.
"""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    model_name: str
    dimension: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...
