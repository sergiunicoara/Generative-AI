"""Central LLM client — routes text generation to Groq, embeddings stay on Gemini.

Usage:
    from graphrag.core.llm_client import get_llm, get_embedder

    text  = await get_llm().generate(prompt, model=cfg.gemini_ingest_model, json_mode=True)
    vecs  = await get_embedder().embed(texts)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# ── Groq text-generation client ───────────────────────────────────────────────

class GroqLLM:
    """Async wrapper around Groq chat completions (sync SDK via executor)."""

    def __init__(self, api_key: str, default_model: str):
        from groq import Groq
        self._client = Groq(api_key=api_key)
        self._default_model = default_model

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> str:
        model = model or self._default_model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.chat.completions.create(**kwargs),
        )
        return response.choices[0].message.content or ""


# ── Gemini embedding client (unchanged) ──────────────────────────────────────

class GeminiEmbedder:
    """Thin async wrapper around Gemini embed_content (kept for 3072-d vectors)."""

    def __init__(self, api_key: str, model: str):
        from google import genai
        from google.genai import types as genai_types
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._types = genai_types

    async def embed(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.models.embed_content(
                model=self._model,
                contents=texts,
                config=self._types.EmbedContentConfig(task_type="retrieval_document"),
            ),
        )
        return [e.values for e in response.embeddings]


# ── Singletons ────────────────────────────────────────────────────────────────

_llm: GroqLLM | None = None
_fast_llm: GroqLLM | None = None
_embedder: GeminiEmbedder | None = None


def get_llm() -> GroqLLM:
    """Return the primary (large) LLM — llama-3.3-70b for synthesis."""
    global _llm
    if _llm is None:
        from graphrag.core.config import get_settings
        cfg = get_settings()
        _llm = GroqLLM(
            api_key=cfg.groq_api_key,
            default_model=cfg.groq_model,
        )
    return _llm


def get_fast_llm() -> GroqLLM:
    """Return the fast (small) LLM — llama-3.1-8b-instant for cheap reasoning steps.

    Used by the agentic retriever for intermediate SEARCH/ANSWER decisions.
    At ~800 tok/s on Groq vs ~150 tok/s for 70B, each reasoning step costs
    ~0.2s instead of ~1.5s. The final synthesis always uses the 70B model.
    """
    global _fast_llm
    if _fast_llm is None:
        from graphrag.core.config import get_settings
        cfg = get_settings()
        _fast_llm = GroqLLM(
            api_key=cfg.groq_api_key,
            default_model=cfg.groq_fast_model,
        )
    return _fast_llm


def get_embedder() -> GeminiEmbedder:
    global _embedder
    if _embedder is None:
        from graphrag.core.config import get_settings
        cfg = get_settings()
        _embedder = GeminiEmbedder(
            api_key=cfg.google_api_key,
            model=cfg.gemini_embed_model,
        )
    return _embedder
