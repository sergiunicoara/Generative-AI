"""Central LLM client — routes text generation to Groq with DeepSeek fallback;
embeddings use OpenAI text-embedding-3-large (3072d).

Usage:
    from graphrag.core.llm_client import get_llm, get_embedder

    text  = await get_llm().generate(prompt, json_mode=True)
    vecs  = await get_embedder().embed(texts)

Provider strategy
-----------------
Primary:  Groq (llama-3.3-70b-versatile) — lowest latency (~150 tok/s).
Fallback: DeepSeek-V3 (deepseek-chat) — kicks in immediately on Groq rate-limit;
          no sleep, no queuing, same OpenAI-compatible API.
Embeddings: OpenAI text-embedding-3-large (3072d) — replaced Gemini; same
          dimensions, same Neo4j schema, no re-indexing required.

Rate-limit handling (Groq)
---------------------------
When Groq returns a 429, the error message contains the exact wait time in the
form "Please try again in XmY.Zs".  ``GroqLLM.generate()`` parses that value
and sleeps for that duration (capped at ``_MAX_RETRY_WAIT`` seconds) before
retrying.  After ``_MAX_RETRIES`` failed attempts the exception is propagated
to ``FallbackLLM``, which transparently re-issues the call to DeepSeek instead.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# ── Retry config ──────────────────────────────────────────────────────────────
_MAX_RETRIES    = 5          # maximum retry attempts per generate() call
_MAX_RETRY_WAIT = 900        # seconds — cap the Groq-reported wait at 15 min
_MIN_RETRY_WAIT = 10         # seconds — floor so we never hammer the API


def _parse_retry_after(message: str) -> float:
    """Extract wait seconds from Groq error message like 'try again in 10m26.4s'."""
    m = re.search(r"try again in\s+(?:(\d+)m)?(\d+(?:\.\d+)?)s", message, re.IGNORECASE)
    if not m:
        return _MIN_RETRY_WAIT
    minutes = float(m.group(1) or 0)
    seconds = float(m.group(2) or 0)
    return max(_MIN_RETRY_WAIT, min(_MAX_RETRY_WAIT, minutes * 60 + seconds))


# ── Shared interface ─────────────────────────────────────────────────────────

class BaseLLM:
    """Minimal interface every LLM backend must implement."""

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> str:
        raise NotImplementedError


# ── Groq text-generation client ───────────────────────────────────────────────

class GroqLLM(BaseLLM):
    """Async wrapper around Groq chat completions (sync SDK via executor).

    Parameters
    ----------
    max_retries:
        How many times to retry on RateLimitError before raising.
        Default ``_MAX_RETRIES`` (5) — suitable for standalone use where
        sleeping is acceptable.  Pass ``1`` inside ``FallbackLLM`` so the
        first 429 immediately propagates to the fallback without sleeping.
    """

    def __init__(self, api_key: str, default_model: str, max_retries: int = _MAX_RETRIES):
        from groq import Groq
        self._client = Groq(api_key=api_key)
        self._default_model = default_model
        self._max_retries = max_retries

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> str:
        from groq import RateLimitError

        model = model or self._default_model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        loop = asyncio.get_running_loop()
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: self._client.chat.completions.create(**kwargs),
                )
                return response.choices[0].message.content or ""

            except RateLimitError as exc:
                wait = _parse_retry_after(str(exc))
                log.warning(
                    "llm_client.rate_limit",
                    attempt=attempt,
                    max_retries=self._max_retries,
                    wait_seconds=wait,
                    model=model,
                )
                last_exc = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(wait)
                # else fall through to re-raise

        raise last_exc  # type: ignore[misc]


# ── Gemini text-generation client (fallback) ─────────────────────────────────

class GeminiLLM(BaseLLM):
    """Async wrapper around Gemini generateContent — used as Groq rate-limit fallback.

    Uses the same google-genai SDK and API key already wired for embeddings.
    Supports JSON mode via ``response_mime_type="application/json"``.
    Free tier: 1M tokens/day (10× Groq free tier) — enough for the full corpus.

    Retries on 429 (quota) and 503 (overload) up to ``max_retries`` times,
    honouring the ``retryDelay`` from the error when present.
    """

    _MAX_RETRIES = 5
    _MIN_WAIT    = 10.0   # seconds
    _MAX_WAIT    = 120.0  # seconds

    def __init__(self, api_key: str, default_model: str):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._default_model = default_model

    @staticmethod
    def _parse_retry_delay(message: str) -> float:
        """Extract seconds from Gemini error like 'retryDelay: 18s' or 'retry in Xs'."""
        m = re.search(r"(?:retryDelay['\"]?\s*:\s*['\"]?|retry in\s+)(\d+(?:\.\d+)?)s",
                      message, re.IGNORECASE)
        if m:
            return max(GeminiLLM._MIN_WAIT, min(GeminiLLM._MAX_WAIT, float(m.group(1))))
        return GeminiLLM._MIN_WAIT

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> str:
        from google.genai import types as genai_types
        from google.genai.errors import ClientError, ServerError

        model = model or self._default_model
        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain",
        )

        loop = asyncio.get_running_loop()
        last_exc: Exception | None = None

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: self._client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    ),
                )
                return response.text or ""

            except (ClientError, ServerError) as exc:
                status = getattr(exc, 'status_code', 0) or 0
                if status in (429, 503):
                    wait = self._parse_retry_delay(str(exc))
                    log.warning(
                        "llm_client.gemini_rate_limit",
                        attempt=attempt,
                        max_retries=self._MAX_RETRIES,
                        wait_seconds=wait,
                        model=model,
                        status=status,
                    )
                    last_exc = exc
                    if attempt < self._MAX_RETRIES:
                        await asyncio.sleep(wait)
                else:
                    raise

        raise last_exc  # type: ignore[misc]


# ── DeepSeek text-generation client ──────────────────────────────────────────

class DeepSeekLLM(BaseLLM):
    """Async wrapper around DeepSeek chat completions via OpenAI-compatible API.

    DeepSeek-V3 supports JSON mode, has generous rate limits, and costs
    ~$0.07/1M input tokens — used as the Groq rate-limit fallback.
    """

    _BASE_URL     = "https://api.deepseek.com"
    _DEFAULT_MODEL = "deepseek-chat"   # DeepSeek-V3
    _MAX_RETRIES  = 3
    _RETRY_WAIT   = 10.0  # seconds between retries on 429/503

    def __init__(self, api_key: str, default_model: str = _DEFAULT_MODEL):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=self._BASE_URL)
        self._default_model = default_model

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> str:
        from openai import RateLimitError, APIStatusError

        model = model or self._default_model
        kwargs: dict[str, Any] = {
            "model":    model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        loop = asyncio.get_running_loop()
        last_exc: Exception | None = None

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: self._client.chat.completions.create(**kwargs),
                )
                return response.choices[0].message.content or ""

            except (RateLimitError, APIStatusError) as exc:
                log.warning(
                    "llm_client.deepseek_rate_limit",
                    attempt=attempt,
                    max_retries=self._MAX_RETRIES,
                    wait_seconds=self._RETRY_WAIT,
                    model=model,
                )
                last_exc = exc
                if attempt < self._MAX_RETRIES:
                    await asyncio.sleep(self._RETRY_WAIT)

        raise last_exc  # type: ignore[misc]


# ── Fallback LLM — Groq primary, Gemini on rate-limit exhaustion ──────────────

class FallbackLLM(BaseLLM):
    """Groq primary (one attempt, fail-fast) → DeepSeek fallback on rate limit.

    The Groq instance inside uses ``max_retries=1`` so the first 429 raises
    immediately without sleeping.  ``FallbackLLM`` catches that and issues the
    same call to DeepSeek-V3 instead — no delay, no sleeping.

    - Normal operation : Groq handles the call (~280 tok/s).
    - Groq rate-limited: DeepSeek handles it instantly, generous rate limits.
    - Callers see no difference — same ``generate()`` signature.
    """

    def __init__(self, api_key_groq: str, default_model_groq: str,
                 api_key_deepseek: str):
        # fail-fast Groq: one attempt only, no sleep on 429
        self._groq = GroqLLM(
            api_key=api_key_groq,
            default_model=default_model_groq,
            max_retries=1,
        )
        self._deepseek = DeepSeekLLM(api_key=api_key_deepseek)

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> str:
        from groq import RateLimitError

        try:
            return await self._groq.generate(
                prompt, model=model, json_mode=json_mode, temperature=temperature
            )
        except RateLimitError:
            log.warning(
                "llm_client.fallback_to_deepseek",
                reason="groq_rate_limit",
            )
            return await self._deepseek.generate(
                prompt, json_mode=json_mode, temperature=temperature
            )


# ── OpenAI embedding client ───────────────────────────────────────────────────

class OpenAIEmbedder:
    """Async wrapper around OpenAI text-embedding-3-large (3072d).

    Drop-in replacement for GeminiEmbedder — same dimensions, same interface.
    Uses the openai SDK already installed in the project.
    Cost: ~$0.13/1M tokens (~$0.001 for the full 12-doc corpus).
    """

    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.embeddings.create(
                model=self._model,
                input=texts,
            ),
        )
        return [item.embedding for item in response.data]

    async def embed_text(self, text: str, task_type: str = "retrieval_document") -> list[float]:
        results = await self.embed([text])
        return results[0]


# ── Gemini embedding client (kept for reference) ──────────────────────────────

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

_llm:      BaseLLM | None = None
_fast_llm: FallbackLLM | None = None
_embedder: OpenAIEmbedder | None = None


def get_llm() -> BaseLLM:
    """Return the primary (large) LLM — llama-3.3-70b via Groq, DeepSeek-V3 fallback.

    Normal path:   Groq llama-3.3-70b (~280 tok/s, lowest latency).
    Fallback path: DeepSeek-V3 on first Groq 429 — no sleep, generous limits.

    One-shot override — ``LLM_INGEST_PROVIDER=deepseek``:
        Bypasses Groq/FallbackLLM entirely and returns a bare ``DeepSeekLLM``.
        For populating the deterministic-ingestion cache (``LLM_CACHE_ENABLED=1``)
        in a single clean pass: one provider's extraction "voice" for the whole
        corpus, no Groq daily-cap stalls, ~$0.07/1M input tokens. Unset this
        env var afterwards — it's a one-shot knob, not a permanent switch.
    """
    global _llm
    if _llm is None:
        from graphrag.core.config import get_settings
        cfg = get_settings()
        if cfg.llm_ingest_provider == "deepseek":
            log.warning(
                "llm_client.single_provider_override",
                provider="deepseek",
                reason="LLM_INGEST_PROVIDER=deepseek — Groq bypassed for this run",
            )
            _llm = DeepSeekLLM(api_key=cfg.deepseek_api_key)
        else:
            _llm = FallbackLLM(
                api_key_groq=cfg.groq_api_key,
                default_model_groq=cfg.groq_model,
                api_key_deepseek=cfg.deepseek_api_key,
            )
    return _llm


def get_fast_llm() -> FallbackLLM:
    """Return the fast (small) LLM — llama-3.1-8b-instant via Groq, DeepSeek fallback.

    Used by the agentic retriever for intermediate SEARCH/ANSWER decisions.
    At ~800 tok/s on Groq vs ~150 tok/s for 70B, each reasoning step costs
    ~0.2s instead of ~1.5s. The final synthesis always uses the 70B model.
    """
    global _fast_llm
    if _fast_llm is None:
        from graphrag.core.config import get_settings
        cfg = get_settings()
        _fast_llm = FallbackLLM(
            api_key_groq=cfg.groq_api_key,
            default_model_groq=cfg.groq_fast_model,
            api_key_deepseek=cfg.deepseek_api_key,
        )
    return _fast_llm


def get_embedder() -> OpenAIEmbedder:
    """Return the embedder — OpenAI text-embedding-3-large (3072d).

    Replaces GeminiEmbedder; same vector dimensions so the Neo4j schema
    and all retrieval queries are unaffected.
    """
    global _embedder
    if _embedder is None:
        from graphrag.core.config import get_settings
        cfg = get_settings()
        _embedder = OpenAIEmbedder(
            api_key=cfg.openai_api_key,
            model=cfg.openai_embed_model,
        )
    return _embedder
