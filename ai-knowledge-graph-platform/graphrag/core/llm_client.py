"""Central LLM client — routes text generation to Groq with DeepSeek fallback;
embeddings use OpenAI text-embedding-3-large (3072d).

Usage:
    from graphrag.core.llm_client import get_llm, get_embedder

    text  = await get_llm().generate(prompt, json_mode=True)
    vecs  = await get_embedder().embed(texts)

Provider strategy
-----------------
Primary:  DeepSeek-V4 (deepseek-v4-pro) via ``FallbackLLM.deepseek_primary()``
          — single provider for the full ingestion run gives a consistent
          extraction "voice" (entity/relation style, confidence calibration)
          across the whole graph — important for reproducibility. Falls over
          to Groq transparently on rate-limit, timeout, or an API error (e.g.
          a bad/deprecated model id — see the 2026-07-24 incident note on
          ``DeepSeekLLM``).
Opt-in:   Groq (llama-3.3-70b-versatile) — set ``LLM_INGEST_PROVIDER=groq`` to
          use ``FallbackLLM.groq_primary()`` instead: Groq-primary with
          instant DeepSeek fallback on rate-limit, e.g. for quick
          low-volume/dev runs.
Embeddings: OpenAI text-embedding-3-large (3072d) — replaced Gemini; same
          dimensions, same Neo4j schema, no re-indexing required.

Rate-limit / failure handling
------------------------------
When Groq returns a 429, the error message contains the exact wait time in the
form "Please try again in XmY.Zs".  ``GroqLLM.generate()`` parses that value
and sleeps for that duration (capped at ``_MAX_RETRY_WAIT`` seconds) before
retrying.  After all retry attempts are exhausted the exception propagates to
``FallbackLLM``, which transparently re-issues the call to the secondary
provider instead.

Both ``GroqLLM`` and ``DeepSeekLLM`` also track recent success/failure via
``graphrag.core.provider_health`` — once a provider looks broken (see that
module for the exact trip conditions), retries drop to a single fail-fast
attempt instead of the full retry budget, so a sustained outage on the
primary fails over to the secondary quickly instead of burning the full
retry-and-sleep sequence on every call.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import structlog

from graphrag.core.provider_health import is_healthy, record_result

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

    _TIMEOUT = 60.0  # seconds — without this a stalled Groq response hangs forever
                      # and FallbackLLM never gets a chance to fall back.
    _PROVIDER_NAME = "groq"

    def __init__(self, api_key: str, default_model: str, max_retries: int = _MAX_RETRIES):
        from groq import Groq
        self._client = Groq(api_key=api_key, timeout=self._TIMEOUT)
        self._default_model = default_model
        self._max_retries = max_retries

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> str:
        from groq import RateLimitError, APITimeoutError, APIConnectionError

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

        # Fail fast once this provider looks broken — skip most of the retry
        # budget instead of re-attempting a call that's failed the last N
        # times in a row. Checked once per generate() call, not re-checked
        # mid-loop. See graphrag.core.provider_health.
        effective_max_retries = (
            self._max_retries if is_healthy(self._PROVIDER_NAME) else 1
        )

        for attempt in range(1, effective_max_retries + 1):
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: self._client.chat.completions.create(**kwargs),
                )
                record_result(self._PROVIDER_NAME, True)
                return response.choices[0].message.content or ""

            except RateLimitError as exc:
                record_result(self._PROVIDER_NAME, False)
                wait = _parse_retry_after(str(exc))
                log.warning(
                    "llm_client.rate_limit",
                    attempt=attempt,
                    max_retries=effective_max_retries,
                    wait_seconds=wait,
                    model=model,
                )
                last_exc = exc
                if attempt < effective_max_retries:
                    await asyncio.sleep(wait)
                # else fall through to re-raise

            except (APITimeoutError, APIConnectionError) as exc:
                record_result(self._PROVIDER_NAME, False)
                log.warning(
                    "llm_client.groq_timeout",
                    attempt=attempt,
                    max_retries=effective_max_retries,
                    model=model,
                    error=type(exc).__name__,
                )
                last_exc = exc
                if attempt < effective_max_retries:
                    await asyncio.sleep(_MIN_RETRY_WAIT)
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
    _DEFAULT_MODEL = "deepseek-v4-pro"   # was "deepseek-chat" — DeepSeek deprecated that
                                          # id; API now rejects it with 400 on every call
                                          # (found 2026-07-24 via worker.log retry/DLQ spam)
    _MAX_RETRIES  = 3
    _RETRY_WAIT   = 10.0  # seconds between retries on 429/503/timeout
    _TIMEOUT      = 60.0  # seconds — DeepSeek's API can stall under load with
                           # no error; without this the call hangs forever.
    _PROVIDER_NAME = "deepseek"

    def __init__(self, api_key: str, default_model: str = _DEFAULT_MODEL,
                 max_retries: int = _MAX_RETRIES):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=self._BASE_URL, timeout=self._TIMEOUT)
        self._default_model = default_model
        self._max_retries = max_retries

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> str:
        from openai import RateLimitError, APIStatusError, APITimeoutError, APIConnectionError

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

        # See GroqLLM.generate() — same fail-fast pattern once this provider
        # looks broken. This is the path that was silently retrying 3x10s per
        # call for ~40 minutes during the 2026-07-24 deepseek-chat incident.
        effective_max_retries = (
            self._max_retries if is_healthy(self._PROVIDER_NAME) else 1
        )

        for attempt in range(1, effective_max_retries + 1):
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: self._client.chat.completions.create(**kwargs),
                )
                record_result(self._PROVIDER_NAME, True)
                return response.choices[0].message.content or ""

            except (RateLimitError, APIStatusError, APITimeoutError, APIConnectionError) as exc:
                record_result(self._PROVIDER_NAME, False)
                log.warning(
                    "llm_client.deepseek_retry",
                    attempt=attempt,
                    max_retries=effective_max_retries,
                    wait_seconds=self._RETRY_WAIT,
                    model=model,
                    error=type(exc).__name__,
                )
                last_exc = exc
                if attempt < effective_max_retries:
                    await asyncio.sleep(self._RETRY_WAIT)

        raise last_exc  # type: ignore[misc]


# ── Fallback LLM — primary provider (fail-fast) → secondary on failure ───────

class FallbackLLM(BaseLLM):
    """Wraps a primary provider (one attempt, fail-fast) and a secondary
    provider used whenever the primary raises one of `fallback_exceptions`.

    Generalized 2026-07-24: previously hardcoded Groq-primary/DeepSeek-
    secondary only. The 2026-07-24 incident (DeepSeek's model id deprecated,
    ``get_llm()``'s default path had zero fallback) showed the primary
    generation path needs the same redundancy `get_fast_llm()` already had —
    so this now supports either direction via the two classmethods below,
    with one shared try/except implementation instead of a near-duplicate
    class.

    Build via ``FallbackLLM.groq_primary(cfg)`` or
    ``FallbackLLM.deepseek_primary(cfg)`` — don't call ``__init__`` directly
    unless you're constructing the provider instances yourself.
    """

    def __init__(self, primary: BaseLLM, primary_name: str, secondary: BaseLLM,
                 fallback_exceptions: tuple[type[Exception], ...]):
        self._primary = primary
        self._primary_name = primary_name
        self._secondary = secondary
        self._fallback_exceptions = fallback_exceptions

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> str:
        try:
            return await self._primary.generate(
                prompt, model=model, json_mode=json_mode, temperature=temperature
            )
        except self._fallback_exceptions as exc:
            log.warning(
                "llm_client.fallback",
                primary=self._primary_name,
                reason=type(exc).__name__,
            )
            return await self._secondary.generate(
                prompt, json_mode=json_mode, temperature=temperature
            )

    @classmethod
    def groq_primary(cls, cfg, model: str | None = None) -> "FallbackLLM":
        """Groq primary (~280 tok/s, fail-fast) -> DeepSeek on rate-limit/timeout.

        `model` overrides `cfg.groq_model` — used by `get_fast_llm()` to
        select the fast 8B model instead of the default 70B.
        """
        from groq import RateLimitError, APITimeoutError, APIConnectionError
        return cls(
            primary=GroqLLM(
                api_key=cfg.groq_api_key,
                default_model=model or cfg.groq_model,
                max_retries=1,  # fail fast: first 429 raises immediately, no sleep
            ),
            primary_name="groq",
            secondary=DeepSeekLLM(api_key=cfg.deepseek_api_key),
            fallback_exceptions=(RateLimitError, APITimeoutError, APIConnectionError),
        )

    @classmethod
    def deepseek_primary(cls, cfg) -> "FallbackLLM":
        """DeepSeek primary (fail-fast) -> Groq on rate-limit/timeout/API error.

        ``APIStatusError`` is included deliberately — that's the exact
        exception a 400 (e.g. an invalid/deprecated model id, as in the
        2026-07-24 incident) raises. Without it in this tuple, a repeat of
        that incident would still take down synthesis entirely instead of
        transparently failing over to Groq.
        """
        from openai import RateLimitError, APIStatusError, APITimeoutError, APIConnectionError
        return cls(
            primary=DeepSeekLLM(
                api_key=cfg.deepseek_api_key,
                max_retries=1,  # fail fast: same philosophy as groq_primary()
            ),
            primary_name="deepseek",
            secondary=GroqLLM(api_key=cfg.groq_api_key, default_model=cfg.groq_model),
            fallback_exceptions=(RateLimitError, APIStatusError, APITimeoutError, APIConnectionError),
        )


# ── OpenAI embedding client ───────────────────────────────────────────────────

class OpenAIEmbedder:
    """Async wrapper around OpenAI text-embedding-3-large (3072d).

    Drop-in replacement for GeminiEmbedder — same dimensions, same interface.
    Uses the openai SDK already installed in the project.
    Cost: ~$0.13/1M tokens (~$0.001 for the full 12-doc corpus).
    """

    _TIMEOUT = 60.0  # seconds — without this an embedding call can hang for
                      # 30+ minutes on a stalled connection (SDK default is
                      # 600s x retries with backoff). Mirrors GroqLLM/DeepSeekLLM.

    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, timeout=self._TIMEOUT)
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
    """Return the primary (large) LLM — DeepSeek-V4 primary, Groq fallback.

    Normal path: ``FallbackLLM.deepseek_primary()`` — DeepSeek (deepseek-v4-pro)
    handles the call for one provider's consistent extraction "voice" across
    the whole corpus; Groq is used transparently if DeepSeek fails (rate
    limit, timeout, or an API error like the 2026-07-24 incident where
    DeepSeek's model id was deprecated and every call 400'd). Before
    2026-07-24 this was a bare ``DeepSeekLLM`` with no fallback at all — that
    gap is why the incident took down synthesis for ~40 minutes.

    Opt-in override — ``LLM_INGEST_PROVIDER=groq``:
        ``FallbackLLM.groq_primary()`` — Groq llama-3.3-70b as primary
        (~280 tok/s) with instant DeepSeek fallback on rate-limit. Useful for
        quick/low-volume dev runs where Groq's free tier won't be exhausted.
    """
    global _llm
    if _llm is None:
        from graphrag.core.config import get_settings
        cfg = get_settings()
        if cfg.llm_ingest_provider == "groq":
            log.warning(
                "llm_client.single_provider_override",
                provider="groq",
                reason="LLM_INGEST_PROVIDER=groq — Groq-primary with DeepSeek fallback for this run",
            )
            _llm = FallbackLLM.groq_primary(cfg)
        else:
            _llm = FallbackLLM.deepseek_primary(cfg)
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
        _fast_llm = FallbackLLM.groq_primary(cfg, model=cfg.groq_fast_model)
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
