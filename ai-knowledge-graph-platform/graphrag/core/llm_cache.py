"""Content-addressable disk cache for LLM extraction calls.

Why this exists
---------------
Groq and DeepSeek are NOT deterministic at ``temperature=0.0`` — batched
GPU/LPU inference means floating-point summation order (and therefore token
selection) can vary run to run for an identical prompt. Layer the Groq→DeepSeek
rate-limit fallback on top (two different model families extracting different
chunks of the same corpus) and every ``--wipe --commit`` re-ingest of the same
corpus produces a different entity/edge/conflict/community shape.

This cache memoizes the raw LLM response by a hash of
``(model, temperature, json_mode, prompt)``. On a hit, the LLM is never
called — the exact same response is replayed, so re-ingesting the same corpus
produces a byte-identical graph every time. This is what makes a "live demo"
script's hardcoded examples (specific entity names, confidence values,
inferred edges) safe to present without re-verifying before every run.

Scope
-----
Wired ONLY into ``Extractor.extract()`` — the single call site that determines
graph shape. Retrieval/query/synthesis LLM calls are deliberately NOT cached;
those must stay live (a cached query answer would be stale and misleading).

Enable with ``LLM_CACHE_ENABLED=1`` (or ``llm_cache_enabled: true`` in
``config/settings.yml``). Off by default — production ingestion of new
documents should always hit the live LLM.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "llm_extraction"


class LLMCache:
    """Disk-backed content-addressable cache — one JSON file per (model, prompt) pair."""

    def __init__(self, cache_dir: Path | None = None):
        self._dir = cache_dir or _CACHE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(model: str, temperature: float, json_mode: bool, prompt: str) -> str:
        """Hash the full call signature — any change invalidates the entry.

        ``model`` here is the *configured* model name (e.g. ``llama-3.3-70b-versatile``),
        not necessarily the model that actually served the original request (Groq vs.
        DeepSeek fallback are interchangeable from the cache's point of view — we want
        to replay whatever was returned the first time, regardless of which provider
        served it, because that's what makes the graph shape reproducible).
        """
        payload = f"{model}|{temperature}|{json_mode}|{prompt}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, *, model: str, temperature: float, json_mode: bool, prompt: str) -> str | None:
        """Return the cached response, or ``None`` on a miss / corrupt entry."""
        key = self._key(model, temperature, json_mode, prompt)
        path = self._path(key)
        if not path.exists():
            self._misses += 1
            return None
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            response = cached["response"]
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            log.warning("llm_cache.read_failed", key=key[:12], error=str(exc))
            self._misses += 1
            return None
        self._hits += 1
        log.debug("llm_cache.hit", key=key[:12])
        return response

    def set(self, *, model: str, temperature: float, json_mode: bool,
            prompt: str, response: str) -> None:
        """Persist a fresh LLM response so future runs can replay it."""
        key = self._key(model, temperature, json_mode, prompt)
        path = self._path(key)
        try:
            path.write_text(
                json.dumps({
                    "model": model,
                    "temperature": temperature,
                    "json_mode": json_mode,
                    "response": response,
                }),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("llm_cache.write_failed", key=key[:12], error=str(exc))

    @property
    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses, "dir": str(self._dir)}


_cache: LLMCache | None = None


def get_llm_cache() -> LLMCache:
    """Process-wide singleton — shared across all `Extractor` instances."""
    global _cache
    if _cache is None:
        _cache = LLMCache()
    return _cache
