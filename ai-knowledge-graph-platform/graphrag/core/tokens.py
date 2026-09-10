"""Token estimation for context-size accounting.

Why an estimate rather than the provider's own count
----------------------------------------------------
Provider-reported usage (see `graphrag/observability/genai_telemetry.py`) is
authoritative but arrives *after* the call — too late to decide what to put in
the prompt, and attributed to the whole request rather than to the individual
sections that made it up. This module answers the question the telemetry
cannot: how many tokens is each part of the assembled context costing, before
it is sent.

`tiktoken` is used when importable and its encoding is cached, since building
one is expensive relative to a per-request call. When it is unavailable the
fallback is a character-count heuristic, which is deliberately crude: these
numbers drive a composition *metric* and an opt-in budget, never a hard
provider limit, so a consistent estimator matters more than an exact one. The
degradation is logged once rather than silently, because a fleet reporting
heuristic numbers while an operator believes they are exact is worse than one
reporting nothing.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)

# cl100k_base is the encoding used by GPT-4/3.5 and is close enough to
# Anthropic's tokenizer for the ratios this module exists to report. It is not
# claimed to be exact for any specific model.
_ENCODING_NAME = "cl100k_base"

# Average characters per token for English prose. Only used when tiktoken is
# unavailable.
_CHARS_PER_TOKEN = 4

_encoding: object | None = None
_encoding_loaded = False
_fallback_warned = False


def _get_encoding() -> object | None:
    """Return a cached tiktoken encoding, or None when unavailable."""
    global _encoding, _encoding_loaded, _fallback_warned
    if _encoding_loaded:
        return _encoding
    _encoding_loaded = True
    try:
        import tiktoken

        _encoding = tiktoken.get_encoding(_ENCODING_NAME)
    except Exception as exc:  # noqa: BLE001 - any failure means "use the heuristic"
        _encoding = None
        if not _fallback_warned:
            _fallback_warned = True
            log.warning(
                "tokens.tiktoken_unavailable",
                error=str(exc)[:200],
                impact=(
                    "context size is estimated from character count; "
                    "composition metrics remain comparable but are not exact"
                ),
            )
    return _encoding


def estimate_tokens(text: str) -> int:
    """Approximate the token count of `text`.

    Never raises: a counting failure must not be able to break the assembly of
    a prompt that would otherwise have been answered correctly.
    """
    if not text:
        return 0
    encoding = _get_encoding()
    if encoding is not None:
        try:
            return len(encoding.encode(text))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - fall through to the heuristic
            pass
    return max(1, len(text) // _CHARS_PER_TOKEN)


def uses_exact_tokenizer() -> bool:
    """True when counts come from tiktoken rather than the char heuristic.

    Exposed so a report can state which of the two it is showing instead of
    presenting a heuristic as a measurement.
    """
    return _get_encoding() is not None


__all__ = ["estimate_tokens", "uses_exact_tokenizer"]
