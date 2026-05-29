"""Shared utilities for LLM response handling.

Centralises the `safe_response_text()` guard that all Gemini call sites need.

Problem solved
--------------
Every `response.text.strip()` in the codebase crashes with `AttributeError`
when Gemini blocks a response (safety filter) or returns no candidates —
`response.text` is `None` in those cases.  In the hot query path this
surfaces as an opaque 500 that gets retried 3× by the RabbitMQ consumer
before dead-lettering, even though the root cause is a content policy block
that will never succeed on retry.

`safe_response_text()` converts the crash into a controlled empty-or-default
return so every caller can decide how to degrade gracefully.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)


def safe_response_text(response: object, default: str = "") -> str:
    """Return the stripped text of a Gemini response, or *default* if blocked.

    Gemini sets ``response.text = None`` when:
    - The safety filter blocks the content.
    - The model returns no candidates (e.g. empty prompt, quota exceeded).

    Accessing ``.strip()`` on ``None`` raises ``AttributeError``, crashing the
    entire call site.  This helper converts the crash into a predictable return
    value so callers can degrade gracefully (log a warning, fall back to
    agentic retrieval, etc.).

    Parameters
    ----------
    response : Any Gemini GenerateContentResponse object (or None).
    default  : Value to return when ``response.text`` is None or empty.

    Returns
    -------
    Stripped response text, or *default*.
    """
    try:
        text = getattr(response, "text", None)
        if text:
            return text.strip()
        log.warning(
            "llm_utils.empty_response",
            response_type=type(response).__name__,
            hint="Gemini may have blocked the content or returned no candidates",
        )
        return default
    except AttributeError:
        log.warning("llm_utils.text_attr_missing", response_type=type(response).__name__)
        return default
