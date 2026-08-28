"""Conservative trace sanitization before a trajectory leaves the source project."""

from __future__ import annotations

import re
from typing import Any

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d().\-\s]{6,}\d)(?!\w)")
_API_KEY = re.compile(r"\b(?:AIza[\w-]{20,}|sk-[\w-]{16,}|[A-Za-z0-9_-]{32,})\b")


def sanitize_text(value: str) -> tuple[str, bool]:
    """Replace common direct identifiers and credential-like strings.

    This is intentionally conservative, not a claim of full PII detection.
    """
    sanitized = _EMAIL.sub("[REDACTED_EMAIL]", value)
    sanitized = _PHONE.sub("[REDACTED_PHONE]", sanitized)
    sanitized = _API_KEY.sub("[REDACTED_SECRET]", sanitized)
    return sanitized, sanitized != value


def sanitize_value(value: Any) -> tuple[Any, bool]:
    """Recursively sanitize JSON-compatible trace fields without dropping keys."""
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, list):
        changed = False
        out: list[Any] = []
        for item in value:
            safe, item_changed = sanitize_value(item)
            out.append(safe)
            changed = changed or item_changed
        return out, changed
    if isinstance(value, dict):
        changed = False
        out: dict[str, Any] = {}
        for key, item in value.items():
            safe, item_changed = sanitize_value(item)
            out[str(key)] = safe
            changed = changed or item_changed
        return out, changed
    return value, False
