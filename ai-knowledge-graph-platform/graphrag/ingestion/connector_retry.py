"""Shared bounded-retry helper for HTTP source connectors.

Honors a server-sent `Retry-After` header (seconds or HTTP-date form, RFC
9110 §10.2.3) on a retryable response; falls back to exponential backoff
with jitter otherwise. Used by `graphrag/ingestion/sap_source.py`,
`graphrag/ingestion/snowflake_source.py`, and the hardened
`graphrag/enterprise/sharepoint.py` -- three connectors that each need the
same throttling behavior, rather than three subtly different inline copies
of it.

`graphrag/ingestion/http_source.py`'s own simpler linear backoff is
deliberately left as-is: it works, it's already tested, and refactoring a
shipped module to adopt this is noted as documented follow-up, not done
here (minimal-impact discipline).
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

_DEFAULT_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class RetryExhaustedError(RuntimeError):
    """Raised when every retry attempt still returned a retryable status."""


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse a `Retry-After` header (RFC 9110 §10.2.3): either an integer
    number of seconds, or an HTTP-date. Returns None when the header is
    absent or unparseable -- the caller falls back to backoff+jitter."""
    header = response.headers.get("Retry-After")
    if not header:
        return None
    header = header.strip()
    try:
        return max(0.0, float(header))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(header)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


async def send_with_retry(
    send: Callable[[], Awaitable[httpx.Response]],
    *,
    max_retries: int,
    base_backoff_seconds: float,
    retryable_statuses: frozenset[int] = _DEFAULT_RETRYABLE_STATUSES,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    jitter: Callable[[], float] = random.random,
) -> httpx.Response:
    """Call `send()` up to `max_retries` times.

    A response whose status is in `retryable_statuses` is retried, honoring
    `Retry-After` when the server sent one, else exponential backoff with
    jitter (`base_backoff_seconds * 2**attempt * (1 + jitter())`). Any other
    status -- success or a non-retryable error -- returns/raises immediately
    via `response.raise_for_status()`, so a plain 4xx is never silently
    retried the way a 429 is. Raises `RetryExhaustedError` if the final
    attempt is still retryable.

    `sleep`/`jitter` are injectable so tests can assert retry *happened*
    (call count, and the exact wait when `Retry-After` is honored) without
    real delays or nondeterministic timing.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    last_status: int | None = None
    for attempt in range(max_retries):
        response = await send()
        if response.status_code not in retryable_statuses:
            response.raise_for_status()
            return response
        last_status = response.status_code
        if attempt + 1 >= max_retries:
            break
        wait = _retry_after_seconds(response)
        if wait is None:
            wait = base_backoff_seconds * (2**attempt) * (1 + jitter())
        await sleep(wait)
    raise RetryExhaustedError(
        f"gave up after {max_retries} attempt(s); last status was {last_status}"
    )


__all__ = ["RetryExhaustedError", "send_with_retry"]
