"""Async retry decorator for transient infrastructure failures.

Problem solved
--------------
Neo4j, Redis, and Wikidata API calls can fail transiently (connection
resets, leader elections, rate limits).  Without retries these failures
surface as 500s to the user even though a second attempt would succeed.

Architecture
------------
``with_retry`` is a decorator factory that wraps any async function with
exponential-backoff retry logic.  It catches only the exception types
passed in ``exceptions``, so real programming errors (KeyError, TypeError)
still propagate immediately.

Usage::

    from graphrag.core.retry import with_retry
    from neo4j.exceptions import TransientError, ServiceUnavailable

    class Neo4jClient:
        @with_retry(exceptions=(TransientError, ServiceUnavailable), max_attempts=3)
        async def run(self, cypher: str, **params) -> list[dict]:
            ...

Configuration
-------------
``max_attempts``:  total attempts including the first (default 3)
``base_delay_s``:  initial backoff in seconds (default 0.5)
``max_delay_s``:   cap on any single sleep (default 8.0)
``backoff``:       multiplier per attempt (default 2.0 — exponential)
``jitter``:        add ±20 % random jitter to avoid thundering herd (default True)
"""

from __future__ import annotations

import asyncio
import functools
import random
from collections.abc import Callable
from typing import Any, TypeVar

import structlog

log = structlog.get_logger(__name__)

F = TypeVar("F")


def with_retry(
    exceptions: tuple[type[Exception], ...],
    *,
    max_attempts: int = 3,
    base_delay_s: float = 0.5,
    max_delay_s: float = 8.0,
    backoff: float = 2.0,
    jitter: bool = True,
) -> Callable[[Any], Any]:
    """Decorator factory — wrap an *async* function with exponential-backoff retry.

    Parameters
    ----------
    exceptions:
        Tuple of exception classes that should trigger a retry.  All other
        exceptions propagate immediately.
    max_attempts:
        Maximum total call attempts (first attempt counts as attempt 1).
    base_delay_s:
        Delay before the *second* attempt, in seconds.
    max_delay_s:
        Upper bound on any single retry delay.
    backoff:
        Multiplicative factor applied after each failure (default 2 → exponential).
    jitter:
        When True, multiply each delay by a uniform random factor in [0.8, 1.2]
        to spread retries under thundering-herd conditions.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay_s
            last_exc: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    actual_delay = min(delay, max_delay_s)
                    if jitter:
                        actual_delay *= random.uniform(0.8, 1.2)
                    log.warning(
                        "retry.transient_failure",
                        fn=fn.__qualname__,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay_s=round(actual_delay, 3),
                        error=str(exc),
                    )
                    await asyncio.sleep(actual_delay)
                    delay = min(delay * backoff, max_delay_s)

            log.error(
                "retry.exhausted",
                fn=fn.__qualname__,
                attempts=max_attempts,
                error=str(last_exc),
            )
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
