"""Unit tests for graphrag.core.retry — exponential-backoff retry decorator."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, call

import pytest

from graphrag.core.retry import with_retry


# ── helpers ────────────────────────────────────────────────────────────────────

class _Transient(Exception):
    """Stand-in for a transient infrastructure error."""


class _Fatal(Exception):
    """Stand-in for a programming error that should NOT be retried."""


# ── tests ──────────────────────────────────────────────────────────────────────

async def test_succeeds_on_first_attempt():
    """No-failure path: function returns immediately without any sleep."""
    calls = []

    @with_retry(exceptions=(_Transient,), max_attempts=3)
    async def fn():
        calls.append(1)
        return "ok"

    result = await fn()
    assert result == "ok"
    assert len(calls) == 1


async def test_retries_and_succeeds():
    """Fails once, succeeds on second attempt."""
    attempt = [0]

    @with_retry(exceptions=(_Transient,), max_attempts=3, base_delay_s=0.01, jitter=False)
    async def fn():
        attempt[0] += 1
        if attempt[0] < 2:
            raise _Transient("boom")
        return "recovered"

    result = await fn()
    assert result == "recovered"
    assert attempt[0] == 2


async def test_exhausts_attempts_and_raises():
    """Always fails → raises the last exception after max_attempts."""

    @with_retry(exceptions=(_Transient,), max_attempts=3, base_delay_s=0.01, jitter=False)
    async def fn():
        raise _Transient("persistent")

    with pytest.raises(_Transient, match="persistent"):
        await fn()


async def test_non_retryable_exception_propagates_immediately():
    """Fatal errors that are not in the exceptions tuple propagate on first raise."""
    attempts = [0]

    @with_retry(exceptions=(_Transient,), max_attempts=5, base_delay_s=0.01)
    async def fn():
        attempts[0] += 1
        raise _Fatal("bug")

    with pytest.raises(_Fatal):
        await fn()

    assert attempts[0] == 1   # no retries


async def test_max_attempts_one_means_no_retry():
    """max_attempts=1 means exactly one call, no retry on failure."""
    calls = [0]

    @with_retry(exceptions=(_Transient,), max_attempts=1, base_delay_s=0.01)
    async def fn():
        calls[0] += 1
        raise _Transient("once")

    with pytest.raises(_Transient):
        await fn()

    assert calls[0] == 1


async def test_invalid_max_attempts_raises_at_decoration_time():
    """max_attempts < 1 should raise ValueError immediately."""
    with pytest.raises(ValueError, match="max_attempts"):
        @with_retry(exceptions=(_Transient,), max_attempts=0)
        async def fn():
            pass


async def test_return_value_preserved_after_retry():
    """Return value of the wrapped function is passed through unchanged."""
    call_count = [0]

    @with_retry(exceptions=(_Transient,), max_attempts=3, base_delay_s=0.01, jitter=False)
    async def fn() -> dict:
        call_count[0] += 1
        if call_count[0] < 3:
            raise _Transient("not yet")
        return {"answer": 42}

    result = await fn()
    assert result == {"answer": 42}
    assert call_count[0] == 3


async def test_jitter_does_not_block_indefinitely():
    """With jitter=True the function still completes in reasonable time."""

    @with_retry(exceptions=(_Transient,), max_attempts=3, base_delay_s=0.001, jitter=True)
    async def fn():
        raise _Transient("jitter")

    with pytest.raises(_Transient):
        await asyncio.wait_for(fn(), timeout=5.0)


async def test_backoff_grows_between_retries(monkeypatch):
    """Each retry delay should be larger than the previous (before jitter/cap)."""
    slept: list[float] = []

    async def fake_sleep(s: float):
        slept.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    @with_retry(exceptions=(_Transient,), max_attempts=4, base_delay_s=0.1,
                backoff=2.0, max_delay_s=100.0, jitter=False)
    async def fn():
        raise _Transient("grow")

    with pytest.raises(_Transient):
        await fn()

    # 3 sleeps for 4 attempts (no sleep after last attempt)
    assert len(slept) == 3
    assert slept[0] == pytest.approx(0.1)
    assert slept[1] == pytest.approx(0.2)
    assert slept[2] == pytest.approx(0.4)
