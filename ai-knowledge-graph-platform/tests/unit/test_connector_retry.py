"""graphrag/ingestion/connector_retry.py's send_with_retry() in isolation."""

from __future__ import annotations

import httpx
import pytest

from graphrag.ingestion.connector_retry import RetryExhaustedError, send_with_retry


def _response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    request = httpx.Request("GET", "https://example.test/x")
    return httpx.Response(status, headers=headers or {}, request=request)


class TestNonRetryableStatusPassesStraightThrough:
    async def test_success_returns_immediately_with_no_sleep(self):
        sleeps: list[float] = []

        async def sleep(seconds: float) -> None:
            sleeps.append(seconds)

        calls = 0

        async def send() -> httpx.Response:
            nonlocal calls
            calls += 1
            return _response(200)

        response = await send_with_retry(send, max_retries=3, base_backoff_seconds=0.01, sleep=sleep)
        assert response.status_code == 200
        assert calls == 1
        assert sleeps == []

    async def test_a_plain_4xx_is_not_retried(self):
        calls = 0

        async def send() -> httpx.Response:
            nonlocal calls
            calls += 1
            return _response(404)

        async def sleep(_seconds: float) -> None:
            raise AssertionError("must not sleep for a non-retryable status")

        with pytest.raises(httpx.HTTPStatusError):
            await send_with_retry(send, max_retries=3, base_backoff_seconds=0.01, sleep=sleep)
        assert calls == 1


class TestRetryAfterIsHonored:
    async def test_integer_seconds_form_is_honored_exactly(self):
        waits: list[float] = []

        async def sleep(seconds: float) -> None:
            waits.append(seconds)

        responses = [_response(429, {"Retry-After": "2"}), _response(200)]

        async def send() -> httpx.Response:
            return responses.pop(0)

        response = await send_with_retry(send, max_retries=3, base_backoff_seconds=0.01, sleep=sleep)
        assert response.status_code == 200
        assert waits == [2.0]

    async def test_http_date_form_is_honored_approximately(self):
        from datetime import datetime, timedelta, timezone
        from email.utils import format_datetime

        waits: list[float] = []

        async def sleep(seconds: float) -> None:
            waits.append(seconds)

        future = datetime.now(timezone.utc) + timedelta(seconds=5)
        responses = [_response(503, {"Retry-After": format_datetime(future, usegmt=True)}), _response(200)]

        async def send() -> httpx.Response:
            return responses.pop(0)

        await send_with_retry(send, max_retries=3, base_backoff_seconds=0.01, sleep=sleep)
        assert len(waits) == 1
        assert 3.0 <= waits[0] <= 5.5  # clock-skew tolerant, not exact


class TestBackoffFallsBackToExponentialWithJitter:
    async def test_no_retry_after_header_uses_backoff_and_jitter(self):
        waits: list[float] = []

        async def sleep(seconds: float) -> None:
            waits.append(seconds)

        responses = [_response(500), _response(500), _response(200)]

        async def send() -> httpx.Response:
            return responses.pop(0)

        await send_with_retry(
            send, max_retries=3, base_backoff_seconds=1.0, sleep=sleep, jitter=lambda: 0.0,
        )
        # attempt 0 -> wait 1.0*2**0*(1+0)=1.0; attempt 1 -> 1.0*2**1*(1+0)=2.0
        assert waits == [1.0, 2.0]

    async def test_jitter_is_applied_on_top_of_the_base_backoff(self):
        waits: list[float] = []

        async def sleep(seconds: float) -> None:
            waits.append(seconds)

        responses = [_response(500), _response(200)]

        async def send() -> httpx.Response:
            return responses.pop(0)

        await send_with_retry(
            send, max_retries=2, base_backoff_seconds=1.0, sleep=sleep, jitter=lambda: 0.5,
        )
        assert waits == [1.5]  # 1.0 * 2**0 * (1 + 0.5)


class TestExhaustion:
    async def test_raises_retry_exhausted_after_the_final_attempt_with_no_extra_sleep(self):
        waits: list[float] = []

        async def sleep(seconds: float) -> None:
            waits.append(seconds)

        calls = 0

        async def send() -> httpx.Response:
            nonlocal calls
            calls += 1
            return _response(503)

        with pytest.raises(RetryExhaustedError, match="3 attempt"):
            await send_with_retry(send, max_retries=3, base_backoff_seconds=0.01, sleep=sleep, jitter=lambda: 0.0)
        assert calls == 3
        assert len(waits) == 2  # slept between attempts, not after the last one

    async def test_max_retries_below_one_is_rejected(self):
        async def send() -> httpx.Response:
            raise AssertionError("must not be called")

        with pytest.raises(ValueError, match="max_retries"):
            await send_with_retry(send, max_retries=0, base_backoff_seconds=0.01)
