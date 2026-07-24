"""Unit tests for graphrag.core.provider_health — the LLM circuit-breaker state."""

from __future__ import annotations

import pytest

from graphrag.core import provider_health as ph


@pytest.fixture(autouse=True)
def _reset_provider_health():
    """The module holds global state — clear it before and after every test
    so test order/isolation doesn't leak between test functions."""
    ph.reset()
    yield
    ph.reset()


class TestUnseenProvider:
    def test_unseen_provider_is_healthy(self):
        assert ph.is_healthy("nonexistent") is True

    def test_stays_healthy_below_min_samples(self):
        # _MIN_SAMPLES = 3 — two failures alone shouldn't trip anything
        ph.record_result("deepseek", False)
        ph.record_result("deepseek", False)
        assert ph.is_healthy("deepseek") is True


class TestConsecutiveFailureTrip:
    def test_three_consecutive_failures_trips_unhealthy(self):
        ph.record_result("deepseek", False)
        ph.record_result("deepseek", False)
        ph.record_result("deepseek", False)
        assert ph.is_healthy("deepseek") is False

    def test_success_after_trip_recovers_immediately(self):
        ph.record_result("deepseek", False)
        ph.record_result("deepseek", False)
        ph.record_result("deepseek", False)
        assert ph.is_healthy("deepseek") is False

        ph.record_result("deepseek", True)
        assert ph.is_healthy("deepseek") is True

    def test_streak_broken_by_success_does_not_trip(self):
        # Max run length is 2 (never reaches the 3-streak trip), and overall
        # failure rate is 4/8 = 50%, well under the 80% rate threshold.
        for outcome in [False, False, True, False, False, True, True, True]:
            ph.record_result("deepseek", outcome)
        assert ph.is_healthy("deepseek") is True


class TestFailureRateTrip:
    def test_high_failure_rate_without_streak_trips_unhealthy(self):
        # 17 failures then 3 successes (window size 20): rate = 17/20 = 85%,
        # above the 80% threshold. Trailing successes clear the consecutive-
        # failure streak, so this trips on RATE, not the streak condition.
        for _ in range(17):
            ph.record_result("groq", False)
        for _ in range(3):
            ph.record_result("groq", True)
        assert ph.is_healthy("groq") is False

    def test_failure_rate_below_threshold_stays_healthy(self):
        # 5 failures / 20 calls = 25%, well under 80%. Failures interspersed
        # so no run ever reaches the 3-consecutive streak trip either.
        for i in range(20):
            ph.record_result("groq", i % 4 != 0)  # False on i=0,4,8,12,16 → 5 failures
        assert ph.is_healthy("groq") is True

    def test_window_is_bounded_stale_failures_age_out(self):
        # Fill the window with failures, then push enough successes through
        # to fully evict them (window size 20) — should recover even without
        # ever recording an explicit "reset".
        for _ in range(20):
            ph.record_result("deepseek", False)
        assert ph.is_healthy("deepseek") is False

        for _ in range(20):
            ph.record_result("deepseek", True)
        assert ph.is_healthy("deepseek") is True


class TestProviderIsolation:
    def test_providers_tracked_independently(self):
        ph.record_result("deepseek", False)
        ph.record_result("deepseek", False)
        ph.record_result("deepseek", False)
        assert ph.is_healthy("deepseek") is False
        assert ph.is_healthy("groq") is True


class TestReset:
    def test_reset_single_provider_clears_only_that_provider(self):
        ph.record_result("deepseek", False)
        ph.record_result("deepseek", False)
        ph.record_result("deepseek", False)
        ph.record_result("groq", False)
        ph.record_result("groq", False)
        ph.record_result("groq", False)

        ph.reset("deepseek")
        assert ph.is_healthy("deepseek") is True
        assert ph.is_healthy("groq") is False

    def test_reset_all_clears_every_provider(self):
        ph.record_result("deepseek", False)
        ph.record_result("deepseek", False)
        ph.record_result("deepseek", False)
        ph.record_result("groq", False)
        ph.record_result("groq", False)
        ph.record_result("groq", False)

        ph.reset()
        assert ph.is_healthy("deepseek") is True
        assert ph.is_healthy("groq") is True
