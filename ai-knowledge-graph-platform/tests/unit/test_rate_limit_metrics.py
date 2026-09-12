"""graphrag_rate_limit_rejected_total / graphrag_quota_rejected_total.

docs/performance-metrics-inventory.md named this exact gap: "no dedicated
Prometheus counter for 429 events exists yet". Same before/after-delta
pattern against the default registry as test_operational_metrics.py, so a
concurrent test bumping the same counter doesn't make this flaky.
"""
from __future__ import annotations

import pytest

from graphrag.observability import access_control_metrics as acm

prometheus_client = pytest.importorskip("prometheus_client")


def _sample(name: str, **labels) -> float:
    from prometheus_client import REGISTRY

    value = REGISTRY.get_sample_value(name, labels or None)
    return float(value) if value is not None else 0.0


class TestRateLimitRejectedCounter:
    def test_increments_for_the_rejected_route(self):
        before = _sample("graphrag_rate_limit_rejected_total", route="/query")
        acm.record_rate_limit_rejected("/query")
        assert _sample("graphrag_rate_limit_rejected_total", route="/query") == before + 1

    def test_is_scoped_per_route(self):
        before_a = _sample("graphrag_rate_limit_rejected_total", route="/search")
        before_b = _sample("graphrag_rate_limit_rejected_total", route="/ingest")
        acm.record_rate_limit_rejected("/search")
        assert _sample("graphrag_rate_limit_rejected_total", route="/search") == before_a + 1
        assert _sample("graphrag_rate_limit_rejected_total", route="/ingest") == before_b


class TestQuotaRejectedCounter:
    def test_increments_for_tenant_and_dimension(self):
        before = _sample("graphrag_quota_rejected_total", tenant="aerospace", dimension="requests")
        acm.record_quota_rejected("aerospace", "requests")
        assert (
            _sample("graphrag_quota_rejected_total", tenant="aerospace", dimension="requests")
            == before + 1
        )

    def test_is_scoped_per_dimension(self):
        before_requests = _sample("graphrag_quota_rejected_total", tenant="automotive", dimension="requests")
        before_cost = _sample("graphrag_quota_rejected_total", tenant="automotive", dimension="cost_usd")
        acm.record_quota_rejected("automotive", "cost_usd")
        assert (
            _sample("graphrag_quota_rejected_total", tenant="automotive", dimension="requests")
            == before_requests
        )
        assert (
            _sample("graphrag_quota_rejected_total", tenant="automotive", dimension="cost_usd")
            == before_cost + 1
        )
