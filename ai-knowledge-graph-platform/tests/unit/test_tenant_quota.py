"""Per-tenant consumption quotas.

The failure this guards against is invisible to a rate limiter: one tenant
running steadily *just under* the rate limit consumes an entire day of shared
LLM spend. Every individual request is well-behaved, nothing throttles, and the
first signal is the provider bill or a quota exhaustion that takes every other
tenant down with it.

So the tests below care about three things a naive implementation gets wrong:
isolation between tenants, the fail-open direction when the counter store is
unreachable, and the fact that a quota is a *budget for a period* rather than a
rolling window.
"""

from __future__ import annotations


import pytest

from graphrag.core import tenant_quota as tq
from graphrag.core.tenant_quota import (
    UNLIMITED,
    QuotaBackendUnavailable,
    QuotaPolicy,
    TenantQuotaStore,
    policies_from_config,
)


async def _store(**kwargs) -> TenantQuotaStore:
    store = TenantQuotaStore(**kwargs)
    await store.connect()
    return store


class TestPolicyConfiguration:
    def test_absent_block_means_unlimited(self):
        # Adding the config block must not silently start throttling a
        # deployment that has not chosen numbers yet.
        tenants, default = policies_from_config(None)
        assert tenants == {}
        assert default.is_unlimited()

    def test_zero_is_the_unlimited_sentinel(self):
        _, default = policies_from_config({"default": {"max_requests": 0, "max_cost_usd": 0}})
        assert default.is_unlimited()

    def test_per_tenant_policy_overrides_the_default(self):
        tenants, default = policies_from_config({
            "default": {"max_requests": 100},
            "tenants": {"aerospace": {"max_requests": 5000}},
        })
        assert tenants["aerospace"].max_requests == 5000
        assert default.max_requests == 100

    def test_tenant_may_override_the_window_too(self):
        tenants, _ = policies_from_config({
            "window_seconds": 86400,
            "tenants": {"burst": {"window_seconds": 3600, "max_requests": 10}},
        })
        assert tenants["burst"].window_seconds == 3600

    def test_unknown_tenant_falls_back_to_default(self):
        store = TenantQuotaStore(
            policies={"a": QuotaPolicy(max_requests=1)},
            default_policy=QuotaPolicy(max_requests=99),
        )
        assert store.policy_for("somebody-else").max_requests == 99


class TestRequestQuota:
    async def test_requests_are_allowed_up_to_the_ceiling(self):
        store = await _store(default_policy=QuotaPolicy(max_requests=3))
        allowed = []
        for _ in range(5):
            verdict = await store.check("acme", additional_requests=1)
            allowed.append(verdict.allowed)
            if verdict.allowed:
                await store.consume("acme", requests=1)
        assert allowed == [True, True, True, False, False]

    async def test_rejection_names_the_dimension_and_the_reset(self):
        # A bare "quota exceeded" leaves the caller unable to tell a request
        # ceiling from a spend ceiling, or whether to retry in a minute or
        # tomorrow.
        store = await _store(default_policy=QuotaPolicy(max_requests=1))
        await store.consume("acme", requests=1)
        verdict = await store.check("acme", additional_requests=1)
        assert verdict.allowed is False
        assert verdict.dimension == "requests"
        assert verdict.limit == 1
        assert verdict.reset_after_seconds > 0
        detail = verdict.as_detail()
        assert detail["error"] == "tenant_quota_exceeded"
        assert detail["tenant"] == "acme"

    async def test_tenants_are_isolated(self):
        store = await _store(default_policy=QuotaPolicy(max_requests=1))
        await store.consume("acme", requests=1)
        # One tenant exhausting its budget must not throttle anyone else --
        # that would turn a billing control into a shared outage.
        assert (await store.check("other", additional_requests=1)).allowed is True

    async def test_unlimited_policy_short_circuits(self):
        store = await _store(default_policy=QuotaPolicy())
        for _ in range(50):
            assert (await store.check("acme", additional_requests=1)).allowed is True


class TestCostQuota:
    async def test_cost_accumulates_as_a_float(self):
        # `limits` counts events, not magnitudes: a token-heavy request is not
        # equivalent to a cheap one, which is why this is a separate counter.
        store = await _store(default_policy=QuotaPolicy(max_cost_usd=1.0))
        await store.consume("acme", requests=0, cost_usd=0.4)
        await store.consume("acme", requests=0, cost_usd=0.4)
        assert (await store.check("acme")).allowed is True
        await store.consume("acme", requests=0, cost_usd=0.4)
        verdict = await store.check("acme")
        assert verdict.allowed is False
        assert verdict.dimension == "cost_usd"
        assert verdict.used == pytest.approx(1.2)

    async def test_request_and_cost_ceilings_are_independent(self):
        store = await _store(
            default_policy=QuotaPolicy(max_requests=100, max_cost_usd=0.5),
        )
        await store.consume("acme", requests=1, cost_usd=0.9)
        verdict = await store.check("acme", additional_requests=1)
        # Well within the request ceiling, over the spend ceiling.
        assert verdict.allowed is False
        assert verdict.dimension == "cost_usd"


class TestFixedWindow:
    async def test_usage_resets_when_the_window_rolls(self, monkeypatch):
        store = await _store(
            default_policy=QuotaPolicy(window_seconds=3600, max_requests=1),
        )
        base = 1_000_000.0
        monkeypatch.setattr(tq.time, "time", lambda: base)
        await store.consume("acme", requests=1)
        assert (await store.check("acme", additional_requests=1)).allowed is False

        # A budget is for a period; crossing into the next one restores it.
        monkeypatch.setattr(tq.time, "time", lambda: base + 3600)
        assert (await store.check("acme", additional_requests=1)).allowed is True

    async def test_usage_report_shows_window_and_reset(self):
        store = await _store(default_policy=QuotaPolicy(window_seconds=3600, max_requests=10))
        await store.consume("acme", requests=2, cost_usd=0.25)
        usage = await store.usage("acme")
        assert usage["requests"]["used"] == 2
        assert usage["requests"]["limit"] == 10
        assert usage["cost_usd"]["used"] == pytest.approx(0.25)
        assert 0 < usage["reset_after_seconds"] <= 3600


class TestFailureBehaviour:
    async def test_unreachable_store_allows_the_request(self):
        # A quota is a cost control, not an authorization control: denying all
        # traffic because the counter is unavailable converts a billing
        # safeguard into an outage.
        store = await _store(default_policy=QuotaPolicy(max_requests=1))

        class _Broken:
            async def get(self, *_a, **_k):
                raise ConnectionError("redis gone")

            async def incrbyfloat(self, *_a, **_k):
                raise ConnectionError("redis gone")

            async def expire(self, *_a, **_k):
                raise ConnectionError("redis gone")

        store._redis = _Broken()
        assert (await store.check("acme", additional_requests=1)).allowed is True

    async def test_strict_mode_inverts_the_direction(self):
        store = TenantQuotaStore(
            default_policy=QuotaPolicy(max_requests=1), strict=True,
        )

        class _Broken:
            async def get(self, *_a, **_k):
                raise ConnectionError("redis gone")

        store._redis = _Broken()
        with pytest.raises(QuotaBackendUnavailable):
            await store.check("acme", additional_requests=1)

    async def test_strict_mode_refuses_a_process_local_counter(self):
        store = TenantQuotaStore(redis_url=None, strict=True)
        with pytest.raises(QuotaBackendUnavailable, match="no redis_url"):
            await store.connect()

    async def test_degradation_is_reported_to_metrics(self, monkeypatch):
        recorded: list[tuple[str, bool]] = []
        monkeypatch.setattr(
            tq, "set_store_degraded", lambda store, degraded: recorded.append((store, degraded)),
        )
        # Per-replica counting multiplies the effective ceiling by the replica
        # count, so it has to be visible rather than only logged.
        await _store(redis_url=None)
        assert ("tenant_quota", True) in recorded


class TestQuotaDependency:
    async def test_dependency_raises_429_with_retry_after(self, monkeypatch):
        from api.quota import TenantQuotaExceeded, enforce_tenant_quota

        store = await _store(default_policy=QuotaPolicy(max_requests=1))
        monkeypatch.setattr(tq, "_store", store)
        monkeypatch.setattr(tq, "_store_lock", None)

        assert await enforce_tenant_quota(tenant="acme") == "acme"
        with pytest.raises(TenantQuotaExceeded) as excinfo:
            await enforce_tenant_quota(tenant="acme")

        assert excinfo.value.status_code == 429
        assert excinfo.value.detail["dimension"] == "requests"
        assert int(excinfo.value.headers["Retry-After"]) >= 1

    async def test_usage_recording_never_fails_completed_work(self, monkeypatch):
        from graphrag.core.tenant_quota import record_tenant_usage

        async def _explode():
            raise ConnectionError("quota store gone")

        monkeypatch.setattr(tq, "get_quota_store", _explode)
        # The money is already spent; failing here would turn an accounting
        # problem into a user-visible error for work that succeeded.
        await record_tenant_usage("acme", cost_usd=0.5)


class TestQuotaIsNotARateLimit:
    def test_default_window_is_a_day_not_a_minute(self):
        # If this ever becomes minutes, quotas have silently turned into a
        # second rate limiter and the budget protection is gone.
        assert QuotaPolicy().window_seconds == 86_400

    def test_unlimited_sentinel_is_zero_not_none(self):
        assert UNLIMITED == 0.0
        assert QuotaPolicy(max_requests=UNLIMITED).is_unlimited()
