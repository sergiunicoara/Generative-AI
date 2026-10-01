"""Per-tenant consumption quotas over a long window.

Quotas are not rate limits
--------------------------
The two are often conflated and solve different problems:

- A **rate limit** (`api/limiter.py`) protects the *system* from a burst. It is
  per-caller, over seconds or minutes, and a rejected request is expected to be
  retried shortly.
- A **quota** protects the *budget* from sustained use. It is per-tenant, over
  hours or a day, and a rejected request will keep being rejected until the
  window rolls — retrying does not help.

Without quotas, one tenant running steadily just under the rate limit can
consume an entire day of shared LLM spend. Every request is individually
well-behaved, so nothing throttles, and the first signal is the provider bill
or a quota exhaustion that takes every other tenant down with it. That is the
failure this module exists to prevent, and it is invisible to a rate limiter by
construction.

Two dimensions
--------------
`requests` is an integer count. `cost_usd` is a float accumulation, which is
why this is a plain Redis counter rather than another `limits` window —
`limits` counts events, not magnitudes, and a token-heavy request is not
equivalent to a cheap one.

Cost is recorded *after* the work completes (the true cost is unknown before
it), so a tenant can overshoot its cost ceiling by at most the cost of the
requests already in flight when it crossed. Bounding that precisely would need
a reservation protocol; the overshoot is small, self-correcting within the
window, and not worth the added failure modes.

Fail-open
---------
An unreachable quota store allows the request. A quota is a cost-control
mechanism, not an authorization one: denying all traffic because the counter is
unavailable converts a billing safeguard into an outage. `strict` inverts this
for deployments where exceeding spend is worse than being unavailable, and is
off by default so the safe-by-default choice is the available one.

Windows are fixed, not rolling: a quota is a budget for a period, and "requests
in the last 24 hours" is a materially harder thing to explain on an invoice
than "requests today".
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import structlog

from graphrag.observability.operational_metrics import set_store_degraded

log = structlog.get_logger(__name__)

_KEY_PREFIX = "graphrag:tenant-quota:v1:"

# Sentinel meaning "no ceiling on this dimension".
UNLIMITED = 0.0


@dataclass(frozen=True)
class QuotaPolicy:
    """Ceilings for one tenant over one window."""

    window_seconds: int = 86_400
    max_requests: float = UNLIMITED
    max_cost_usd: float = UNLIMITED

    def is_unlimited(self) -> bool:
        return self.max_requests == UNLIMITED and self.max_cost_usd == UNLIMITED


@dataclass(frozen=True)
class QuotaVerdict:
    """The outcome of a quota check, safe to return to a caller."""

    allowed: bool
    tenant: str
    dimension: str = ""          # "requests" | "cost_usd" | "" when allowed
    used: float = 0.0
    limit: float = 0.0
    reset_after_seconds: int = 0

    def as_detail(self) -> dict[str, Any]:
        """A 429 body that tells the caller what to do, not just that it failed."""
        return {
            "error": "tenant_quota_exceeded",
            "tenant": self.tenant,
            "dimension": self.dimension,
            "used": round(self.used, 6),
            "limit": self.limit,
            "reset_after_seconds": self.reset_after_seconds,
        }


class QuotaBackendUnavailable(RuntimeError):
    """Raised in strict mode when the quota store cannot be reached."""


class TenantQuotaStore:
    """Fixed-window per-tenant counters, Redis-backed with a local fallback."""

    def __init__(
        self,
        redis_url: str | None = None,
        *,
        policies: dict[str, QuotaPolicy] | None = None,
        default_policy: QuotaPolicy | None = None,
        strict: bool = False,
    ) -> None:
        self._redis_url = redis_url
        self._redis = None
        self._strict = strict
        self._policies = dict(policies or {})
        self._default = default_policy or QuotaPolicy()
        # tenant -> (window_start, {dimension: used})
        self._memory: dict[str, tuple[int, dict[str, float]]] = {}

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        if not self._redis_url:
            if self._strict:
                raise QuotaBackendUnavailable(
                    "tenant_quota_strict is set but no redis_url is configured"
                )
            log.warning(
                "tenant_quota.no_redis",
                impact="quotas are counted per replica, so the effective ceiling "
                       "is multiplied by the replica count",
            )
            set_store_degraded("tenant_quota", True)
            return
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
            await self._redis.ping()
            log.info("tenant_quota.redis_connected")
            set_store_degraded("tenant_quota", False)
        except Exception as exc:  # Redis error types vary across redis-py versions.
            self._redis = None
            if self._strict:
                raise QuotaBackendUnavailable(
                    f"tenant quotas require Redis but it is unreachable: {exc}"
                ) from exc
            log.warning("tenant_quota.redis_unavailable", error=str(exc))
            set_store_degraded("tenant_quota", True)

    async def close(self) -> None:
        redis, self._redis = self._redis, None
        if redis is not None:
            try:
                await redis.aclose()
            except Exception as exc:  # noqa: BLE001 - shutdown must never raise
                log.warning("tenant_quota.close_error", error=str(exc))

    # ── Policy ───────────────────────────────────────────────────────────────

    def policy_for(self, tenant: str) -> QuotaPolicy:
        return self._policies.get(tenant, self._default)

    def _window_start(self, policy: QuotaPolicy, now: float | None = None) -> int:
        seconds = max(1, policy.window_seconds)
        return int((now if now is not None else time.time()) // seconds) * seconds

    def _key(self, tenant: str, dimension: str, window_start: int) -> str:
        return f"{_KEY_PREFIX}{tenant}:{dimension}:{window_start}"

    # ── Checks and consumption ───────────────────────────────────────────────

    async def check(self, tenant: str, *, additional_requests: float = 0.0) -> QuotaVerdict:
        """Would `tenant` be within quota if it made one more request?

        Checked *before* the work runs, so an over-quota tenant is rejected
        cheaply rather than after consuming the resource it has no budget for.
        """
        policy = self.policy_for(tenant)
        if policy.is_unlimited():
            return QuotaVerdict(allowed=True, tenant=tenant)

        window_start = self._window_start(policy)
        reset_after = max(1, window_start + policy.window_seconds - int(time.time()))

        for dimension, ceiling, increment in (
            ("requests", policy.max_requests, additional_requests),
            ("cost_usd", policy.max_cost_usd, 0.0),
        ):
            if ceiling == UNLIMITED:
                continue
            used = await self._read(tenant, dimension, window_start)
            if used + increment > ceiling:
                return QuotaVerdict(
                    allowed=False, tenant=tenant, dimension=dimension,
                    used=used, limit=ceiling, reset_after_seconds=reset_after,
                )
        return QuotaVerdict(allowed=True, tenant=tenant, reset_after_seconds=reset_after)

    async def consume(
        self, tenant: str, *, requests: float = 1.0, cost_usd: float = 0.0,
    ) -> None:
        """Record consumption against `tenant`'s current window."""
        policy = self.policy_for(tenant)
        if policy.is_unlimited():
            return
        window_start = self._window_start(policy)
        ttl = policy.window_seconds * 2  # outlive the window, then expire itself
        if requests:
            await self._increment(tenant, "requests", window_start, requests, ttl)
        if cost_usd:
            await self._increment(tenant, "cost_usd", window_start, cost_usd, ttl)

    async def usage(self, tenant: str) -> dict[str, Any]:
        """Current window usage, for an operator or a quota endpoint."""
        policy = self.policy_for(tenant)
        window_start = self._window_start(policy)
        return {
            "tenant": tenant,
            "window_seconds": policy.window_seconds,
            "reset_after_seconds": max(
                0, window_start + policy.window_seconds - int(time.time()),
            ),
            "requests": {
                "used": await self._read(tenant, "requests", window_start),
                "limit": policy.max_requests,
            },
            "cost_usd": {
                "used": await self._read(tenant, "cost_usd", window_start),
                "limit": policy.max_cost_usd,
            },
        }

    # ── Storage ──────────────────────────────────────────────────────────────

    async def _read(self, tenant: str, dimension: str, window_start: int) -> float:
        if self._redis is not None:
            try:
                raw = await self._redis.get(self._key(tenant, dimension, window_start))
                return float(raw) if raw else 0.0
            except Exception as exc:  # noqa: BLE001
                log.warning("tenant_quota.read_error", error=str(exc), tenant=tenant)
                if self._strict:
                    raise QuotaBackendUnavailable(str(exc)) from exc
                set_store_degraded("tenant_quota", True)
                # Fail open: an unknown usage figure must not deny the request.
                return 0.0
        entry = self._memory.get(tenant)
        if entry is None or entry[0] != window_start:
            return 0.0
        return float(entry[1].get(dimension, 0.0))

    async def _increment(
        self, tenant: str, dimension: str, window_start: int, amount: float, ttl: int,
    ) -> None:
        if self._redis is not None:
            try:
                key = self._key(tenant, dimension, window_start)
                # INCRBYFLOAT is atomic, so concurrent replicas cannot lose an
                # increment through read-modify-write.
                await self._redis.incrbyfloat(key, amount)
                await self._redis.expire(key, ttl)
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("tenant_quota.write_error", error=str(exc), tenant=tenant)
                if self._strict:
                    raise QuotaBackendUnavailable(str(exc)) from exc
                set_store_degraded("tenant_quota", True)
        window, counters = self._memory.get(tenant, (window_start, {}))
        if window != window_start:
            window, counters = window_start, {}
        counters[dimension] = counters.get(dimension, 0.0) + amount
        self._memory[tenant] = (window, counters)


def policies_from_config(raw: dict[str, Any] | None) -> tuple[dict[str, QuotaPolicy], QuotaPolicy]:
    """Build policies from the `quotas` block in settings.yml.

    Shape::

        quotas:
          window_seconds: 86400
          default:
            max_requests: 10000
            max_cost_usd: 50
          tenants:
            aerospace:
              max_requests: 50000

    An absent or zero ceiling means unlimited on that dimension, so adding the
    block does not silently start throttling a deployment that has not chosen
    numbers yet.
    """
    raw = raw or {}
    window = int(raw.get("window_seconds", 86_400) or 86_400)

    def _policy(entry: dict[str, Any] | None) -> QuotaPolicy:
        entry = entry or {}
        return QuotaPolicy(
            window_seconds=int(entry.get("window_seconds", window) or window),
            max_requests=float(entry.get("max_requests", UNLIMITED) or UNLIMITED),
            max_cost_usd=float(entry.get("max_cost_usd", UNLIMITED) or UNLIMITED),
        )

    default = _policy(raw.get("default"))
    tenants = {
        name: _policy(entry) for name, entry in (raw.get("tenants") or {}).items()
    }
    return tenants, default


_store: TenantQuotaStore | None = None
_store_lock: asyncio.Lock | None = None


async def get_quota_store() -> TenantQuotaStore:
    """Process singleton, safe against concurrent cold-start."""
    global _store, _store_lock
    if _store_lock is None:
        _store_lock = asyncio.Lock()
    async with _store_lock:
        if _store is None:
            import os

            from graphrag.core.config import get_settings

            cfg = get_settings()
            quotas = cfg.quotas
            policies, default = policies_from_config(quotas)
            redis_url = os.getenv("REDIS_URL") or cfg.retrieval.get("redis_url", "") or None
            candidate = TenantQuotaStore(
                redis_url=redis_url,
                policies=policies,
                default_policy=default,
                strict=bool(quotas.get("strict", False)),
            )
            await candidate.connect()
            _store = candidate
    return _store


async def record_tenant_usage(tenant: str, *, cost_usd: float) -> None:
    """Record realised spend against `tenant`'s window.

    Called after the work completes, from the worker that actually spent the
    money. A tenant can therefore overshoot its cost ceiling by at most the
    cost of requests already in flight when it crossed; bounding that exactly
    would need a reservation protocol, and the overshoot self-corrects within
    the window.

    Lives here rather than in api/quota.py because the caller is
    graphrag/observability/genai_telemetry.py, which runs inside every worker
    image -- none of which install the web framework api/quota.py imports.
    """
    if not cost_usd:
        return
    try:
        store = await get_quota_store()
        await store.consume(tenant, requests=0.0, cost_usd=cost_usd)
    except Exception as exc:  # noqa: BLE001
        # Quota accounting must never fail a request whose work already
        # succeeded; the money is spent either way.
        log.warning("quota.usage_record_failed", tenant=tenant, error=str(exc))


async def close_quota_store() -> None:
    """Close and reset the process singleton when it was initialized."""
    global _store, _store_lock
    store, _store = _store, None
    _store_lock = None
    if store is not None:
        await store.close()
