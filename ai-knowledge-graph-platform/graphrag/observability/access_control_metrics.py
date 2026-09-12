"""Rejection counters for the two access-control gates in front of the API:
the rate limiter (`api/limiter.py`) and the tenant quota
(`api/quota.py`).

`docs/performance-metrics-inventory.md` names this exact gap: "no dedicated
Prometheus counter for 429 events exists yet" — both gates already log a
structured event on rejection (`rate_limit.rejected`,
`rate_limit.conversation_rejected`, `quota.rejected`), but neither exposes an
aggregable count. A log line answers "did this happen"; a counter answers
"how often, and is it trending up" without grepping logs.

Same optional-dependency shape as `graphrag/observability/context_size.py`:
`prometheus_client` is not a hard runtime dependency of every deployment, so
the Counters degrade to a no-op rather than an import error when it's absent.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)

try:
    from prometheus_client import Counter
except ImportError:  # pragma: no cover - optional local dependency
    Counter = None

_rate_limit_rejected = Counter(
    "graphrag_rate_limit_rejected_total",
    "Requests rejected by the rate limiter (429), by route",
    ["route"],
) if Counter else None

_quota_rejected = Counter(
    "graphrag_quota_rejected_total",
    "Requests rejected by the tenant quota (429), by tenant and dimension",
    ["tenant", "dimension"],
) if Counter else None


def record_rate_limit_rejected(route: str) -> None:
    """Publish one rate-limit rejection for `route`.

    Never raises — a metrics-backend problem must not turn an already-decided
    429 into a 500, matching `record_context_composition`'s posture.
    """
    try:
        if _rate_limit_rejected:
            _rate_limit_rejected.labels(route).inc()
    except Exception as exc:  # noqa: BLE001 - metrics must never break the request
        log.warning("observability.rate_limit_metric_failed", error=str(exc)[:200])


def record_quota_rejected(tenant: str, dimension: str) -> None:
    """Publish one quota rejection for `tenant` on `dimension` (e.g.
    "requests" or "cost_usd" — whichever ceiling `TenantQuotaVerdict.dimension`
    named). Never raises, same rationale as `record_rate_limit_rejected`.
    """
    try:
        if _quota_rejected:
            _quota_rejected.labels(tenant, dimension).inc()
    except Exception as exc:  # noqa: BLE001
        log.warning("observability.quota_metric_failed", error=str(exc)[:200])


__all__ = ["record_rate_limit_rejected", "record_quota_rejected"]
