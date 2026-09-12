"""FastAPI dependency enforcing per-tenant consumption quotas.

Separate from `api/limiter.py` because the two answer different questions and
fail differently. See `graphrag/core/tenant_quota.py` for the full argument;
the short version:

- the rate limiter protects the system from a burst, is keyed per *caller*, and
  a rejection means "retry shortly";
- the quota protects the budget from sustained use, is keyed per *tenant*, and
  a rejection means "not until the window rolls".

The tenant comes from `get_tenant`, i.e. the signed token — never from the
request body. A client-supplied tenant would let any caller spend another
tenant's budget, or dodge its own by naming a different one.
"""

from __future__ import annotations

import structlog
from fastapi import Depends, HTTPException, status

from api.auth.dependencies import get_tenant
from graphrag.observability.access_control_metrics import record_quota_rejected

log = structlog.get_logger(__name__)


class TenantQuotaExceeded(HTTPException):
    """429 whose body says which ceiling was hit and when it resets.

    A bare "quota exceeded" leaves the caller unable to distinguish a
    request-count ceiling from a spend ceiling, or to know whether to retry in
    a minute or tomorrow. Both are in the body, and `Retry-After` carries the
    window reset so an automated client can back off correctly.
    """

    def __init__(self, verdict) -> None:
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=verdict.as_detail(),
            headers={"Retry-After": str(max(1, verdict.reset_after_seconds))},
        )


async def enforce_tenant_quota(tenant: str = Depends(get_tenant)) -> str:
    """Reject the request when `tenant` is out of quota for the window.

    Checked *before* the work runs: an over-quota tenant should be refused
    cheaply, not after consuming the LLM and graph capacity it has no budget
    for. Consumption is recorded separately, after the work completes, because
    the true cost is not known in advance — see `record_tenant_usage`.
    """
    from graphrag.core.tenant_quota import get_quota_store

    store = await get_quota_store()
    verdict = await store.check(tenant, additional_requests=1.0)
    if not verdict.allowed:
        log.info(
            "quota.rejected",
            tenant=tenant,
            dimension=verdict.dimension,
            used=verdict.used,
            limit=verdict.limit,
        )
        record_quota_rejected(tenant, verdict.dimension)
        raise TenantQuotaExceeded(verdict)
    await store.consume(tenant, requests=1.0)
    return tenant


async def record_tenant_usage(tenant: str, *, cost_usd: float) -> None:
    """Record realised spend against `tenant`'s window.

    Called after the work completes, from the worker that actually spent the
    money. A tenant can therefore overshoot its cost ceiling by at most the
    cost of requests already in flight when it crossed; bounding that exactly
    would need a reservation protocol, and the overshoot self-corrects within
    the window.
    """
    if not cost_usd:
        return
    from graphrag.core.tenant_quota import get_quota_store

    try:
        store = await get_quota_store()
        await store.consume(tenant, requests=0.0, cost_usd=cost_usd)
    except Exception as exc:  # noqa: BLE001
        # Quota accounting must never fail a request whose work already
        # succeeded; the money is spent either way.
        log.warning("quota.usage_record_failed", tenant=tenant, error=str(exc))
