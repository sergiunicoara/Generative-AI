"""ASGI middleware recording one AccessAuditEvent per request — see
graphrag/observability/access_audit.py for why this exists (the confirmed
gap: no generic, durable, queryable "who called what route" trail).

Reads request.state.user/correlation_id, both already populated by
RequireAuthMiddleware and the correlation-id middleware in api/main.py by
the time this runs -- no new claim-extraction logic, just reuses what's
already on the request. Ordering relative to RequireAuthMiddleware doesn't
matter for correctness here: this reads request.state only *after*
`call_next` returns, by which point every inner middleware (including
RequireAuthMiddleware) has already run and set it, regardless of which of
the two is registered as more outer.
"""

from __future__ import annotations

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from graphrag.observability.access_audit import (
    AccessAuditSink,
    event_from_claims,
)

log = structlog.get_logger(__name__)

# Kubelet probes are not audited. They carry no caller, arrive every few
# seconds per pod, and the sink write sits on the response path: with Neo4j
# unreachable it adds seconds of retries, so /health missed its 1 s probe
# timeout and the liveness probe restarted a healthy API pod (found by the
# kind smoke deploy). Liveness must not depend on the audit backend.
UNAUDITED_PATHS = frozenset({"/health", "/health/ready"})


class AccessAuditMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, sink: AccessAuditSink | None = None):
        super().__init__(app)
        self._sink = sink or _default_sink()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        if request.url.path in UNAUDITED_PATHS:
            return response
        try:
            event = event_from_claims(
                user=getattr(request.state, "user", None),
                route=request.url.path,
                method=request.method,
                status_code=response.status_code,
                correlation_id=str(getattr(request.state, "correlation_id", "")),
            )
            await self._sink.record(event)
        except Exception as exc:  # noqa: BLE001 - auditing must never break a served response
            log.warning("access_audit.middleware_failed", error=str(exc)[:200])
        return response


def _default_sink() -> AccessAuditSink:
    # Imported lazily so constructing the middleware (at app-startup time,
    # before any event loop or Neo4j connection necessarily exists) never
    # requires one -- Neo4jAccessAuditSink only touches the client inside
    # record(), which runs per-request.
    from graphrag.graph.neo4j_client import get_neo4j
    from graphrag.observability.access_audit import Neo4jAccessAuditSink

    return Neo4jAccessAuditSink(get_neo4j())


__all__ = ["AccessAuditMiddleware"]
