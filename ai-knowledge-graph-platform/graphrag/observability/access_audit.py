"""A unified, durable "who called what route" audit log for the API.

The platform already has real, layered auth (OAuth2/M2M-JWT, scopes, tenant
isolation, key rotation, revocation — `api/auth/*`) and several *domain*
audit trails (GDPR deletions, graph-edge corrections via
`graphrag/graph/audit_trail.py`) — but nothing generic answers "who called
what route, with what scope/tenant, at what time, with what outcome" across
every request. Structured log lines exist at individual decision points
(`auth.middleware_denied`, `quota.rejected`, ...) but are per-instance and
not queryable as a durable trail.

This module is deliberately small: an event shape, a `Protocol` sink so the
recording logic is testable without a live Neo4j, an in-memory sink for
tests, and a production sink that appends an `AccessAuditEvent` node —
mirroring `graphrag/graph/audit_trail.py`'s append-only-ChangeLog pattern,
just for API access rather than graph mutations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

import structlog

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class AccessAuditEvent:
    timestamp: str
    subject: str
    scope: str
    tenant: str
    route: str
    method: str
    status_code: int
    correlation_id: str = ""


class AccessAuditSink(Protocol):
    async def record(self, event: AccessAuditEvent) -> None: ...


class InMemoryAccessAuditSink:
    """Test double — records every event in a list, nothing more."""

    def __init__(self) -> None:
        self.events: list[AccessAuditEvent] = []

    async def record(self, event: AccessAuditEvent) -> None:
        self.events.append(event)


class Neo4jAccessAuditSink:
    """Production sink — one append-only AccessAuditEvent node per request.

    Never raises: a metrics/audit-backend problem must not turn an
    already-served response into a 500, matching AuditTrail's own
    try/except-and-log posture on every write method.
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    async def record(self, event: AccessAuditEvent) -> None:
        try:
            await self._neo4j.run(
                """
                CREATE (:AccessAuditEvent {
                    id:             $id,
                    timestamp:      $timestamp,
                    subject:        $subject,
                    scope:          $scope,
                    tenant:         $tenant,
                    route:          $route,
                    method:         $method,
                    status_code:    $status_code,
                    correlation_id: $correlation_id
                })
                """,
                id=str(uuid4()),
                timestamp=event.timestamp,
                subject=event.subject,
                scope=event.scope,
                tenant=event.tenant,
                route=event.route,
                method=event.method,
                status_code=event.status_code,
                correlation_id=event.correlation_id,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "access_audit.record_failed",
                route=event.route,
                error=str(exc)[:200],
            )


def event_from_claims(
    *,
    user: dict | None,
    route: str,
    method: str,
    status_code: int,
    correlation_id: str = "",
) -> AccessAuditEvent:
    """Build an event from the claims dict RequireAuthMiddleware attaches to
    `request.state.user` (see api/auth/default_auth.py) — `None` for a public
    route that never went through token verification.
    """
    claims = user or {}
    return AccessAuditEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        subject=str(claims.get("sub", "")),
        scope=str(claims.get("scope", "")),
        tenant=str(claims.get("tenant", "")),
        route=route,
        method=method,
        status_code=status_code,
        correlation_id=correlation_id,
    )


__all__ = [
    "AccessAuditEvent",
    "AccessAuditSink",
    "InMemoryAccessAuditSink",
    "Neo4jAccessAuditSink",
    "event_from_claims",
]
