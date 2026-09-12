"""AccessAuditMiddleware + graphrag.observability.access_audit.

A minimal FastAPI app with the middleware and an InMemoryAccessAuditSink,
asserting a request produces exactly one correctly-populated event -- no
live Neo4j needed (Neo4jAccessAuditSink is exercised separately below
against a fake neo4j_client, mirroring AuditTrail's own test style).
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth.access_audit_middleware import AccessAuditMiddleware
from graphrag.observability.access_audit import (
    AccessAuditEvent,
    InMemoryAccessAuditSink,
    Neo4jAccessAuditSink,
    event_from_claims,
)


def _make_app(sink) -> FastAPI:
    app = FastAPI()
    app.add_middleware(AccessAuditMiddleware, sink=sink)

    @app.get("/kg/entities")
    async def entities():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


class TestAccessAuditMiddleware:
    def test_records_exactly_one_event_per_request(self):
        sink = InMemoryAccessAuditSink()
        client = TestClient(_make_app(sink))

        response = client.get("/kg/entities")

        assert response.status_code == 200
        assert len(sink.events) == 1

    def test_event_captures_route_method_and_status(self):
        sink = InMemoryAccessAuditSink()
        client = TestClient(_make_app(sink))

        client.get("/kg/entities")

        event = sink.events[0]
        assert event.route == "/kg/entities"
        assert event.method == "GET"
        assert event.status_code == 200

    def test_no_state_user_produces_empty_subject_not_a_crash(self):
        # No RequireAuthMiddleware in this minimal app, so request.state.user
        # is never set -- a public/unauthenticated route must still audit
        # cleanly rather than raise.
        sink = InMemoryAccessAuditSink()
        client = TestClient(_make_app(sink))

        response = client.get("/health")

        assert response.status_code == 200
        assert sink.events[0].subject == ""
        assert sink.events[0].tenant == ""

    def test_sink_failure_never_breaks_the_response(self):
        class ExplodingSink:
            async def record(self, event):
                raise RuntimeError("audit backend down")

        client = TestClient(_make_app(ExplodingSink()))
        response = client.get("/kg/entities")

        assert response.status_code == 200  # served despite the audit failure


class TestEventFromClaims:
    def test_populates_fields_from_claims(self):
        event = event_from_claims(
            user={"sub": "client-1", "scope": "read write", "tenant": "aerospace"},
            route="/kg/entities",
            method="GET",
            status_code=200,
            correlation_id="corr-1",
        )
        assert event.subject == "client-1"
        assert event.scope == "read write"
        assert event.tenant == "aerospace"
        assert event.correlation_id == "corr-1"

    def test_none_user_produces_empty_fields_not_an_error(self):
        event = event_from_claims(user=None, route="/health", method="GET", status_code=200)
        assert event.subject == ""
        assert event.tenant == ""


class TestNeo4jAccessAuditSink:
    @pytest.mark.asyncio
    async def test_record_calls_run_with_event_fields(self):
        calls = []

        class FakeNeo4j:
            async def run(self, query, **kwargs):
                calls.append(kwargs)
                return []

        sink = Neo4jAccessAuditSink(FakeNeo4j())
        event = AccessAuditEvent(
            timestamp="2026-01-01T00:00:00+00:00", subject="s1", scope="read",
            tenant="aerospace", route="/kg/entities", method="GET", status_code=200,
            correlation_id="corr-1",
        )
        await sink.record(event)

        assert len(calls) == 1
        assert calls[0]["route"] == "/kg/entities"
        assert calls[0]["status_code"] == 200

    @pytest.mark.asyncio
    async def test_backend_failure_is_swallowed_not_raised(self):
        class FailingNeo4j:
            async def run(self, *a, **kw):
                raise RuntimeError("connection refused")

        sink = Neo4jAccessAuditSink(FailingNeo4j())
        event = AccessAuditEvent(
            timestamp="t", subject="s", scope="", tenant="t", route="/x",
            method="GET", status_code=200,
        )
        await sink.record(event)  # must not raise
