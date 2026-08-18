"""Unit tests for api/routes/query.py — the 503-vs-404 distinction added
when ResultStore can't reach Redis (see tasks/lessons.md, ResultStore
hardening). Uses a minimal FastAPI app with just this router mounted and
auth/result-store dependencies overridden, rather than calling the
route functions directly, since submit_query is wrapped by slowapi's
rate-limit decorator and needs a real Request to satisfy it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes import query as query_routes
from graphrag.retrieval.result_store import ResultStoreUnavailable
from graphrag.retrieval.session_store import SessionContextUnavailable


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(query_routes.router, prefix="/query")
    # The token must carry a tenant: routes now take tenant from the signed
    # claim (get_tenant) rather than the request body, and a tenantless token
    # is rejected with 403 before the handler runs.
    app.dependency_overrides[get_current_user] = lambda: {
        "scope": "read", "sub": "test", "tenant": "test-tenant",
    }

    # The real app sets request.state.correlation_id via middleware
    # (api/main.py); stand in with a fixed value so submit_query's use of it
    # doesn't fail before reaching the logic this test file actually covers.
    @app.middleware("http")
    async def _set_correlation_id(request, call_next):
        request.state.correlation_id = "test-correlation-id"
        return await call_next(request)

    return TestClient(app)


class TestSubmitQueryResultStoreDown:
    def test_503_when_status_cannot_be_persisted(self):
        client = _make_client()
        mock_store = AsyncMock()
        mock_store.set_status = AsyncMock(side_effect=ResultStoreUnavailable("Redis down"))

        with (
            patch("api.routes.query.get_result_store", return_value=mock_store),
            patch("api.routes.query.publish_query", new_callable=AsyncMock) as mock_publish,
        ):
            resp = client.post("/query", json={"question": "hello"})

        assert resp.status_code == 503
        mock_publish.assert_not_awaited()  # must not enqueue work with nowhere to land

    def test_200_and_publishes_when_status_persists(self):
        client = _make_client()
        mock_store = AsyncMock()
        mock_store.set_status = AsyncMock(return_value=None)

        with (
            patch("api.routes.query.get_result_store", return_value=mock_store),
            patch("api.routes.query.publish_query", new_callable=AsyncMock) as mock_publish,
        ):
            resp = client.post("/query", json={"question": "hello"})

        assert resp.status_code == 200
        mock_publish.assert_awaited_once()


class TestRequiresSessionContext:
    """A156: a follow-up marked requires_session_context must be refused
    (not silently answered without history) if session storage can't be
    reached — checked synchronously at enqueue time, since POST /query
    returns immediately and a later worker-side failure could only ever
    surface via polling, not as a direct response."""

    def test_400_when_no_session_id(self):
        client = _make_client()
        with patch("api.routes.query.get_result_store", return_value=AsyncMock()):
            resp = client.post("/query", json={
                "question": "what about their engines?",
                "requires_session_context": True,
            })
        assert resp.status_code == 400

    def test_503_when_session_store_unavailable_and_never_publishes(self):
        client = _make_client()
        mock_session_store = AsyncMock()
        mock_session_store.load_turns = AsyncMock(
            side_effect=SessionContextUnavailable("Redis down")
        )
        mock_result_store = AsyncMock()

        with (
            patch("api.routes.query.get_session_store", return_value=mock_session_store),
            patch("api.routes.query.get_result_store", return_value=mock_result_store),
            patch("api.routes.query.publish_query", new_callable=AsyncMock) as mock_publish,
        ):
            resp = client.post("/query", json={
                "question": "what about their engines?",
                "session_id": "s1",
                "requires_session_context": True,
            })

        assert resp.status_code == 503
        mock_result_store.set_status.assert_not_awaited()
        mock_publish.assert_not_awaited()

    def test_200_when_session_store_available(self):
        client = _make_client()
        mock_session_store = AsyncMock()
        mock_session_store.load_turns = AsyncMock(return_value=[])
        mock_result_store = AsyncMock()
        mock_result_store.set_status = AsyncMock(return_value=None)

        with (
            patch("api.routes.query.get_session_store", return_value=mock_session_store),
            patch("api.routes.query.get_result_store", return_value=mock_result_store),
            patch("api.routes.query.publish_query", new_callable=AsyncMock) as mock_publish,
        ):
            resp = client.post("/query", json={
                "question": "what about their engines?",
                "session_id": "s1",
                "requires_session_context": True,
            })

        assert resp.status_code == 200
        mock_publish.assert_awaited_once()

    def test_default_false_skips_precheck_entirely(self):
        """When requires_session_context is left at its default, no
        session-store call should happen at all — existing single-question
        callers are completely unaffected."""
        client = _make_client()
        mock_session_store = AsyncMock()
        mock_result_store = AsyncMock()
        mock_result_store.set_status = AsyncMock(return_value=None)

        with (
            patch("api.routes.query.get_session_store", return_value=mock_session_store),
            patch("api.routes.query.get_result_store", return_value=mock_result_store),
            patch("api.routes.query.publish_query", new_callable=AsyncMock),
        ):
            resp = client.post("/query", json={"question": "hello"})

        assert resp.status_code == 200
        mock_session_store.load_turns.assert_not_awaited()


class TestGetQueryResultResultStoreDown:
    def test_503_not_404_when_redis_unavailable(self):
        """A 404 here would claim the query doesn't exist — but we don't
        actually know that; storage just couldn't be checked."""
        client = _make_client()
        mock_store = AsyncMock()
        mock_store.get = AsyncMock(side_effect=ResultStoreUnavailable("Redis down"))

        with patch("api.routes.query.get_result_store", return_value=mock_store):
            resp = client.get("/query/some-id")

        assert resp.status_code == 503

    def test_404_when_redis_responds_but_key_missing(self):
        client = _make_client()
        mock_store = AsyncMock()
        mock_store.get = AsyncMock(return_value=None)

        with patch("api.routes.query.get_result_store", return_value=mock_store):
            resp = client.get("/query/missing-id")

        assert resp.status_code == 404

    def test_200_when_result_found(self):
        client = _make_client()
        mock_store = AsyncMock()
        mock_store.get = AsyncMock(return_value={
            "status": "completed", "answer": "42", "tenant": "test-tenant",
        })

        with patch("api.routes.query.get_result_store", return_value=mock_store):
            resp = client.get("/query/found-id")

        assert resp.status_code == 200
        assert resp.json()["answer"] == "42"


class TestGetQueryResultTenantIsolation:
    """Adversarial: the result-store key is the query_id alone, so ownership
    must be enforced in the handler. Before this check, any caller holding a
    'read' scope could fetch any tenant's stored answer and cited source text
    by id — and /kpis/timeseries?metric=query_id handed out the ids, removing
    the uuid4-guessing barrier that was the only control.
    """

    def test_other_tenants_result_is_not_returned(self):
        client = _make_client()   # token tenant: "test-tenant"
        mock_store = AsyncMock()
        mock_store.get = AsyncMock(return_value={
            "status": "completed",
            "answer": "victim tenant's confidential answer",
            "tenant": "victim-tenant",
        })

        with patch("api.routes.query.get_result_store", return_value=mock_store):
            resp = client.get("/query/someone-elses-id")

        assert resp.status_code == 404
        assert "confidential" not in resp.text

    def test_entry_without_tenant_fails_closed(self):
        """A result written by a worker from before the tenant field existed
        is treated as not-yours, never as not-checked."""
        client = _make_client()
        mock_store = AsyncMock()
        mock_store.get = AsyncMock(return_value={"status": "completed", "answer": "42"})

        with patch("api.routes.query.get_result_store", return_value=mock_store):
            resp = client.get("/query/legacy-id")

        assert resp.status_code == 404
