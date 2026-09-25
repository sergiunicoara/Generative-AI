"""Regression tests for the audit-2026-09-23 "Not fixed" items closed on 2026-09-25."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user


def _client(router, *, scope: str, tenant: str = "acme", sub: str = "user-1") -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: {
        "scope": scope, "sub": sub, "tenant": tenant,
    }
    return TestClient(app)


class TestCatalogPaginationBounded:
    def test_limit_above_cap_is_rejected(self):
        from api.routes.kg import catalog

        with patch.object(catalog, "get_neo4j") as get_neo4j:
            get_neo4j.return_value.get_documents_catalog = AsyncMock(return_value=[])
            resp = _client(catalog.router, scope="read").get("/catalog/documents?limit=100000")
        assert resp.status_code == 422

    def test_negative_offset_is_rejected(self):
        from api.routes.kg import catalog

        resp = _client(catalog.router, scope="read").get("/catalog/documents?offset=-1")
        assert resp.status_code == 422

    async def test_listing_has_a_deterministic_tie_breaker(self):
        from graphrag.graph.neo4j_client import Neo4jClient

        client = Neo4jClient.__new__(Neo4jClient)
        client.run = AsyncMock(return_value=[])
        await client.get_documents_catalog(tenant="acme")
        assert "ORDER BY d.ingested_at DESC, d.id ASC" in client.run.call_args[0][0]


class TestGdprErasure:
    def test_write_scope_can_no_longer_erase(self):
        from api.routes.kg import compliance

        resp = _client(compliance.router, scope="read write").post(
            "/gdpr/forget-document", json={"doc_id": "d1"},
        )
        assert resp.status_code == 403

    def test_audit_actor_comes_from_the_token_not_the_body(self):
        from api.routes.kg import compliance

        forget = AsyncMock(return_value={"status": "ok"})
        with patch("graphrag.graph.gdpr.GDPRService") as svc, patch.object(compliance, "get_neo4j"):
            svc.return_value.forget_document = forget
            resp = _client(compliance.router, scope="read write admin", sub="dpo-7").post(
                "/gdpr/forget-document", json={"doc_id": "d1", "requested_by": "ceo"},
            )
        assert resp.status_code == 200
        assert forget.call_args.kwargs["requested_by"] == "dpo-7 (on behalf of ceo)"


class TestCookieSecureOutsideDev:
    @pytest.mark.parametrize("env,expected", [
        ("production", True), ("staging", True), ("development", False), ("test", False),
    ])
    def test_secure_flag(self, env, expected):
        from api.routes import auth

        with patch("graphrag.core.config.get_settings", return_value=SimpleNamespace(env=env)):
            assert auth._cookie_secure() is expected


class TestProductionCorsRejectsWildcard:
    def test_wildcard_origin_rejected_in_production(self):
        from pydantic import ValidationError

        from graphrag.core.config import Settings

        with pytest.raises(ValidationError, match="cors_origins"):
            Settings(
                _env_file=None,
                env="production",
                jwt_secret_key="x" * 40,
                session_secret_key="y" * 40,
                neo4j_password="strong-password",
                rabbitmq_url="amqp://u:p@rabbit.internal:5672/",
                cors_origins=["*"],
            )


class TestRevocationStrictByDefaultOutsideDev:
    @pytest.fixture(autouse=True)
    def _reset_store(self, monkeypatch):
        import graphrag.core.token_revocation as tr

        monkeypatch.setattr(tr, "_store", None)
        monkeypatch.setattr(tr, "_store_lock", None)
        monkeypatch.delenv("REDIS_URL", raising=False)
        yield

    def _settings(self, env, retrieval=None):
        return SimpleNamespace(env=env, retrieval=retrieval or {}, jwt_revocation_ttl_seconds=60)

    async def test_production_without_redis_fails_closed(self):
        from graphrag.core.token_revocation import RevocationBackendUnavailable, get_revocation_store

        with patch("graphrag.core.config.get_settings", return_value=self._settings("production")):
            with pytest.raises(RevocationBackendUnavailable):
                await get_revocation_store()

    async def test_development_without_redis_stays_lenient(self):
        from graphrag.core.token_revocation import get_revocation_store

        with patch("graphrag.core.config.get_settings", return_value=self._settings("development")):
            store = await get_revocation_store()
        assert store._strict is False

    async def test_explicit_false_still_opts_out(self):
        from graphrag.core.token_revocation import get_revocation_store

        settings = self._settings("production", {"jwt_revocation_strict": False})
        with patch("graphrag.core.config.get_settings", return_value=settings):
            store = await get_revocation_store()
        assert store._strict is False


class TestAdminLoginRateLimit:
    def test_repeated_failures_are_throttled_and_success_resets(self, monkeypatch):
        import graphrag.dashboard.app as dash_app

        monkeypatch.setattr(dash_app, "ADMIN_TOKEN", "correct-token")
        monkeypatch.setattr(dash_app, "_login_failures", {})
        client = dash_app.app.server.test_client()

        for _ in range(dash_app._LOGIN_MAX_FAILURES):
            resp = client.post("/admin/_login", data={"token": "wrong"})
            assert resp.status_code == 302
        blocked = client.post("/admin/_login", data={"token": "correct-token"})
        assert blocked.status_code == 429

        monkeypatch.setattr(dash_app, "_login_failures", {})
        ok = client.post("/admin/_login", data={"token": "correct-token"})
        assert ok.status_code == 302
        assert ok.headers["Location"].endswith("/admin/")


class TestKpiUncomputedMetricsExcludedFromAverages:
    async def test_none_metric_does_not_drag_the_average_down(self, tmp_path, monkeypatch):
        import graphrag.business_matrix.kpi_store as kpi_store
        from graphrag.business_matrix.kpi_tracker import KPITracker
        from graphrag.core.models import KPIEvent

        monkeypatch.setenv("KPI_DB_PATH", str(tmp_path / "kpis.db"))
        monkeypatch.delenv("TIMESCALE_DB_URL", raising=False)
        monkeypatch.setattr(kpi_store, "_engine", None)
        monkeypatch.setattr(kpi_store, "_session_factory", None)

        tracker = KPITracker()
        await tracker.record(KPIEvent(query_id="q1", tenant="acme", latency_ms=10.0, faithfulness=0.8))
        await tracker.record(KPIEvent(query_id="q2", tenant="acme", latency_ms=10.0))
        summary = await tracker.get_summary(tenant="acme")
        await kpi_store._engine.dispose()

        assert summary["total_queries"] == 2
        assert summary["avg_faithfulness"] == 0.8  # was 0.4 when q2 recorded a fake 0.0


class TestSessionTurnFailureDoesNotFailTheAnswer:
    async def test_record_turn_error_is_swallowed(self):
        from graphrag.core.models import CitationEvidence
        from tests.unit.test_hybrid_retriever import _make_hybrid_retriever

        hr = _make_hybrid_retriever({"agentic_fallback": False})
        hr._local.search = AsyncMock(return_value={"chunks": []})
        hr._global.search = AsyncMock(return_value={})
        hr._context_builder.build.return_value = (
            "context", ["DocA"], [CitationEvidence(source_id="DocA", source_label="DocA")],
        )
        hr._use_session_ctx = True
        hr._session_ctx = MagicMock()
        hr._session_ctx.record_turn = AsyncMock(side_effect=RuntimeError("redis down"))
        hr._session_ctx.get_context = AsyncMock(return_value="")

        # test_hybrid_retriever.py's autouse LLM-patch fixture only applies to
        # tests in that module; this test lives elsewhere and imports the
        # helper without it, so without this patch it hits a real provider --
        # passed locally with API keys configured, failed in CI with none.
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="A confident answer.")
        with patch("graphrag.retrieval.hybrid_retriever.get_llm", return_value=mock_llm):
            result = await hr.retrieve_and_answer("question", mode="local", session_id="s1")

        assert result.answer
        hr._session_ctx.record_turn.assert_awaited()


class TestAdminLoginTrustedProxyIp:
    def test_uses_forwarded_for_only_with_trusted_proxies_configured(self, monkeypatch):
        import graphrag.dashboard.app as dash_app

        monkeypatch.setattr(dash_app, "ADMIN_TOKEN", "correct-token")
        monkeypatch.setattr(dash_app, "_login_failures", {})
        client = dash_app.app.server.test_client()

        monkeypatch.setenv("GRAPHRAG_TRUSTED_PROXIES", "1")
        for _ in range(dash_app._LOGIN_MAX_FAILURES):
            client.post(
                "/admin/_login", data={"token": "wrong"},
                headers={"X-Forwarded-For": "203.0.113.9"},
            )
        assert "203.0.113.9" in dash_app._login_failures
        blocked = client.post(
            "/admin/_login", data={"token": "correct-token"},
            headers={"X-Forwarded-For": "203.0.113.9"},
        )
        assert blocked.status_code == 429
        # A different real client behind the same proxy is unaffected.
        other = client.post(
            "/admin/_login", data={"token": "correct-token"},
            headers={"X-Forwarded-For": "203.0.113.99"},
        )
        assert other.status_code == 302

    def test_untrusted_forwarded_for_is_ignored(self, monkeypatch):
        import graphrag.dashboard.app as dash_app

        monkeypatch.setattr(dash_app, "ADMIN_TOKEN", "correct-token")
        monkeypatch.setattr(dash_app, "_login_failures", {})
        monkeypatch.delenv("GRAPHRAG_TRUSTED_PROXIES", raising=False)
        client = dash_app.app.server.test_client()

        for i in range(dash_app._LOGIN_MAX_FAILURES):
            client.post(
                "/admin/_login", data={"token": "wrong"},
                headers={"X-Forwarded-For": f"1.1.1.{i}"},  # would forge a fresh bucket if trusted
            )
        assert list(dash_app._login_failures) != [f"1.1.1.{i}" for i in range(dash_app._LOGIN_MAX_FAILURES)]
        assert len(dash_app._login_failures) == 1  # all 5 attempts collapsed into one real-IP bucket


class TestAdminLoginFailuresBounded:
    def test_dict_does_not_grow_past_the_cap(self, monkeypatch):
        import graphrag.dashboard.app as dash_app

        monkeypatch.setattr(dash_app, "_LOGIN_FAILURES_MAX_TRACKED_IPS", 3)
        monkeypatch.setattr(dash_app, "_login_failures", {})

        now = 1000.0
        for i in range(5):
            dash_app._record_login_failure(f"ip-{i}", now + i)

        assert len(dash_app._login_failures) == 3
        assert "ip-0" not in dash_app._login_failures  # oldest evicted first
        assert "ip-4" in dash_app._login_failures


class TestGdprErasureRequiresIdentifiableActor:
    def test_token_without_sub_is_rejected_not_recorded_as_unknown(self):
        from api.routes.kg import compliance

        app = FastAPI()
        app.include_router(compliance.router)
        app.dependency_overrides[get_current_user] = lambda: {
            "scope": "read write admin", "tenant": "acme",
        }
        client = TestClient(app)

        resp = client.post("/gdpr/forget-document", json={"doc_id": "d1"})
        assert resp.status_code == 403
