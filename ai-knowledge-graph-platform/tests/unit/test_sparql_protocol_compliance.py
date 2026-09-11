"""SPARQL 1.1 Protocol content-negotiation tests for POST /kg/sparql.

The route's original custom JSON shape must survive byte-identically
(tests/unit/test_tenant_isolation.py's TestSPARQLPerTenantExport already
pins that; this file is additive, not a replacement). These tests cover the
standard Content-Type/Accept forms layered on top, and the remote-endpoint
pass-through path added alongside them.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes.kg import knowledge as kg_knowledge


def _client(tenant: str = "acme") -> TestClient:
    app = FastAPI()
    app.include_router(kg_knowledge.router)
    app.dependency_overrides[get_current_user] = lambda: {
        "scope": "read write", "sub": "test", "tenant": tenant,
    }
    return TestClient(app)


def _write_export(tmp_path, tenant: str = "acme") -> None:
    export_dir = tmp_path / tenant
    export_dir.mkdir()
    (export_dir / "graph_export.ttl").write_text(
        "@prefix ex: <http://example.org/> .\n"
        'ex:alice ex:name "Alice" .\n'
    )


class TestLegacyShapeIsUnchanged:
    """Redundant with test_tenant_isolation.py by design -- this file's own
    regression guard, kept local so it doesn't depend on another file."""

    def test_default_content_type_and_accept_return_the_legacy_shape(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_RDF_EXPORT_DIR", str(tmp_path))
        monkeypatch.delenv("GRAPHRAG_SPARQL_ENDPOINT", raising=False)
        _write_export(tmp_path)

        resp = _client().post("/sparql", json={"query": "SELECT ?s WHERE { ?s ?p ?o }"})

        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {"rows", "count"}
        assert body["count"] == 1


class TestRequestContentNegotiation:
    def test_application_sparql_query_body_is_accepted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_RDF_EXPORT_DIR", str(tmp_path))
        monkeypatch.delenv("GRAPHRAG_SPARQL_ENDPOINT", raising=False)
        _write_export(tmp_path)

        resp = _client().post(
            "/sparql",
            content="SELECT ?s WHERE { ?s ?p ?o }",
            headers={"Content-Type": "application/sparql-query"},
        )

        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    def test_form_urlencoded_query_field_is_accepted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_RDF_EXPORT_DIR", str(tmp_path))
        monkeypatch.delenv("GRAPHRAG_SPARQL_ENDPOINT", raising=False)
        _write_export(tmp_path)

        resp = _client().post(
            "/sparql",
            data={"query": "SELECT ?s WHERE { ?s ?p ?o }"},
        )

        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    def test_unrecognised_content_type_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_RDF_EXPORT_DIR", str(tmp_path))
        monkeypatch.delenv("GRAPHRAG_SPARQL_ENDPOINT", raising=False)
        _write_export(tmp_path)

        resp = _client().post(
            "/sparql", content="whatever", headers={"Content-Type": "text/plain"},
        )
        assert resp.status_code == 415

    def test_empty_sparql_query_body_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_RDF_EXPORT_DIR", str(tmp_path))
        _write_export(tmp_path)
        resp = _client().post(
            "/sparql", content="", headers={"Content-Type": "application/sparql-query"},
        )
        assert resp.status_code == 400


class TestResponseContentNegotiation:
    def test_sparql_results_json_accept_returns_w3c_shape(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_RDF_EXPORT_DIR", str(tmp_path))
        monkeypatch.delenv("GRAPHRAG_SPARQL_ENDPOINT", raising=False)
        _write_export(tmp_path)

        resp = _client().post(
            "/sparql",
            json={"query": "SELECT ?s ?o WHERE { ?s ?p ?o }"},
            headers={"Accept": "application/sparql-results+json"},
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/sparql-results+json")
        body = resp.json()
        # W3C shape: {"head": {"vars": [...]}, "results": {"bindings": [...]}}
        # -- the legacy shape has no "head"/"results" keys at all.
        assert "head" in body
        assert "results" in body
        assert "bindings" in body["results"]
        binding = body["results"]["bindings"][0]
        assert set(binding["s"]) >= {"type", "value"}

    def test_sparql_results_csv_accept_returns_csv(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_RDF_EXPORT_DIR", str(tmp_path))
        monkeypatch.delenv("GRAPHRAG_SPARQL_ENDPOINT", raising=False)
        _write_export(tmp_path)

        resp = _client().post(
            "/sparql",
            json={"query": "SELECT ?s WHERE { ?s ?p ?o }"},
            headers={"Accept": "text/csv"},
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")
        assert "s\r\n" in resp.text or "s\n" in resp.text


class TestRemoteBackedReads:
    def test_remote_endpoint_json_accept_is_passed_through_verbatim(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_SPARQL_ENDPOINT", "http://store.example/sparql")
        raw_payload = {
            "head": {"vars": ["s"]},
            "results": {"bindings": [{"s": {"type": "uri", "value": "http://x"}}]},
        }
        mock_endpoint = AsyncMock()
        mock_endpoint.query_json_raw = AsyncMock(return_value=raw_payload)

        with patch(
            "graphrag.graph.triplestore.remote_sparql_source_from_env",
            return_value=mock_endpoint,
        ):
            resp = _client().post(
                "/sparql",
                json={"query": "SELECT ?s WHERE { ?s ?p ?o }"},
                headers={"Accept": "application/sparql-results+json"},
            )

        assert resp.status_code == 200
        assert resp.json() == raw_payload
        mock_endpoint.query_json_raw.assert_awaited_once()

    def test_remote_endpoint_legacy_accept_uses_flattened_rows(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_SPARQL_ENDPOINT", "http://store.example/sparql")
        mock_endpoint = AsyncMock()
        mock_endpoint.query = AsyncMock(return_value=[{"s": "http://x"}])

        with patch(
            "graphrag.graph.triplestore.remote_sparql_source_from_env",
            return_value=mock_endpoint,
        ):
            resp = _client().post("/sparql", json={"query": "SELECT ?s WHERE { ?s ?p ?o }"})

        assert resp.status_code == 200
        assert resp.json() == {"rows": [{"s": "http://x"}], "count": 1}

    def test_remote_endpoint_xml_accept_is_501_not_a_silent_conversion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_SPARQL_ENDPOINT", "http://store.example/sparql")
        mock_endpoint = AsyncMock()

        with patch(
            "graphrag.graph.triplestore.remote_sparql_source_from_env",
            return_value=mock_endpoint,
        ):
            resp = _client().post(
                "/sparql",
                json={"query": "SELECT ?s WHERE { ?s ?p ?o }"},
                headers={"Accept": "application/sparql-results+xml"},
            )

        assert resp.status_code == 501

    def test_local_snapshot_is_used_when_no_remote_endpoint_is_configured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_RDF_EXPORT_DIR", str(tmp_path))
        monkeypatch.delenv("GRAPHRAG_SPARQL_ENDPOINT", raising=False)
        _write_export(tmp_path)

        resp = _client().post("/sparql", json={"query": "SELECT ?s WHERE { ?s ?p ?o }"})
        assert resp.status_code == 200
        assert resp.json()["count"] == 1


class TestUpdateStillTargetsLocalSnapshotWhenRemoteConfigured:
    def test_update_persists_to_the_local_file_not_the_remote_store(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_RDF_EXPORT_DIR", str(tmp_path))
        monkeypatch.setenv("GRAPHRAG_SPARQL_ENDPOINT", "http://store.example/sparql")
        _write_export(tmp_path)
        mock_endpoint = AsyncMock()

        with patch(
            "graphrag.graph.triplestore.remote_sparql_source_from_env",
            return_value=mock_endpoint,
        ):
            resp = _client().post(
                "/sparql/update",
                json={
                    "query": 'INSERT DATA { <http://example.org/bob> <http://example.org/name> "Bob" }',
                    "persist": True,
                },
            )

        assert resp.status_code == 200
        # No network call to the "remote" store for the update itself.
        mock_endpoint.query.assert_not_called()
        mock_endpoint.update.assert_not_called()
        persisted = (tmp_path / "acme" / "graph_export.ttl").read_text()
        assert "Bob" in persisted
