"""Authorization + happy-path coverage for the document-catalog read API
(api/routes/kg/catalog.py). Mirrors test_taxonomy_route_security.py's
convention: mount only catalog.router into a throwaway FastAPI() app,
override get_current_user, plain Starlette TestClient calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes.kg import catalog as catalog_routes


def _client(scope: str, tenant: str = "acme") -> TestClient:
    app = FastAPI()
    app.include_router(catalog_routes.router)
    app.dependency_overrides[get_current_user] = lambda: {
        "sub": "user-1", "scope": scope, "tenant": tenant,
    }
    return TestClient(app)


class TestAuthorization:
    def test_list_without_read_scope_is_forbidden(self) -> None:
        response = _client("write").get("/catalog/documents")
        assert response.status_code == 403

    def test_detail_without_read_scope_is_forbidden(self) -> None:
        response = _client("write").get("/catalog/documents/doc-1")
        assert response.status_code == 403


class TestListCatalogDocuments:
    def test_happy_path_passes_filters_and_tenant_through(self) -> None:
        fake_client = AsyncMock()
        fake_client.get_documents_catalog = AsyncMock(return_value=[
            {"id": "doc-1", "filename": "AD-2024-01.pdf", "collection": "regulatory"},
        ])
        with patch.object(catalog_routes, "get_neo4j", return_value=fake_client):
            response = _client("read", tenant="acme").get(
                "/catalog/documents",
                params={"collection": "regulatory", "limit": 10, "offset": 0},
            )

        assert response.status_code == 200
        assert response.json() == [{"id": "doc-1", "filename": "AD-2024-01.pdf", "collection": "regulatory"}]
        fake_client.get_documents_catalog.assert_awaited_once_with(
            tenant="acme", collection="regulatory", classification=None,
            source_system=None, limit=10, offset=0,
        )


class TestGetCatalogDocument:
    def test_happy_path_returns_detail(self) -> None:
        fake_client = AsyncMock()
        fake_client.get_document_catalog_detail = AsyncMock(return_value={
            "id": "doc-1", "filename": "AD-2024-01.pdf", "ingestion_runs": [],
        })
        with patch.object(catalog_routes, "get_neo4j", return_value=fake_client):
            response = _client("read", tenant="acme").get("/catalog/documents/doc-1")

        assert response.status_code == 200
        assert response.json()["id"] == "doc-1"
        fake_client.get_document_catalog_detail.assert_awaited_once_with("doc-1", tenant="acme")

    def test_missing_document_is_404(self) -> None:
        fake_client = AsyncMock()
        fake_client.get_document_catalog_detail = AsyncMock(return_value=None)
        with patch.object(catalog_routes, "get_neo4j", return_value=fake_client):
            response = _client("read", tenant="acme").get("/catalog/documents/nonexistent")

        assert response.status_code == 404
