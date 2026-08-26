"""Authorization and tenant-boundary tests for enterprise API routes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes import enterprise as enterprise_routes


def _client(scope: str = "write", tenant: str = "acme") -> TestClient:
    app = FastAPI()
    app.include_router(enterprise_routes.router)
    app.dependency_overrides[get_current_user] = lambda: {
        "scope": scope,
        "sub": "reviewer-1",
        "tenant": tenant,
    }
    return TestClient(app)


def test_register_schema_accepts_and_binds_the_authenticated_tenant():
    service = MagicMock()
    service.register_schema = AsyncMock(return_value={"id": "schema-1", "tenant": "acme"})

    with patch("api.routes.enterprise.MetadataGovernanceService", return_value=service):
        response = _client().post("/governance/schemas", json={
            "collection": "contracts",
            "version": "v1",
            "status": "active",
            "tenant": "acme",
        })

    assert response.status_code == 200
    registered_schema = service.register_schema.await_args.args[0]
    assert registered_schema.tenant == "acme"


def test_schema_registration_rejects_a_cross_tenant_request():
    response = _client().post("/governance/schemas", json={
        "collection": "contracts",
        "version": "v1",
        "tenant": "another-tenant",
    })

    assert response.status_code == 403


def test_sync_changes_uses_the_authenticated_tenant():
    service = MagicMock()
    service.apply_changes = AsyncMock(return_value={"queued": 1, "tombstoned": 0})
    payload = {
        "cursor": "delta-42",
        "changes": [{
            "change_type": "upsert",
            "external_id": "item-42",
            "filename": "contract.pdf",
            "text": "An executed contract.",
        }],
    }

    with patch("api.routes.enterprise.ContentSyncService", return_value=service):
        response = _client().post("/sync/sharepoint/changes", json=payload)

    assert response.status_code == 200
    args, kwargs = service.apply_changes.await_args
    assert args[0] == "sharepoint"
    assert args[2] == "acme"
    assert args[1][0].external_id == "item-42"
    assert kwargs == {"cursor": "delta-42", "trigger": "delta"}


def test_review_approval_uses_authenticated_reviewer_and_tenant():
    service = MagicMock()
    service.approve_lineage = AsyncMock(return_value={"status": "approved"})

    with patch("api.routes.enterprise.LineageService", return_value=service):
        response = _client().post("/lineage/reviews/review-1/approve")

    assert response.status_code == 200
    service.approve_lineage.assert_awaited_once_with("review-1", "reviewer-1", "acme")


def test_read_endpoints_require_read_scope():
    assert _client(scope="write").get("/governance/coverage").status_code == 403


def test_sharepoint_run_is_tenant_bound():
    connector = MagicMock()
    connector.config.tenant = "acme"
    connector.sync_once = AsyncMock(return_value={"queued": 2, "source_id": "legal-sharepoint"})

    with patch("api.routes.enterprise.SharePointSyncConnector.from_settings", return_value=connector):
        response = _client().post("/sync/sharepoint/legal-sharepoint/run")

    assert response.status_code == 200
    connector.sync_once.assert_awaited_once()
