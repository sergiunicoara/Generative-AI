"""Authorization + tenant-scoping coverage for the confidence-transition
endpoint (api/routes/kg/confidence.py). Mirrors test_catalog_routes.py's
convention: mount only confidence.router into a throwaway FastAPI() app,
override get_current_user, plain Starlette TestClient calls.

Regression coverage for the bug found in the 2026-09-23 audit: the route had
no tenant dependency, so `ConfidenceLifecycleService.transition_relation`
always fell back to its `tenant="default"` default regardless of the
caller's actual tenant, and `changed_by` was taken verbatim from the request
body, letting any caller forge the audit trail.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes.kg import confidence as confidence_routes

# ConfidenceLifecycleService is imported inside the route handler (not at
# module scope), so it must be patched where it's defined.
_SERVICE_PATH = "graphrag.graph.confidence_lifecycle.ConfidenceLifecycleService"

PAYLOAD = {
    "src_name": "Boeing 737",
    "src_type": "Aircraft",
    "relation": "MANUFACTURED_BY",
    "tgt_name": "Boeing",
    "tgt_type": "Organization",
    "target_state": "VERIFIED",
    "reason": "manual review",
}


def _client(scope: str, tenant: str = "acme", sub: str = "user-1") -> TestClient:
    app = FastAPI()
    app.include_router(confidence_routes.router)
    app.dependency_overrides[get_current_user] = lambda: {
        "sub": sub, "scope": scope, "tenant": tenant,
    }
    return TestClient(app)


class TestAuthorization:
    def test_without_write_scope_is_forbidden(self) -> None:
        response = _client("read").post("/confidence/transition", json=PAYLOAD)
        assert response.status_code == 403


class TestTenantScoping:
    def test_callers_own_tenant_is_used_not_default(self) -> None:
        fake_service = AsyncMock()
        fake_service.transition_relation = AsyncMock(return_value={"state": "VERIFIED"})
        with patch(_SERVICE_PATH, return_value=fake_service):
            response = _client("write", tenant="acme").post(
                "/confidence/transition", json=PAYLOAD,
            )

        assert response.status_code == 200
        _, kwargs = fake_service.transition_relation.await_args
        assert kwargs["tenant"] == "acme"

    def test_two_tenants_are_not_conflated(self) -> None:
        fake_service = AsyncMock()
        fake_service.transition_relation = AsyncMock(return_value={"state": "VERIFIED"})
        with patch(_SERVICE_PATH, return_value=fake_service):
            _client("write", tenant="tenant-a").post("/confidence/transition", json=PAYLOAD)
            _client("write", tenant="tenant-b").post("/confidence/transition", json=PAYLOAD)

        tenants_seen = {
            call.kwargs["tenant"] for call in fake_service.transition_relation.await_args_list
        }
        assert tenants_seen == {"tenant-a", "tenant-b"}
        assert "default" not in tenants_seen


class TestChangedByIsFromToken:
    def test_changed_by_comes_from_authenticated_user_not_body(self) -> None:
        fake_service = AsyncMock()
        fake_service.transition_relation = AsyncMock(return_value={"state": "VERIFIED"})
        forged_payload = {**PAYLOAD, "changed_by": "someone-else"}
        with patch(_SERVICE_PATH, return_value=fake_service):
            response = _client("write", tenant="acme", sub="real-caller").post(
                "/confidence/transition", json=forged_payload,
            )

        assert response.status_code == 200
        _, kwargs = fake_service.transition_relation.await_args
        assert kwargs["changed_by"] == "real-caller"
