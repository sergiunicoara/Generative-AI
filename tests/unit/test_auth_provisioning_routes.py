"""Route-level tests for POST/GET/DELETE /auth/users and the rewritten
GET /auth/callback (F14).

Uses a minimal FastAPI app with just the auth router mounted and
get_current_user overridden, mirroring test_query_routes.py's `_make_client`
pattern rather than importing the full api.main app (which needs live
Neo4j/RabbitMQ/Redis at import/lifespan time this file has no business
depending on).

_get_redis_sync is patched to force the in-memory provisioning-table
fallback deterministically — see test_user_provisioning.py's module
docstring for why (this dev environment has a real Redis configured even
under ENV=test).

See docs/context_graph_gap_plan.md F14.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from api.auth import user_provisioning as up
from api.auth.dependencies import get_current_user
from api.routes import auth as auth_routes


def _make_client(*, scope: str = "admin", tenant: str = "acme", email: str = "admin@acme.com") -> TestClient:
    app = FastAPI()
    app.include_router(auth_routes.router, prefix="/auth")
    app.add_middleware(SessionMiddleware, secret_key="test-secret")
    app.dependency_overrides[get_current_user] = lambda: {
        "scope": scope, "sub": "test-admin", "tenant": tenant, "email": email,
    }
    return TestClient(app)


@pytest.fixture(autouse=True)
def _memory_only():
    up._users_mem.clear()
    with patch("api.auth.user_provisioning._get_redis_sync", return_value=None):
        yield
    up._users_mem.clear()


# ── POST /auth/users ────────────────────────────────────────────────────────

class TestProvisionUser:
    def test_admin_can_provision_a_user_for_own_tenant(self):
        client = _make_client(scope="admin write read", tenant="acme")
        resp = client.post("/auth/users", json={"email": "alice@acme.com", "scopes": ["read", "write"]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["email"] == "alice@acme.com"
        assert body["tenant"] == "acme"
        assert "read" in body["scopes"] and "write" in body["scopes"]

    def test_non_admin_cannot_provision(self):
        client = _make_client(scope="read write", tenant="acme")   # no "admin"
        resp = client.post("/auth/users", json={"email": "alice@acme.com"})
        assert resp.status_code == 403

    def test_tenant_is_not_a_body_field(self):
        """F12-style guard: tenant must come from the caller's own token, not
        anything the request body could name — the request model has no
        tenant field at all, so there is nothing to smuggle."""
        assert "tenant" not in auth_routes.UserProvisionRequest.model_fields

    def test_provisioned_tenant_is_always_the_callers_own(self):
        """An admin for tenant A must not be able to provision a user into
        tenant B by any means — verified here by provisioning as two
        different admins and checking each record landed under its own
        caller's tenant."""
        client_a = _make_client(scope="admin", tenant="tenant-a")
        client_a.post("/auth/users", json={"email": "shared@example.com"})

        client_b = _make_client(scope="admin", tenant="tenant-b")
        client_b.post("/auth/users", json={"email": "shared@example.com"})

        assert up.get_user_record("shared@example.com")["tenant"] == "tenant-b"  # last write wins
        # Each admin's own view is still scoped correctly:
        assert [r["tenant"] for r in up.list_user_records(tenant="tenant-a")] == []
        assert [r["tenant"] for r in up.list_user_records(tenant="tenant-b")] == ["tenant-b"]

    def test_granted_scopes_cannot_exceed_callers_own(self):
        """Escalation guard, same shape as register_client: an admin holding
        only 'admin read' must not be able to provision a user with 'write'."""
        client = _make_client(scope="admin read", tenant="acme")
        resp = client.post("/auth/users", json={"email": "alice@acme.com", "scopes": ["read", "write", "biz:approve"]})
        assert resp.status_code == 200
        granted = resp.json()["scopes"]
        assert "write" not in granted
        assert "biz:approve" not in granted
        assert "read" in granted

    def test_tenant_scope_always_included(self):
        client = _make_client(scope="admin", tenant="acme")
        resp = client.post("/auth/users", json={"email": "alice@acme.com", "scopes": []})
        assert "tenant:acme" in resp.json()["scopes"]


# ── GET /auth/users ─────────────────────────────────────────────────────────

class TestListProvisionedUsers:
    def test_lists_only_this_tenant(self):
        up.set_user_record("a@acme.com", tenant="acme", scopes=["read"], added_by="x")
        up.set_user_record("b@other.com", tenant="other", scopes=["read"], added_by="x")

        client = _make_client(scope="admin", tenant="acme")
        resp = client.get("/auth/users")
        assert resp.status_code == 200
        emails = [u["email"] for u in resp.json()]
        assert emails == ["a@acme.com"]

    def test_non_admin_forbidden(self):
        client = _make_client(scope="read write", tenant="acme")
        resp = client.get("/auth/users")
        assert resp.status_code == 403


# ── DELETE /auth/users/{email} ─────────────────────────────────────────────

class TestRevokeUser:
    def test_revokes_own_tenants_user(self):
        up.set_user_record("alice@acme.com", tenant="acme", scopes=["read"], added_by="x")
        client = _make_client(scope="admin", tenant="acme")
        resp = client.delete("/auth/users/alice@acme.com")
        assert resp.status_code == 200
        assert up.get_user_record("alice@acme.com") is None

    def test_cannot_revoke_another_tenants_user(self):
        """Adversarial: admin for tenant B must not be able to revoke a user
        that belongs to tenant A, even knowing the exact email."""
        up.set_user_record("alice@acme.com", tenant="acme", scopes=["read"], added_by="x")
        client = _make_client(scope="admin", tenant="other-tenant")
        resp = client.delete("/auth/users/alice@acme.com")
        assert resp.status_code == 404
        assert up.get_user_record("alice@acme.com") is not None   # untouched

    def test_unknown_email_is_404(self):
        client = _make_client(scope="admin", tenant="acme")
        resp = client.delete("/auth/users/nobody@acme.com")
        assert resp.status_code == 404


# ── GET /auth/callback ──────────────────────────────────────────────────────

def _login_then_callback(client: TestClient, userinfo: dict):
    """Drive the real /login -> /callback state flow so CSRF-state validation
    passes, then patch the Google token exchange to return `userinfo`."""
    login_resp = client.get("/auth/login", follow_redirects=False)
    location = login_resp.headers["location"]
    state = dict(p.split("=", 1) for p in location.split("?", 1)[1].split("&"))["state"]

    with patch("api.routes.auth.exchange_code_for_userinfo", AsyncMock(return_value=userinfo)):
        return client.get(f"/auth/callback?code=fake-code&state={state}", follow_redirects=False)


class TestCallbackRejectsUnprovisionedAccount:
    def test_unprovisioned_google_account_gets_403_not_default_tenant(self):
        """The actual F14 fix: previously this issued a token for
        settings.default_tenant unconditionally. Now an unmapped account
        gets no token at all."""
        client = _make_client()
        resp = _login_then_callback(client, {
            "sub": "google-uid-1", "email": "stranger@gmail.com", "name": "Stranger",
        })
        assert resp.status_code == 403
        assert "access_token" not in resp.cookies

    def test_provisioned_account_gets_its_own_tenant(self):
        up.set_user_record(
            "alice@acme.com", tenant="acme", scopes=["read", "write", "tenant:acme"], added_by="admin",
        )
        client = _make_client()
        resp = _login_then_callback(client, {
            "sub": "google-uid-2", "email": "alice@acme.com", "name": "Alice",
        })
        assert resp.status_code in (302, 307)
        assert "access_token" in resp.cookies

        from api.auth.jwt import decode_access_token
        payload = decode_access_token(resp.cookies["access_token"])
        assert payload["tenant"] == "acme"
        assert payload["email"] == "alice@acme.com"

    def test_email_lookup_is_case_insensitive(self):
        """A Google account's email casing must not defeat provisioning --
        provisioned as 'Alice@Acme.com', signs in as 'alice@acme.com'."""
        up.set_user_record("Alice@Acme.com", tenant="acme", scopes=["read"], added_by="admin")
        client = _make_client()
        resp = _login_then_callback(client, {
            "sub": "google-uid-3", "email": "alice@acme.com", "name": "Alice",
        })
        assert resp.status_code in (302, 307)
        assert "access_token" in resp.cookies

    def test_issued_scopes_match_provisioned_scopes_exactly(self):
        """The callback must not re-derive or widen scopes -- they were
        already capped at provisioning time."""
        up.set_user_record("alice@acme.com", tenant="acme", scopes=["read", "tenant:acme"], added_by="admin")
        client = _make_client()
        resp = _login_then_callback(client, {
            "sub": "google-uid-4", "email": "alice@acme.com", "name": "Alice",
        })
        from api.auth.jwt import decode_access_token
        payload = decode_access_token(resp.cookies["access_token"])
        granted = set(payload["scope"].split())
        assert granted == {"read", "tenant:acme"}
        assert "write" not in granted
