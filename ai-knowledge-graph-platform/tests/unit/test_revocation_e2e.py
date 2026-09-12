"""Token revocation, proven through the real HTTP boundary.

tests/unit/test_jwt_signing_and_revocation.py already covers
``TokenRevocationStore``/``assert_not_revoked`` at the library level -- this
file closes the gap a follow-up platform review named explicitly: nothing
exercised revocation through ``RequireAuthMiddleware`` + the real
``POST /auth/revoke`` route + a subsequent call carrying the same token.

Builds a minimal app the same way test_auth_provisioning_routes.py does
(mount the real router, no full api.main import -- that needs live
Neo4j/RabbitMQ/Redis at lifespan time this file has no business depending
on), but here with ``RequireAuthMiddleware`` genuinely installed, since the
whole point is proving revocation is honoured at the boundary that actually
decodes and checks it, not just calling ``is_revoked()`` directly.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI
from starlette.testclient import TestClient

from api.auth import user_provisioning as up
from api.auth.default_auth import RequireAuthMiddleware
from api.auth.dependencies import require_scope
from api.auth.jwt import create_access_token
from api.routes import auth as auth_routes
from graphrag.core import token_revocation as revocation_module
from graphrag.core.token_revocation import TokenRevocationStore


@pytest.fixture(autouse=True)
def _in_memory_revocation_store():
    """Force the process-singleton revocation store to a fresh, in-memory-only
    instance for each test.

    This dev environment may have a real Redis configured even under
    ENV=test (see test_auth_provisioning_routes.py's module docstring for
    the identical concern with the user-provisioning table) -- constructing
    the store directly with redis_url=None, rather than going through
    get_revocation_store()'s env-driven lookup, keeps this test hermetic and
    unable to leak revocations into or read them from a shared Redis.
    """
    store = TokenRevocationStore(redis_url=None)
    revocation_module._store = store
    yield store
    revocation_module._store = None


@pytest.fixture(autouse=True)
def _in_memory_client_and_user_registries():
    """Same non-hermeticity concern as the revocation store above, for the
    two other stores this file's cross-tenant test touches: the M2M client
    registry (auth.py's own `_get_redis_sync`) and the provisioned-user
    table (user_provisioning.py's separate `_get_redis_sync`) -- see
    test_auth_provisioning_routes.py's docstring for why this dev
    environment cannot be assumed to have no Redis configured."""
    auth_routes._m2m_clients_mem.clear()
    up._users_mem.clear()
    up._identities_mem.clear()
    with patch("api.routes.auth._get_redis_sync", return_value=None), \
         patch("api.auth.user_provisioning._get_redis_sync", return_value=None):
        yield
    auth_routes._m2m_clients_mem.clear()
    up._users_mem.clear()
    up._identities_mem.clear()


def _make_app() -> FastAPI:
    app = FastAPI()
    # Order matches api/main.py: RequireAuthMiddleware is the real deny-by-
    # default floor this test means to exercise, not a stand-in.
    app.add_middleware(RequireAuthMiddleware)
    app.include_router(auth_routes.router, prefix="/auth")

    @app.get("/protected")
    async def protected(user: dict = Depends(require_scope("read"))):
        return {"sub": user.get("sub")}

    return app


def _mint(*, tenant: str = "acme", scope: str = "read", sub: str = "client-1") -> str:
    return create_access_token({"sub": sub, "tenant": tenant, "scope": scope, "type": "m2m"})


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class TestRevocationThroughTheRealHTTPBoundary:
    def test_a_freshly_minted_token_is_accepted(self):
        client = TestClient(_make_app())
        token = _mint()
        assert client.get("/protected", headers=_bearer(token)).status_code == 200

    def test_revoking_by_token_denies_the_same_token_on_the_next_call(self):
        client = TestClient(_make_app())
        token = _mint(sub="client-1")
        assert client.get("/protected", headers=_bearer(token)).status_code == 200

        admin = _mint(scope="admin", sub="admin-1")
        revoke_resp = client.post(
            "/auth/revoke",
            json={"token": token, "reason": "leaked in a log"},
            headers=_bearer(admin),
        )
        assert revoke_resp.status_code == 200
        assert revoke_resp.json() == {"revoked_tokens": 1, "revoked_subjects": 0, "durable": False}

        denied = client.get("/protected", headers=_bearer(token))
        assert denied.status_code == 401

    def test_revoking_by_jti_alone_denies_the_same_token(self):
        """RFC 7009 names `token`; an operator holding only a log line (which
        records `jti`, never the credential itself) must be able to act
        without the token back -- see TokenRevokeRequest's docstring."""
        client = TestClient(_make_app())
        token = _mint(sub="client-1")
        from api.auth.jwt import decode_access_token

        jti = decode_access_token(token)["jti"]
        assert client.get("/protected", headers=_bearer(token)).status_code == 200

        admin = _mint(scope="admin", sub="admin-1")
        revoke_resp = client.post("/auth/revoke", json={"jti": jti}, headers=_bearer(admin))
        assert revoke_resp.status_code == 200

        assert client.get("/protected", headers=_bearer(token)).status_code == 401

    def test_revoking_by_subject_denies_every_token_issued_before_the_cutoff(self):
        client = TestClient(_make_app())
        token = _mint(sub="client-2")
        assert client.get("/protected", headers=_bearer(token)).status_code == 200

        # `iat` has 1-second granularity (JWT numeric-date semantics) while
        # the revocation cutoff is a sub-second time.time(); crossing a
        # whole second on each side is what makes "before" vs. "after" the
        # cutoff unambiguous, rather than racing the clock.
        time.sleep(1.1)
        admin = _mint(scope="admin", sub="admin-1")
        revoke_resp = client.post(
            "/auth/revoke",
            json={"subject": "client-2", "reason": "compromised client"},
            headers=_bearer(admin),
        )
        assert revoke_resp.status_code == 200
        assert revoke_resp.json()["revoked_subjects"] == 1

        assert client.get("/protected", headers=_bearer(token)).status_code == 401

        # Subject revocation is a cutoff, not a permanent ban -- a token
        # minted after the revocation call must still work.
        time.sleep(1.1)
        fresh_token = _mint(sub="client-2")
        assert client.get("/protected", headers=_bearer(fresh_token)).status_code == 200

    def test_non_admin_cannot_call_revoke(self):
        client = TestClient(_make_app())
        token = _mint(sub="client-3")
        non_admin = _mint(scope="read", sub="not-admin")
        resp = client.post("/auth/revoke", json={"token": token}, headers=_bearer(non_admin))
        assert resp.status_code == 403
        # Untouched: no admin scope means no revocation happened.
        assert client.get("/protected", headers=_bearer(token)).status_code == 200

    def test_revoke_by_token_rejects_a_cross_tenant_token(self):
        """Adversarial, mirrors test_auth_provisioning_routes.py's
        TestRevokeUser.test_cannot_revoke_another_tenants_user: an admin for
        tenant B must not be able to revoke a token belonging to tenant A,
        even holding the literal token string."""
        client = TestClient(_make_app())
        token = _mint(tenant="tenant-a", sub="client-4")
        assert client.get("/protected", headers=_bearer(token)).status_code == 200

        admin_b = _mint(tenant="tenant-b", scope="admin", sub="admin-b")
        resp = client.post("/auth/revoke", json={"token": token}, headers=_bearer(admin_b))
        assert resp.status_code == 403

        assert client.get("/protected", headers=_bearer(token)).status_code == 200

    def test_revoke_by_subject_rejects_a_cross_tenant_m2m_client(self):
        """The gap this session found while writing this test: revoking by
        `subject` alone previously had NO tenant check at all -- unlike the
        `token` path above. An admin for tenant B who merely knew (from a
        log line, a shared client id) tenant A's client_id could silently
        log every one of tenant A's sessions out. Registers a real M2M
        client via POST /auth/clients (rather than hand-rolling a client_id)
        so this proves the fix against the actual registry lookup
        (`_subject_tenant`), not an assumption about its shape."""
        client = TestClient(_make_app())
        # register_client intersects requested scopes with the caller's own
        # (see its docstring) -- the caller needs "read" itself, not just
        # "write", to grant the new client a "read" scope.
        admin_a = _mint(tenant="tenant-a", scope="write read", sub="admin-a")
        register_resp = client.post(
            "/auth/clients",
            json={"client_name": "tenant-a-service", "scopes": ["read"]},
            headers=_bearer(admin_a),
        )
        assert register_resp.status_code == 200
        tenant_a_client_id = register_resp.json()["client_id"]

        admin_b = _mint(tenant="tenant-b", scope="admin", sub="admin-b")
        resp = client.post(
            "/auth/revoke",
            json={"subject": tenant_a_client_id, "reason": "guessed the client id"},
            headers=_bearer(admin_b),
        )
        assert resp.status_code == 403

        # Untouched: a token for that subject, issued after the rejected
        # revocation attempt, must still be honoured.
        tenant_a_token = _mint(tenant="tenant-a", sub=tenant_a_client_id)
        assert client.get("/protected", headers=_bearer(tenant_a_token)).status_code == 200

    def test_revoke_by_subject_still_works_for_an_unknown_subject(self):
        """Fail-open only when the subject can't be identified in either
        registry -- same "safe-by-default choice is the available one"
        philosophy TokenRevocationStore's own docstring states for the
        Redis-outage case. An operator revoking a subject with no client or
        Google-identity record (e.g. an already-deleted client) must not be
        blocked by this check."""
        client = TestClient(_make_app())
        admin = _mint(scope="admin", sub="admin-1")
        resp = client.post(
            "/auth/revoke",
            json={"subject": "never-registered-anywhere"},
            headers=_bearer(admin),
        )
        assert resp.status_code == 200
        assert resp.json()["revoked_subjects"] == 1

    def test_own_tenants_m2m_client_can_still_be_revoked_by_subject(self):
        """The fix must not turn into a same-tenant false positive."""
        client = TestClient(_make_app())
        admin = _mint(tenant="acme", scope="admin write read", sub="admin-1")
        register_resp = client.post(
            "/auth/clients",
            json={"client_name": "acme-service", "scopes": ["read"]},
            headers=_bearer(admin),
        )
        client_id = register_resp.json()["client_id"]

        # Issued before the revocation call -- subject revocation is a
        # cutoff (see the docstring on TokenRevocationStore.revoke_subject),
        # so only a token that predates it should be denied afterward.
        token = _mint(tenant="acme", sub=client_id)
        assert client.get("/protected", headers=_bearer(token)).status_code == 200

        resp = client.post(
            "/auth/revoke",
            json={"subject": client_id},
            headers=_bearer(admin),
        )
        assert resp.status_code == 200
        assert resp.json()["revoked_subjects"] == 1

        assert client.get("/protected", headers=_bearer(token)).status_code == 401
