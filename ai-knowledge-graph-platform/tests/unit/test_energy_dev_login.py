"""Regression coverage for the local Energy workspace login path."""

from __future__ import annotations

from unittest.mock import patch

from fastapi import Depends, FastAPI
from starlette.middleware.sessions import SessionMiddleware
from starlette.testclient import TestClient

from api.auth.dependencies import get_tenant
from api.routes import auth as auth_routes


def _client() -> TestClient:
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="test-secret")
    app.include_router(auth_routes.router, prefix="/auth")

    @app.get("/energy-demo")
    async def energy_demo(tenant: str = Depends(get_tenant)) -> dict[str, str]:
        return {"tenant": tenant}

    return TestClient(app)


def test_dev_login_for_energy_workspace_issues_energy_demo_tenant_cookie():
    """The documented local URL must not authenticate into the default tenant."""
    with patch("api.routes.auth.is_dev_env", return_value=True):
        response = _client().get("/auth/dev-login?next=/energy-demo")

    assert response.status_code == 200
    assert response.json() == {"tenant": "energy-demo"}
