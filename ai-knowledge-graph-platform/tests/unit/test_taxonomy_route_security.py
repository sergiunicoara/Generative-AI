"""Authorization coverage for the globally shared entity-type taxonomy."""

from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes.kg import knowledge as knowledge_routes


def _client(scope: str) -> TestClient:
    app = FastAPI()
    app.include_router(knowledge_routes.router)
    app.dependency_overrides[get_current_user] = lambda: {
        "sub": "user-1", "scope": scope, "tenant": "acme",
    }
    return TestClient(app)


def test_tenant_writer_cannot_modify_global_taxonomy():
    response = _client("read write").post(
        "/taxonomy/register", json={"child": "REGULATOR", "parent": "ORG"},
    )
    assert response.status_code == 403
