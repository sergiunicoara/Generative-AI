from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from pydantic import ValidationError

from api.auth.dependencies import get_current_user
from api.routes import ingest as ingest_routes


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(ingest_routes.router, prefix="/ingest")
    app.dependency_overrides[get_current_user] = lambda: {
        "scope": "write", "sub": "test", "tenant": "test-tenant",
    }
    return TestClient(app)


@pytest.mark.parametrize(
    "payload",
    [
        {"filename": "", "text": "content"},
        {"filename": "doc.txt", "text": ""},
        {"filename": "doc.txt", "text": "content", "priority": "urgent"},
        {"filename": "x" * 256, "text": "content"},
        {"filename": "doc.txt", "text": "content", "metadata": {str(i): i for i in range(101)}},
    ],
)
def test_ingest_model_rejects_invalid_or_unbounded_inputs(payload) -> None:
    with pytest.raises(ValidationError):
        ingest_routes.IngestRequest(**payload)


def test_queue_failure_does_not_expose_internal_connection_detail() -> None:
    client = _client()
    with patch(
        "api.routes.ingest.publish_document",
        new=AsyncMock(side_effect=RuntimeError("amqp://user:secret@internal")),
    ):
        response = client.post("/ingest", json={"filename": "doc.txt", "text": "content"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Queue unavailable"}
    assert "secret" not in response.text
