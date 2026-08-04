"""§11 required API surface: /health, /ready, POST /api/v1/ingestions/crm,
POST /api/v1/ingestions/content-assets, GET /api/v1/ingestions/{id}. Uses
httpx.AsyncClient + ASGITransport (not fastapi.testclient.TestClient, whose
blocking portal conflicts with already being inside pytest-asyncio's event
loop) against the real Neo4j container.
"""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest

from api.main import app

pytestmark = pytest.mark.asyncio


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_health_is_always_ok():
    async with _client() as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_ready_reports_ready_after_schema_bootstrap(executor):
    async with _client() as client:
        resp = await client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


async def test_crm_ingestion_requires_workspace_header():
    async with _client() as client:
        resp = await client.post("/api/v1/ingestions/crm", json={"accounts": []})
    assert resp.status_code == 422  # missing required X-Workspace-Id header


async def test_crm_ingestion_end_to_end_and_status_lookup(executor):
    workspace_id = f"ws-api-{uuid4().hex[:8]}"
    account_raw = {
        "Id": "001xxAPI1", "Name": "API Test Corp", "Website": "api-test.com",
        "IsDeleted": False, "MasterRecordId": None,
    }

    async with _client() as client:
        post_resp = await client.post(
            "/api/v1/ingestions/crm",
            headers={"X-Workspace-Id": workspace_id},
            json={"accounts": [account_raw]},
        )
        assert post_resp.status_code == 202
        body = post_resp.json()
        assert body["state"] == "COMPLETED"
        ingestion_id = body["ingestion_id"]

        get_resp = await client.get(
            f"/api/v1/ingestions/{ingestion_id}", headers={"X-Workspace-Id": workspace_id}
        )
        assert get_resp.status_code == 200
        get_body = get_resp.json()
        assert get_body["state"] == "COMPLETED"
        assert get_body["item_results"][0]["outcome"] == "CREATED"

        # a different workspace must never see this job — 404, not 403 (a 403
        # would confirm the id's existence to a caller who doesn't own it).
        cross_resp = await client.get(
            f"/api/v1/ingestions/{ingestion_id}", headers={"X-Workspace-Id": "some-other-workspace"}
        )
        assert cross_resp.status_code == 404


async def test_content_asset_ingestion_end_to_end(executor):
    workspace_id = f"ws-api-content-{uuid4().hex[:8]}"
    asset_raw = {
        "id": "asset-1", "title": "Pricing Objection Handling",
        "url": "https://showpad.example/asset-1", "type": "pdf",
        "tags": ["pricing", "objection"],
    }

    async with _client() as client:
        post_resp = await client.post(
            "/api/v1/ingestions/content-assets",
            headers={"X-Workspace-Id": workspace_id},
            json={"division_id": "division-a", "content_assets": [asset_raw]},
        )
        assert post_resp.status_code == 202
        body = post_resp.json()
        assert body["state"] == "COMPLETED"

        get_resp = await client.get(
            f"/api/v1/ingestions/{body['ingestion_id']}", headers={"X-Workspace-Id": workspace_id}
        )
        assert get_resp.json()["item_results"][0]["outcome"] == "CREATED"


async def test_transcript_ingestion_end_to_end():
    workspace_id = f"ws-api-transcript-{uuid4().hex[:8]}"
    raw_call = {
        "id": "call-api-1", "started": "2026-06-15T14:00:00Z", "deleted": False,
        "parties": [{"speakerId": "spk_1", "name": "Buyer", "emailAddress": "buyer@example.com"}],
        "transcript": [
            {"speakerId": "spk_1", "sentences": [{"text": "We are concerned about pricing.", "start": 0, "end": 2000}]},
        ],
    }

    async with _client() as client:
        post_resp = await client.post(
            "/api/v1/ingestions/transcripts",
            headers={"X-Workspace-Id": workspace_id},
            json={"calls": [raw_call]},
        )
        assert post_resp.status_code == 202
        body = post_resp.json()
        assert body["state"] == "COMPLETED"

        get_resp = await client.get(
            f"/api/v1/ingestions/{body['ingestion_id']}", headers={"X-Workspace-Id": workspace_id}
        )
        item = get_resp.json()["item_results"][0]
        assert item["outcome"] == "CREATED"
        assert item["claims_created"] > 0


async def test_get_unknown_ingestion_is_404():
    async with _client() as client:
        resp = await client.get("/api/v1/ingestions/does-not-exist", headers={"X-Workspace-Id": "ws-x"})
    assert resp.status_code == 404
