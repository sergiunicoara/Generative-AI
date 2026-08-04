"""§11 required API: GET /api/v1/unresolved-mentions, POST /api/v1/
unresolved-mentions/{id}/resolve."""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest

from api.main import app
from src.domain.conversation import Mention
from src.domain.enums import ResolutionStatus
from src.domain.identity import mention_id
from src.graph.repositories.review_repository import ReviewRepository

pytestmark = pytest.mark.asyncio


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_list_and_resolve_unresolved_mention(executor):
    workspace_id = f"ws-review-api-{uuid4().hex[:8]}"
    review_repo = ReviewRepository(executor)
    mention = Mention(
        mention_id=mention_id("seg-api", 0, 11, "volks wagen", "ORG"),
        workspace_id=workspace_id, segment_id="seg-api", char_start=0, char_end=11,
        surface_text="Volks Wagen", normalized_surface="volks wagen", entity_type="ORG",
        resolution_status=ResolutionStatus.PENDING_REVIEW,
    )
    await review_repo.upsert_mention(mention)

    async with _client() as client:
        list_resp = await client.get("/api/v1/unresolved-mentions", headers={"X-Workspace-Id": workspace_id})
        assert list_resp.status_code == 200
        mentions = list_resp.json()["mentions"]
        assert any(m["mention_id"] == mention.mention_id for m in mentions)

        resolve_resp = await client.post(
            f"/api/v1/unresolved-mentions/{mention.mention_id}/resolve",
            headers={"X-Workspace-Id": workspace_id},
            json={
                "reviewer_id": "reviewer@example.com",
                "selected_entity_id": "account-vw-group",
                "candidates_shown": ["account-vw-group"],
                "original_scores": {"account-vw-group": 0.84},
            },
        )
        assert resolve_resp.status_code == 200
        body = resolve_resp.json()
        assert body["selected_entity_id"] == "account-vw-group"

        list_resp_after = await client.get("/api/v1/unresolved-mentions", headers={"X-Workspace-Id": workspace_id})
        remaining_ids = {m["mention_id"] for m in list_resp_after.json()["mentions"]}
        assert mention.mention_id not in remaining_ids  # no longer pending


async def test_resolve_requires_selection_or_rejection():
    async with _client() as client:
        resp = await client.post(
            "/api/v1/unresolved-mentions/some-id/resolve",
            headers={"X-Workspace-Id": "ws-x"},
            json={"reviewer_id": "reviewer@example.com"},
        )
    assert resp.status_code == 422


async def test_resolve_unknown_mention_is_404():
    async with _client() as client:
        resp = await client.post(
            "/api/v1/unresolved-mentions/does-not-exist/resolve",
            headers={"X-Workspace-Id": "ws-x"},
            json={"reviewer_id": "reviewer@example.com", "selected_entity_id": "account-1"},
        )
    assert resp.status_code == 404
