"""§11 required API: POST /api/v1/context/build, GET /api/v1/claims/{id}/evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import httpx
import pytest

from api.main import app
from src.domain.assertion import Claim
from src.domain.enums import AdjudicationStatus, Polarity, SpeakerRole
from src.graph.repositories.claim_repository import ClaimRepository

pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 6, 15, tzinfo=timezone.utc)


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_context_build_and_claim_evidence_endpoints(executor):
    workspace_id = f"ws-ctx-api-{uuid4().hex[:8]}"
    claim_repo = ClaimRepository(executor)
    claim = Claim(
        claim_id="claim-api-1", workspace_id=workspace_id, subject_id="contact-api",
        predicate="RAISED_OBJECTION", object_value="pricing", polarity=Polarity.AFFIRMED,
        source_type="transcript", evidence_char_start=0, evidence_char_end=5,
        source_timestamp=_T0, speaker_role=SpeakerRole.BUYER, confidence=0.9,
        valid_from=_T0, transaction_from=_T0, adjudication_status=AdjudicationStatus.UNREVIEWED,
        retention_class="standard", created_at=_T0,
    )
    await claim_repo.create_claim(claim)

    async with _client() as client:
        build_resp = await client.post(
            "/api/v1/context/build", headers={"X-Workspace-Id": workspace_id},
            json={"subject_id": "contact-api"},
        )
        assert build_resp.status_code == 200
        body = build_resp.json()
        assert body["nodes_used"] == 1
        assert body["claims"][0]["claim_id"] == "claim-api-1"

        evidence_resp = await client.get(
            "/api/v1/claims/claim-api-1/evidence", headers={"X-Workspace-Id": workspace_id}
        )
        assert evidence_resp.status_code == 200
        assert evidence_resp.json()["predicate"] == "RAISED_OBJECTION"


async def test_claim_evidence_unknown_claim_is_404():
    async with _client() as client:
        resp = await client.get("/api/v1/claims/does-not-exist/evidence", headers={"X-Workspace-Id": "ws-x"})
    assert resp.status_code == 404


async def test_context_build_workspace_id_comes_from_header_not_body():
    """§12: workspace_id from authenticated context, never the request body —
    ContextBuildRequest has no workspace_id field to smuggle one through."""
    async with _client() as client:
        resp = await client.post(
            "/api/v1/context/build", headers={"X-Workspace-Id": "ws-real"},
            json={"subject_id": "contact-x", "workspace_id": "ws-attacker-supplied"},
        )
    assert resp.status_code == 200  # extra unknown field is ignored, not honored
    assert resp.json()["workspace_id"] == "ws-real"
