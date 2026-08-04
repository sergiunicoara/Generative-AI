"""§12 Context Graph builder — budget enforcement, diversity caps, and
conversation-scoped retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.domain.assertion import Claim
from src.domain.enums import AdjudicationStatus, Polarity, SpeakerRole
from src.context_graph.builder import ContextGraphBuilder, ContextGraphScope
from src.graph.repositories.claim_repository import ClaimRepository

pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 6, 15, tzinfo=timezone.utc)


def _claim(claim_id: str, subject_id: str, predicate: str = "RAISED_OBJECTION", confidence: float = 0.9) -> Claim:
    return Claim(
        claim_id=claim_id, workspace_id="ws-placeholder", subject_id=subject_id,
        predicate=predicate, object_value="pricing", polarity=Polarity.AFFIRMED,
        source_type="transcript", evidence_char_start=0, evidence_char_end=5,
        source_timestamp=_T0, speaker_role=SpeakerRole.BUYER, confidence=confidence,
        valid_from=_T0, transaction_from=_T0, adjudication_status=AdjudicationStatus.UNREVIEWED,
        retention_class="standard", created_at=_T0,
    )


async def test_build_scoped_by_subject_respects_max_nodes(executor):
    workspace_id = f"ws-ctx-{uuid4().hex[:8]}"
    claim_repo = ClaimRepository(executor)
    subject_id = "contact-1"
    for i in range(5):
        await claim_repo.create_claim(_claim(f"claim-{i}", subject_id).model_copy(update={"workspace_id": workspace_id}))

    builder = ContextGraphBuilder(claim_repo)
    result = await builder.build(
        ContextGraphScope(workspace_id=workspace_id, subject_id=subject_id), max_nodes=2, now=_T0,
    )

    assert result.nodes_used == 2
    assert result.truncated is True
    assert len(result.claims) == 2


async def test_build_respects_predicate_diversity_cap(executor):
    workspace_id = f"ws-ctx-{uuid4().hex[:8]}"
    claim_repo = ClaimRepository(executor)
    subject_id = "contact-2"
    for i in range(4):
        await claim_repo.create_claim(
            _claim(f"claim-div-{i}", subject_id, predicate="RAISED_OBJECTION").model_copy(
                update={"workspace_id": workspace_id}
            )
        )

    builder = ContextGraphBuilder(claim_repo)
    result = await builder.build(
        ContextGraphScope(workspace_id=workspace_id, subject_id=subject_id),
        max_nodes=50, predicate_diversity_cap=2, now=_T0,
    )

    assert result.nodes_used == 2  # capped by diversity, not the (much larger) node budget


async def test_build_with_no_scope_returns_empty_result(executor):
    claim_repo = ClaimRepository(executor)
    builder = ContextGraphBuilder(claim_repo)
    result = await builder.build(ContextGraphScope(workspace_id="ws-empty"), now=_T0)
    assert result.claims == []
    assert result.nodes_used == 0
    assert result.truncated is False


async def test_selection_reason_and_evidence_are_populated(executor):
    workspace_id = f"ws-ctx-{uuid4().hex[:8]}"
    claim_repo = ClaimRepository(executor)
    await claim_repo.create_claim(_claim("claim-x", "contact-3").model_copy(update={"workspace_id": workspace_id}))

    builder = ContextGraphBuilder(claim_repo)
    result = await builder.build(ContextGraphScope(workspace_id=workspace_id, subject_id="contact-3"), now=_T0)

    assert len(result.selected_items) == 1
    assert "confidence=" in result.selected_items[0].reason
    assert result.evidence[0].claim_id == "claim-x"
