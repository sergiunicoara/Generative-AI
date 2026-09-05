"""Phase 7 (docs/evaluation.md's B5 item) — the reranker hook in
src/context_graph/builder.py, against live Neo4j. Uses a stub reranker
(monkeypatched into the builder module) rather than loading the real
cross-encoder model -- src/unit/context_graph/test_reranker.py covers the
real model separately, more slowly. This file proves the *wiring*: off by
default, on only when both reranker_enabled and scope.query_text are set,
reordering + rescoring correctly, additive to Claims (never dropping any).
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

import src.context_graph.builder as builder_module
from src.context_graph.builder import ContextGraphBuilder, ContextGraphScope
from src.core.config import get_settings
from src.domain.assertion import Claim
from src.domain.enums import AdjudicationStatus, ErasureStatus, Polarity, SpeakerRole
from src.graph.repositories.claim_repository import ClaimRepository

pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _claim(workspace_id: str, claim_id: str, subject_id: str, object_value: str) -> Claim:
    return Claim(
        claim_id=claim_id, workspace_id=workspace_id, subject_id=subject_id,
        predicate="HAS_BLOCKER", object_value=object_value, polarity=Polarity.AFFIRMED,
        source_type="transcript", source_record_id=None, source_segment_id=None,
        evidence_char_start=0, evidence_char_end=5, source_timestamp=_T0,
        speaker_id=None, speaker_role=SpeakerRole.BUYER, confidence=0.5,  # identical base score
        valid_from=_T0, valid_to=None, transaction_from=_T0, transaction_to=None,
        is_superseded=False, adjudication_status=AdjudicationStatus.UNREVIEWED,
        retention_class="standard", erasure_status=ErasureStatus.ACTIVE, created_at=_T0,
    )


@pytest.fixture(autouse=True)
def _reset_settings():
    yield
    get_settings.cache_clear()


async def test_reranker_disabled_by_default_leaves_original_order(executor, monkeypatch):
    workspace_id = f"ws-rerank-off-{uuid4().hex[:8]}"
    subject_id = "contact-1"
    claim_repo = ClaimRepository(executor)
    await claim_repo.create_claim(_claim(workspace_id, "claim-a", subject_id, "budget concern"))
    await claim_repo.create_claim(_claim(workspace_id, "claim-b", subject_id, "timeline concern"))

    async def _should_not_be_called(query_text, texts):
        raise AssertionError("rerank() must not be called when reranker_enabled is False")

    monkeypatch.setattr(builder_module, "rerank", _should_not_be_called)
    builder = ContextGraphBuilder(claim_repo)
    scope = ContextGraphScope(workspace_id=workspace_id, subject_id=subject_id, query_text="what about the timeline?")

    result = await builder.build(scope)  # reranker_enabled defaults to False
    assert len(result.claims) == 2


async def test_reranker_without_query_text_is_a_no_op_even_when_enabled(executor, monkeypatch):
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    get_settings.cache_clear()
    workspace_id = f"ws-rerank-noquery-{uuid4().hex[:8]}"
    subject_id = "contact-1"
    claim_repo = ClaimRepository(executor)
    await claim_repo.create_claim(_claim(workspace_id, "claim-a", subject_id, "budget concern"))

    async def _should_not_be_called(query_text, texts):
        raise AssertionError("rerank() must not be called with no query_text in scope")

    monkeypatch.setattr(builder_module, "rerank", _should_not_be_called)
    builder = ContextGraphBuilder(claim_repo)
    scope = ContextGraphScope(workspace_id=workspace_id, subject_id=subject_id)  # no query_text

    result = await builder.build(scope)
    assert len(result.claims) == 1


async def test_reranker_reorders_by_relevance_and_blends_the_score(executor, monkeypatch):
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    get_settings.cache_clear()
    workspace_id = f"ws-rerank-on-{uuid4().hex[:8]}"
    subject_id = "contact-1"
    claim_repo = ClaimRepository(executor)
    # Both claims share the identical confidence/recency/adjudication base
    # score (_claim helper above), so any order difference must come from
    # the reranker, not the pre-existing scoring formula.
    await claim_repo.create_claim(_claim(workspace_id, "claim-budget", subject_id, "budget concern"))
    await claim_repo.create_claim(_claim(workspace_id, "claim-timeline", subject_id, "timeline concern"))

    async def _stub_rerank(query_text, texts):
        # Score whichever text mentions "timeline" highest, regardless of
        # the original scored order -- proves the reranker's output
        # actually drives the final order, not just gets called.
        return [1.0 if "timeline" in t.lower() else 0.1 for t in texts]

    monkeypatch.setattr(builder_module, "rerank", _stub_rerank)
    builder = ContextGraphBuilder(claim_repo)
    scope = ContextGraphScope(workspace_id=workspace_id, subject_id=subject_id, query_text="what about the timeline?")

    result = await builder.build(scope)

    assert len(result.claims) == 2  # additive -- nothing dropped
    assert result.claims[0].claim_id == "claim-timeline"  # reranked to the top
    # The score is now a blend of relevance with the base
    # confidence/recency/adjudication/authority score, not the raw stub
    # value -- assert ordering and bounds instead of an exact number that
    # would break every time the blend weights get retuned.
    assert 0.0 <= result.selected_items[1].score < result.selected_items[0].score <= 1.0
    assert result.selected_items[0].score != 1.0  # no longer the raw stub logit
    assert "relevance=" in result.selected_items[0].reason
