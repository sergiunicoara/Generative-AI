"""§16 P3 exit criterion — polarity, overlap, opaque speakers, invalid output,
and prompt injection tests pass. This file covers the end-to-end, graph-backed
parts (overlap dedup, opaque speakers, evidence-span mapping, idempotent
re-ingest); the pure-logic parts (polarity, invalid-output retry, prompt
injection) are covered in tests/unit/extraction/ and tests/security/.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.extraction.fixture_provider import FixtureExtractionProvider
from src.domain.enums import SpeakerRole
from src.graph.repositories.claim_repository import ClaimRepository
from src.graph.repositories.conversation_repository import ConversationRepository
from src.graph.repositories.source_repository import SourceRepository
from src.ingestion.adapters.gong import GongAdapter
from src.ingestion.reconciliation import ReconciliationOutcome
from src.ingestion.transcript_pipeline import TranscriptIngestionPipeline

pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 6, 15, 14, 0, tzinfo=timezone.utc)


def _pipeline(executor) -> tuple[TranscriptIngestionPipeline, ConversationRepository, ClaimRepository]:
    conv_repo = ConversationRepository(executor)
    claim_repo = ClaimRepository(executor)
    pipeline = TranscriptIngestionPipeline(
        conv_repo, SourceRepository(executor), claim_repo, GongAdapter(), FixtureExtractionProvider()
    )
    return pipeline, conv_repo, claim_repo


def _raw_call(call_id: str) -> dict:
    return {
        "id": call_id,
        "started": "2026-06-15T14:00:00Z",
        "deleted": False,
        "parties": [
            {"speakerId": "spk_1", "name": "Elena Popescu", "emailAddress": "elena.popescu@acme.com"},
            {"speakerId": "spk_2", "name": "Sam Seller", "emailAddress": "sam@ourcompany.com"},
            {"speakerId": "spk_3"},  # opaque — no email at all, unresolvable
        ],
        "transcript": [
            {"speakerId": "spk_1", "sentences": [
                {"text": "We are concerned about pricing.", "start": 0, "end": 2000},
            ]},
            {"speakerId": "spk_1", "sentences": [
                {"text": "Pricing worries us a lot.", "start": 2000, "end": 4000},
            ]},
            {"speakerId": "spk_3", "sentences": [
                {"text": "Security is also a concern for us.", "start": 4000, "end": 6000},
            ]},
        ],
    }


async def test_segments_and_participants_are_persisted(executor):
    workspace_id = f"ws-transcript-{uuid4().hex[:8]}"
    pipeline, conv_repo, _ = _pipeline(executor)

    result = await pipeline.ingest_call(
        workspace_id, _raw_call("call-1"), ingestion_run_id="run-1", observed_at=_T0,
        email_to_contact_id={"elena.popescu@acme.com": "contact-elena"},
        email_to_seller_id={"sam@ourcompany.com": "seller-sam"},
    )

    assert result.outcome == ReconciliationOutcome.CREATED

    segments = await conv_repo.list_segments(workspace_id, result.conversation_id)
    assert len(segments) == 3
    assert [s.source_segment_index for s in segments] == [0, 1, 2]

    participants = await conv_repo.list_participants(workspace_id, result.conversation_id)
    by_label = {p.speaker_label: p for p in participants}
    assert by_label["spk_1"].role == SpeakerRole.BUYER
    assert by_label["spk_1"].contact_id == "contact-elena"
    assert by_label["spk_2"].role == SpeakerRole.SELLER
    assert by_label["spk_3"].role == SpeakerRole.UNKNOWN  # opaque, no email to match


async def test_opaque_speaker_still_produces_a_claim(executor):
    workspace_id = f"ws-transcript-{uuid4().hex[:8]}"
    pipeline, _, claim_repo = _pipeline(executor)

    await pipeline.ingest_call(
        workspace_id, _raw_call("call-2"), ingestion_run_id="run-1", observed_at=_T0,
        email_to_contact_id={"elena.popescu@acme.com": "contact-elena"},
        email_to_seller_id={"sam@ourcompany.com": "seller-sam"},
    )

    # spk_3 (opaque) raised a security concern in segment index 2
    claims = await claim_repo.list_claims_by_subject(workspace_id, "spk_3")
    assert len(claims) >= 1
    assert claims[0].speaker_role == SpeakerRole.UNKNOWN
    assert claims[0].speaker_id == "spk_3"  # subject stays the opaque label, never dropped


async def test_evidence_span_maps_to_the_exact_source_segment(executor):
    workspace_id = f"ws-transcript-{uuid4().hex[:8]}"
    pipeline, conv_repo, claim_repo = _pipeline(executor)

    result = await pipeline.ingest_call(
        workspace_id, _raw_call("call-3"), ingestion_run_id="run-1", observed_at=_T0,
    )

    claims = await claim_repo.list_claims_by_subject(workspace_id, "spk_1")
    assert claims
    for claim in claims:
        segment = await conv_repo.get_segment(workspace_id, claim.source_segment_id)
        assert segment is not None
        assert 0 <= claim.evidence_char_start < claim.evidence_char_end <= len(segment.text)
        # the evidence text itself is a real substring of the segment, not an
        # offset relative to some window's own (possibly overlapping) text.
        assert segment.text[claim.evidence_char_start:claim.evidence_char_end]


async def test_overlapping_windows_do_not_duplicate_claims(executor):
    workspace_id = f"ws-transcript-{uuid4().hex[:8]}"
    conv_repo = ConversationRepository(executor)
    claim_repo = ClaimRepository(executor)
    pipeline = TranscriptIngestionPipeline(
        conv_repo, SourceRepository(executor), claim_repo, GongAdapter(), FixtureExtractionProvider()
    )

    # Small token budget forces segment index 1 ("Pricing worries us a lot.")
    # into two overlapping windows (overlap_segments=1 default) — the fixture
    # extractor matches independently per window, so it fires on that segment
    # twice, producing the *same* assertion (same segment_id/char offsets/
    # predicate/object/polarity) both times.
    result = await pipeline.ingest_call(
        workspace_id, _raw_call("call-4"), ingestion_run_id="run-1", observed_at=_T0,
        window_max_tokens=6, window_overlap_segments=1,
    )

    claims = await claim_repo.list_claims_by_subject(workspace_id, "spk_1")
    # Regardless of how many overlapping windows re-extracted the same
    # sentence, claim_repository.create_claim's MERGE-on-claim_id (itself
    # content-deterministic via assertion_id) collapses them to one node per
    # distinct (segment, span, predicate, object, polarity).
    seen = [(c.source_segment_id, c.evidence_char_start, c.evidence_char_end, c.predicate) for c in claims]
    assert len(seen) == len(set(seen)), f"duplicate Claims found for overlapping-window extraction: {claims}"


async def test_identical_reingest_is_a_no_op_and_skips_reextraction(executor):
    workspace_id = f"ws-transcript-{uuid4().hex[:8]}"
    pipeline, _, claim_repo = _pipeline(executor)
    raw_call = _raw_call("call-5")

    first = await pipeline.ingest_call(workspace_id, raw_call, ingestion_run_id="run-1", observed_at=_T0)
    second = await pipeline.ingest_call(workspace_id, raw_call, ingestion_run_id="run-2", observed_at=_T0)

    assert first.outcome == ReconciliationOutcome.CREATED
    assert first.claims_created > 0
    assert second.outcome == ReconciliationOutcome.NO_OP
    assert second.claims_created == 0  # extraction was skipped entirely, not just deduped

    claims = await claim_repo.list_claims_by_subject(workspace_id, "spk_1")
    assert len(claims) == len({(c.source_segment_id, c.predicate, c.evidence_char_start) for c in claims})


async def test_deleted_call_is_tombstoned_without_extraction(executor):
    workspace_id = f"ws-transcript-{uuid4().hex[:8]}"
    pipeline, _, claim_repo = _pipeline(executor)
    raw_call = _raw_call("call-6")
    deleted_call = {**raw_call, "deleted": True}

    await pipeline.ingest_call(workspace_id, raw_call, ingestion_run_id="run-1", observed_at=_T0)
    result = await pipeline.ingest_call(workspace_id, deleted_call, ingestion_run_id="run-2", observed_at=_T0)

    assert result.outcome == ReconciliationOutcome.TOMBSTONED
    assert result.claims_created == 0
