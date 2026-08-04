"""§11 EXTRACTING/RESOLVING states — Gong-shaped transcript ingestion:
parse -> persist segments (unconditionally, before any extraction) -> resolve
speakers -> window -> extract -> persist Claims.

Design note on canonical_subject / Claim.subject_id: assertion_id (§6) is
computed from the opaque `speaker_label`, never a resolved contact/seller id.
If it used the resolved id, the same transcript re-extracted after CRM data
changes speaker resolution's outcome would get a *different* assertion_id,
breaking 'second identical ingest changes zero graph counts' (§15). Claim.
subject_id starts as the opaque speaker_label and is what §9 calls 'targeted
reconciliation' updates in place once a Mention naming that speaker resolves —
claim_id itself never changes. This is also why an unresolved ('opaque') speaker
still produces a Claim: subject_id degrades to the opaque label, never to a
dropped Claim (§15's 'opaque speaker IDs still produce Claims with appropriate
authority').
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.domain.assertion import Claim
from src.domain.enums import AdjudicationStatus, SpeakerRole
from src.domain.identity import assertion_id as _assertion_id
from src.extraction.provider import ExtractionInput, ExtractionProvider, WindowSegmentText
from src.extraction.windowing import build_windows
from src.graph.repositories.claim_repository import ClaimRepository
from src.graph.repositories.conversation_repository import ConversationRepository
from src.graph.repositories.source_repository import SourceRepository
from src.ingestion.reconciliation import ReconciliationOutcome, reconcile_deletion, reconcile_source_record
from src.resolution.speaker import resolve_speaker


@dataclass(frozen=True)
class TranscriptIngestionResult:
    conversation_id: str
    outcome: ReconciliationOutcome
    claims_created: int


class TranscriptIngestionPipeline:
    def __init__(
        self,
        conversation_repo: ConversationRepository,
        source_repo: SourceRepository,
        claim_repo: ClaimRepository,
        adapter,  # GongAdapter-shaped
        extraction_provider: ExtractionProvider,
    ):
        self._conversation_repo = conversation_repo
        self._source_repo = source_repo
        self._claim_repo = claim_repo
        self._adapter = adapter
        self._extraction_provider = extraction_provider

    async def ingest_call(
        self,
        workspace_id: str,
        raw_call: dict,
        *,
        ingestion_run_id: str,
        observed_at: datetime,
        opportunity_id: str | None = None,
        account_id: str | None = None,
        email_to_contact_id: dict[str, str] | None = None,
        email_to_seller_id: dict[str, str] | None = None,
        confidence_default: float = 0.75,
        retention_class: str = "standard",
        window_max_duration_ms: int = 90_000,
        window_max_tokens: int = 200,
        window_overlap_segments: int = 1,
    ) -> TranscriptIngestionResult:
        email_to_contact_id = email_to_contact_id or {}
        email_to_seller_id = email_to_seller_id or {}

        parsed_conv = self._adapter.parse_conversation(
            workspace_id, raw_call, opportunity_id=opportunity_id, account_id=account_id
        )
        reconciliation = await reconcile_source_record(
            self._source_repo,
            workspace_id=workspace_id,
            source_system=self._adapter.source_system,
            object_type=parsed_conv.object_type,
            external_id=parsed_conv.external_id,
            content_hash=parsed_conv.content_hash,
            ingestion_run_id=ingestion_run_id,
            observed_at=observed_at,
        )
        conversation_changed = reconciliation.outcome in (
            ReconciliationOutcome.CREATED, ReconciliationOutcome.SUPERSEDED
        )
        if conversation_changed:
            await self._conversation_repo.upsert_conversation(parsed_conv.entity)

        conversation_id_ = parsed_conv.entity.conversation_id

        if parsed_conv.extra and parsed_conv.extra.get("is_deleted"):
            deletion = await reconcile_deletion(
                self._source_repo,
                workspace_id=workspace_id,
                source_system=self._adapter.source_system,
                object_type=parsed_conv.object_type,
                external_id=parsed_conv.external_id,
                observed_at=observed_at,
                adapter_supports_deletion_signal=self._adapter.supports_deletion_signal,
            )
            return TranscriptIngestionResult(conversation_id_, deletion.outcome, claims_created=0)

        # §7: every source segment is persisted even when extraction is
        # skipped — unconditional, not gated behind conversation_changed.
        segments = self._adapter.parse_segments(workspace_id, conversation_id_, raw_call)
        for seg in segments:
            await self._conversation_repo.upsert_segment(seg)

        participants = self._adapter.parse_participants(workspace_id, conversation_id_, raw_call)
        party_emails = self._adapter.parse_party_emails(raw_call)
        speaker_role_by_label: dict[str, SpeakerRole] = {}
        for participant in participants:
            resolution = resolve_speaker(
                workspace_id=workspace_id,
                conversation_id=conversation_id_,
                speaker_label=participant.speaker_label,
                raw_email=party_emails.get(participant.speaker_label),
                email_to_contact_id=email_to_contact_id,
                email_to_seller_id=email_to_seller_id,
            )
            resolved_participant = participant.model_copy(update={
                "contact_id": resolution.resolved_contact_id,
                "seller_id": resolution.resolved_seller_id,
                "role": resolution.role,
            })
            await self._conversation_repo.upsert_participant(resolved_participant)
            await self._conversation_repo.upsert_speaker_resolution(resolution)
            speaker_role_by_label[participant.speaker_label] = resolution.role

        if not conversation_changed:
            # identical re-ingest — segments/participants already reconciled
            # above (harmless no-op MERGE writes); skip re-windowing/re-
            # extraction entirely rather than re-running an LLM over unchanged
            # content.
            return TranscriptIngestionResult(conversation_id_, reconciliation.outcome, claims_created=0)

        windows = build_windows(
            segments, workspace_id=workspace_id, conversation_id=conversation_id_,
            max_duration_ms=window_max_duration_ms, max_tokens=window_max_tokens,
            overlap_segments=window_overlap_segments,
        )
        segments_by_id = {s.segment_id: s for s in segments}
        inputs = [
            ExtractionInput(
                window=window,
                segments=[
                    WindowSegmentText(
                        segment_id=sid,
                        speaker_label=segments_by_id[sid].speaker_label,
                        text=segments_by_id[sid].text,
                    )
                    for sid in window.segment_ids
                ],
            )
            for window in windows
        ]

        results = await self._extraction_provider.extract(inputs)

        claims_created = 0
        for result in results:
            for assertion in result.assertions:
                segment = segments_by_id[assertion.segment_id]
                speaker_role = speaker_role_by_label.get(segment.speaker_label, SpeakerRole.UNKNOWN)
                claim_id = _assertion_id(
                    workspace_id,
                    segment.segment_id,
                    assertion.evidence_char_start,
                    assertion.evidence_char_end,
                    segment.speaker_label,  # canonical_subject — see module docstring
                    assertion.predicate,
                    assertion.object_text,
                    assertion.polarity.value,
                )
                claim = Claim(
                    claim_id=claim_id,
                    workspace_id=workspace_id,
                    subject_id=segment.speaker_label,
                    predicate=assertion.predicate,
                    object_value=assertion.object_text,
                    polarity=assertion.polarity,
                    source_type="transcript",
                    source_record_id=parsed_conv.entity.source_record_id,
                    source_segment_id=segment.segment_id,
                    evidence_char_start=assertion.evidence_char_start,
                    evidence_char_end=assertion.evidence_char_end,
                    source_timestamp=parsed_conv.entity.occurred_at,
                    speaker_id=segment.speaker_label,
                    speaker_role=speaker_role,
                    confidence=confidence_default,
                    valid_from=observed_at,
                    transaction_from=observed_at,
                    adjudication_status=AdjudicationStatus.UNREVIEWED,
                    retention_class=retention_class,
                    created_at=observed_at,
                )
                await self._claim_repo.create_claim(claim)
                claims_created += 1

        return TranscriptIngestionResult(conversation_id_, reconciliation.outcome, claims_created)
