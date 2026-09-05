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

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import structlog

from src.core.telemetry import RESOLUTION_CANDIDATES_CONSIDERED, RESOLUTION_DURATION_SECONDS
from src.domain.assertion import Claim, ExtractionRun
from src.domain.conversation import Mention
from src.domain.enums import AdjudicationStatus, ResolutionStatus, SpeakerRole
from src.domain.identity import assertion_id as _assertion_id
from src.domain.identity import extraction_run_id as _extraction_run_id
from src.domain.identity import mention_id as _mention_id
from src.extraction.provider import ExtractionInput, ExtractionProvider, WindowSegmentText
from src.extraction.windowing import build_windows
from src.graph.repositories.claim_repository import ClaimRepository
from src.graph.repositories.conversation_repository import ConversationRepository
from src.graph.repositories.extraction_run_repository import ExtractionRunRepository
from src.graph.repositories.review_repository import ReviewRepository
from src.graph.repositories.source_repository import SourceRepository
from src.graph.sales_ontology import validate_claim_predicate
from src.ingestion.reconciliation import (
    ReconciliationOutcome,
    reconcile_deletion,
    reconcile_source_record,
)
from src.resolution.candidates import CandidateGenerator
from src.resolution.pipeline import resolve_mention
from src.resolution.speaker import resolve_speaker


@dataclass(frozen=True)
class TranscriptIngestionResult:
    conversation_id: str
    outcome: ReconciliationOutcome
    claims_created: int
    # How many windows this call produced. 0 on the identical-re-ingest path,
    # which deliberately skips windowing entirely -- so a caller aggregating
    # progress across calls sees "nothing to do" rather than a phantom total.
    windows_total: int = 0


# Reports (windows_processed, windows_total) for one call. Async because every
# implementation this repo has persists the update (worker -> job store).
ProgressCallback = Callable[[int, int], Awaitable[None]]


@dataclass(frozen=True)
class _SpeakerSubject:
    """What one speaker_label resolved to, carried to every Claim it produces.

    Deliberately does NOT carry a replacement for `Claim.subject_id`. That
    field stays the opaque speaker_label, because it is a load-bearing join
    key elsewhere -- `src/usecases/buying_committee.py` matches
    `claim.subject_id == participant.speaker_label` to attribute claims to
    committee members, and this module's own header documents subject_id as
    the opaque label that review later rewrites in place. Resolution is
    therefore recorded in the four additive Claim fields below rather than by
    overwriting the subject.
    """

    entity_id: str | None = None
    entity_type: str | None = None
    status: ResolutionStatus | None = None
    score: float | None = None
    candidate_count: int = 0


log = structlog.get_logger(__name__)


class TranscriptIngestionPipeline:
    def __init__(
        self,
        conversation_repo: ConversationRepository,
        source_repo: SourceRepository,
        claim_repo: ClaimRepository,
        adapter,  # GongAdapter-shaped
        extraction_provider: ExtractionProvider,
        candidate_generator: CandidateGenerator | None = None,
        review_repo: ReviewRepository | None = None,
        extraction_run_repo: ExtractionRunRepository | None = None,
        provider_name: str = "unknown",
        model_name: str = "unknown",
        prompt_version: str = "v1",
        extractor_version: str = "v1",
    ):
        self._conversation_repo = conversation_repo
        self._source_repo = source_repo
        self._claim_repo = claim_repo
        self._adapter = adapter
        self._extraction_provider = extraction_provider
        # Both optional so every existing construction site keeps working and
        # entity resolution stays strictly additive: without a
        # candidate_generator the pipeline behaves exactly as it did before
        # (speaker resolution only, claims keyed on the opaque label). The
        # worker and the API route both pass one; unit tests that only care
        # about windowing or claim shape need not.
        self._candidate_generator = candidate_generator
        # Persisting the Mention is what makes a PENDING_REVIEW outcome
        # reachable by ReviewService.list_pending() -- without it, an ambiguous
        # speaker would be scored and then silently forgotten.
        self._review_repo = review_repo
        # P3.2 extraction provenance -- optional and additive, same story as
        # candidate_generator/review_repo above. ExtractionProvider is a bare
        # Protocol with no model/prompt_version attributes to introspect
        # (src/extraction/provider.py), so these are caller-supplied config,
        # the same shape as confidence_default below.
        self._extraction_run_repo = extraction_run_repo
        self._provider_name = provider_name
        self._model_name = model_name
        self._prompt_version = prompt_version
        self._extractor_version = extractor_version

    async def _resolve_speaker_identity(
        self,
        *,
        workspace_id: str,
        conversation_id: str,
        speaker_label: str,
        display_name: str | None,
        speaker_resolution,
        segments: list,
        decided_at: datetime,
        occurred_at: datetime,
        account_id: str | None,
        email: str | None,
    ) -> _SpeakerSubject:
        """Decide what entity a speaker refers to, for every Claim they produce.

        Three tiers, cheapest and most certain first:

        1. `resolve_speaker` already matched an exact email to a Contact or
           Seller. That is deterministic, so it is treated as AUTO_LINKED with
           score 1.0 and no candidate search is run -- fuzzy scoring could only
           weaken an exact identifier match.
        2. The transcript names the speaker and a CandidateGenerator is wired:
           build a Mention over the speaker's first segment and run the real
           `resolve_mention` (blocking -> scoring -> policy). The existing
           thresholds in src/resolution/policy.py decide AUTO_LINKED /
           PENDING_REVIEW / UNRESOLVED -- no new thresholds are introduced here.
        3. Anything else (opaque speaker, no generator wired): fail safe to the
           opaque label, exactly the pre-existing behaviour.
        """
        # Tier 1 -- deterministic email match already succeeded.
        if speaker_resolution.resolved_contact_id or speaker_resolution.resolved_seller_id:
            entity_id = speaker_resolution.resolved_contact_id or speaker_resolution.resolved_seller_id
            return _SpeakerSubject(
                entity_id=entity_id,
                entity_type="Contact" if speaker_resolution.resolved_contact_id else "Seller",
                status=ResolutionStatus.AUTO_LINKED,
                score=1.0,
            )

        # Tier 3 conditions -- nothing to resolve against.
        if self._candidate_generator is None or not display_name or not display_name.strip():
            return _SpeakerSubject()

        anchor = next((s for s in segments if s.speaker_label == speaker_label), None)
        if anchor is None:
            # A named party who never speaks has no evidence span to anchor a
            # Mention to, and Mention requires a well-formed one. Nothing to
            # attribute claims to either, since they produced no segments.
            return _SpeakerSubject()

        mention = Mention(
            mention_id=_mention_id(anchor.segment_id, 0, len(display_name), speaker_label, "PERSON"),
            workspace_id=workspace_id,
            segment_id=anchor.segment_id,
            # The speaker's name is metadata about who is talking, not a span
            # quoted inside the utterance, so there is no true offset to point
            # at. Anchoring at the start of their first segment keeps the span
            # well-formed and stable across re-ingest (segment_id is
            # content-derived), which is what mention_id needs.
            char_start=0,
            char_end=len(display_name),
            surface_text=display_name,
            # The join key, not a normalization of surface_text. ReviewService
            # reconciles a confirmed mention onto claims via
            # list_claims_by_subject(workspace_id, mention.normalized_surface),
            # and ingested claims are subject-keyed on the opaque speaker
            # label -- so this must be that label for a reviewer's decision to
            # reach them. Candidate lookup and scoring both read surface_text
            # (src/resolution/pipeline.py lines 93 and 117), so match quality
            # is unaffected; only the exact-name short-circuit and the
            # pre-truncation sort read normalized_surface, and both degrade
            # safely to the ordinary fuzzy path.
            normalized_surface=speaker_label,
            entity_type="PERSON",
            # The call's actual occurred-at time, not ingestion wall-clock
            # (`decided_at`/`observed_at` below) -- P3.1's whole point is a
            # source timestamp distinct from write time.
            source_timestamp=occurred_at,
        )

        # P4.6 -- a mention re-resolved after a human rejected every candidate
        # last time must not re-propose the same rejected set. Only the most
        # recent ReviewDecision matters: an older rejection superseded by a
        # later confirmation (or a later, different rejection) no longer
        # reflects what's actually excluded.
        excluded_entity_ids: frozenset[str] = frozenset()
        if self._review_repo is not None:
            history = await self._review_repo.list_review_decisions_for_mention(
                workspace_id, mention.mention_id
            )
            if history and history[0].rejected:
                excluded_entity_ids = frozenset(history[0].candidates_shown)

        started_at = time.monotonic()
        outcome = await resolve_mention(
            workspace_id=workspace_id,
            mention=mention,
            entity_type="Contact",  # PERSON -> Contact, same mapping the review route uses
            candidate_generator=self._candidate_generator,
            decided_at=decided_at,
            excluded_entity_ids=excluded_entity_ids,
        )
        status = outcome.decision.status
        RESOLUTION_DURATION_SECONDS.labels(status=status.value.lower()).observe(
            time.monotonic() - started_at
        )
        RESOLUTION_CANDIDATES_CONSIDERED.observe(len(outcome.candidates_shown))

        if self._review_repo is not None:
            # Persist both so a PENDING_REVIEW speaker is reachable by
            # ReviewService.list_pending(), and so the reviewer sees the same
            # scores the pipeline decided on.
            await self._review_repo.upsert_mention(outcome.mention)
            await self._review_repo.upsert_resolution_decision(outcome.decision)

        log.info(
            "ingestion.speaker_resolution",
            conversation_id=conversation_id,
            # Identifiers and scores only -- never surface_text, segment text,
            # or the email that drove tier 1. The speaker label is an opaque
            # source id by construction (§8), so it is safe to log.
            speaker_label=speaker_label,
            mention_id=mention.mention_id,
            resolution_status=status.value,
            resolution_score=outcome.decision.final_score,
            candidates=len(outcome.candidates_shown),
            account_id=account_id,
        )

        return _SpeakerSubject(
            entity_id=outcome.decision.resolved_entity_id,
            entity_type="Contact" if outcome.decision.resolved_entity_id else None,
            status=status,
            score=outcome.decision.final_score,
            candidate_count=len(outcome.candidates_shown),
        )

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
        on_progress: ProgressCallback | None = None,
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
        party_names = (
            self._adapter.parse_party_names(raw_call)
            if hasattr(self._adapter, "parse_party_names") else {}
        )
        speaker_role_by_label: dict[str, SpeakerRole] = {}
        subject_by_label: dict[str, _SpeakerSubject] = {}
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
            subject_by_label[participant.speaker_label] = await self._resolve_speaker_identity(
                workspace_id=workspace_id,
                conversation_id=conversation_id_,
                speaker_label=participant.speaker_label,
                display_name=party_names.get(participant.speaker_label),
                speaker_resolution=resolution,
                segments=segments,
                decided_at=observed_at,
                occurred_at=parsed_conv.entity.occurred_at,
                account_id=account_id,
                email=party_emails.get(participant.speaker_label),
            )

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

        # Report the denominator before any extraction starts, so a poller
        # that catches the job early sees 0/N rather than 0/0 (which the API
        # contract defines as "this kind has no windows").
        if on_progress is not None:
            await on_progress(0, len(inputs))

        # P3.2 -- one ExtractionRun per actual extraction call, persisted
        # before the call starts so a run that never completes (crash,
        # timeout) is still visible as "started" rather than untraceable.
        # Unwired (extraction_run_repo=None) stays additive like
        # candidate_generator/review_repo above: no run is constructed at all,
        # so Claim.extraction_run_id keeps its default rather than pointing at
        # an id that was never actually persisted anywhere.
        extraction_run: ExtractionRun | None = None
        if self._extraction_run_repo is not None:
            run_nonce = uuid4().hex
            extraction_run = ExtractionRun(
                extraction_run_id=_extraction_run_id(
                    self._provider_name, self._model_name, self._prompt_version,
                    self._extractor_version, run_nonce,
                ),
                workspace_id=workspace_id,
                provider=self._provider_name,
                model=self._model_name,
                prompt_version=self._prompt_version,
                extractor_version=self._extractor_version,
                run_nonce=run_nonce,
                # Real wall-clock time the call actually started, not
                # observed_at (a caller-supplied ingestion timestamp that's
                # identical for started_at and completed_at) -- using
                # observed_at for both left every run showing zero duration
                # regardless of how long extraction actually took.
                started_at=datetime.now(timezone.utc),
            )
            await self._extraction_run_repo.create_extraction_run(extraction_run)

        results = await self._extraction_provider.extract(inputs)

        if self._extraction_run_repo is not None and extraction_run is not None:
            await self._extraction_run_repo.complete_extraction_run(
                workspace_id, extraction_run.extraction_run_id, completed_at=datetime.now(timezone.utc)
            )

        # The provider owns fan-out (and its own bounded concurrency), so the
        # honest report is one update after it returns rather than a
        # fabricated per-window trickle this layer cannot actually observe.
        if on_progress is not None:
            await on_progress(len(results), len(inputs))

        claims_created = 0
        for result in results:
            for assertion in result.assertions:
                predicate = validate_claim_predicate(assertion.predicate)
                segment = segments_by_id[assertion.segment_id]
                speaker_role = speaker_role_by_label.get(segment.speaker_label, SpeakerRole.UNKNOWN)
                claim_id = _assertion_id(
                    workspace_id,
                    segment.segment_id,
                    assertion.evidence_char_start,
                    assertion.evidence_char_end,
                    segment.speaker_label,  # canonical_subject — see module docstring
                    predicate,
                    assertion.object_text,
                    assertion.polarity.value,
                )
                # Resolved identity for this segment's speaker. Defaults to the
                # opaque label so a speaker the pipeline never saw (or a
                # pipeline wired without a CandidateGenerator) behaves exactly
                # as before.
                subject = subject_by_label.get(segment.speaker_label, _SpeakerSubject())
                claim = Claim(
                    claim_id=claim_id,
                    workspace_id=workspace_id,
                    subject_id=segment.speaker_label,
                    predicate=predicate,
                    object_value=assertion.object_text,
                    polarity=assertion.polarity,
                    source_type="transcript",
                    source_record_id=parsed_conv.entity.source_record_id,
                    source_system=parsed_conv.entity.source_system,
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
                    extraction_run_id=extraction_run.extraction_run_id if extraction_run else None,
                    resolved_entity_id=subject.entity_id,
                    resolved_entity_type=subject.entity_type,
                    resolution_status=subject.status,
                    resolution_score=subject.score,
                    # speaker_id below is unchanged and still the opaque label:
                    # provenance survives resolution rather than being replaced
                    # by it, so the original transcript attribution is always
                    # recoverable even after a reviewer rewrites subject_id.
                )
                await self._claim_repo.create_claim(claim)
                claims_created += 1

        return TranscriptIngestionResult(
            conversation_id_, reconciliation.outcome, claims_created, windows_total=len(inputs)
        )
