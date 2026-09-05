"""Tenant-safe repository for Claim — same pattern as crm_repository.py.

Per docs/plan.md §10 ("no direct Claim or Mention fan-out from Account; route
evidence through Conversation, TranscriptSegment, and Opportunity"), this
repository never accepts an Account id — only subject_id (whatever entity the
Claim is actually about) and workspace_id.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from src.core.telemetry import CLAIMS_TOTAL
from src.domain.assertion import Claim
from src.domain.enums import AdjudicationStatus, ErasureStatus
from src.graph.execution import GraphExecutor, scoped_match

_CLAIM_RETURN = (
    "cl.claim_id AS claim_id, cl.workspace_id AS workspace_id, cl.subject_id AS subject_id, "
    "cl.predicate AS predicate, cl.object_id AS object_id, cl.object_value AS object_value, "
    "cl.polarity AS polarity, cl.source_type AS source_type, "
    "cl.source_record_id AS source_record_id, cl.source_segment_id AS source_segment_id, "
    "cl.evidence_char_start AS evidence_char_start, cl.evidence_char_end AS evidence_char_end, "
    "cl.source_timestamp AS source_timestamp, cl.speaker_id AS speaker_id, "
    "cl.speaker_role AS speaker_role, cl.confidence AS confidence, "
    "cl.valid_from AS valid_from, cl.valid_to AS valid_to, "
    "cl.transaction_from AS transaction_from, cl.transaction_to AS transaction_to, "
    "cl.is_superseded AS is_superseded, cl.adjudication_status AS adjudication_status, "
    "cl.adjudication_reason AS adjudication_reason, "
    "cl.adjudication_decided_by AS adjudication_decided_by, "
    "cl.adjudication_decided_at AS adjudication_decided_at, "
    "cl.source_system AS source_system, "
    "cl.retention_class AS retention_class, cl.erasure_status AS erasure_status, "
    "cl.created_at AS created_at, "
    # Resolved subject identity (§8/§9). Nodes written before these existed
    # return null for all four, which is exactly the Claim model's default --
    # so hydration of pre-existing claims keeps working untouched.
    "cl.resolved_entity_id AS resolved_entity_id, "
    "cl.resolved_entity_type AS resolved_entity_type, "
    "cl.resolution_status AS resolution_status, "
    "cl.resolution_score AS resolution_score, "
    # Extraction provenance (P3.2). Same null-safe story as the resolution
    # fields above: nodes written before this existed return null, which is
    # exactly Claim's default.
    "cl.extraction_run_id AS extraction_run_id"
)


def _claim_return(alias: str) -> str:
    """Return the Claim projection for either a current Claim or a revision."""
    return _CLAIM_RETURN.replace("cl.", f"{alias}.")


def _claim_params(claim: Claim) -> dict:
    return {
        "claim_id": claim.claim_id,
        "workspace_id": claim.workspace_id,
        "subject_id": claim.subject_id,
        "predicate": claim.predicate,
        "object_id": claim.object_id,
        "object_value": claim.object_value,
        "polarity": claim.polarity.value,
        "source_type": claim.source_type,
        "source_record_id": claim.source_record_id,
        "source_segment_id": claim.source_segment_id,
        "evidence_char_start": claim.evidence_char_start,
        "evidence_char_end": claim.evidence_char_end,
        "source_timestamp": claim.source_timestamp.isoformat(),
        "speaker_id": claim.speaker_id,
        "speaker_role": claim.speaker_role.value,
        "confidence": claim.confidence,
        "valid_from": claim.valid_from.isoformat(),
        "valid_to": claim.valid_to.isoformat() if claim.valid_to else None,
        "transaction_from": claim.transaction_from.isoformat(),
        "transaction_to": claim.transaction_to.isoformat() if claim.transaction_to else None,
        "is_superseded": claim.is_superseded,
        "adjudication_status": claim.adjudication_status.value,
        "adjudication_reason": claim.adjudication_reason,
        "adjudication_decided_by": claim.adjudication_decided_by,
        "adjudication_decided_at": (
            claim.adjudication_decided_at.isoformat() if claim.adjudication_decided_at else None
        ),
        "source_system": claim.source_system,
        "retention_class": claim.retention_class,
        "erasure_status": claim.erasure_status.value,
        "created_at": claim.created_at.isoformat(),
        "resolved_entity_id": claim.resolved_entity_id,
        "resolved_entity_type": claim.resolved_entity_type,
        "resolution_status": claim.resolution_status.value if claim.resolution_status else None,
        "resolution_score": claim.resolution_score,
        "extraction_run_id": claim.extraction_run_id,
    }


class ClaimRepository:
    def __init__(self, executor: GraphExecutor | None = None):
        self._executor = executor or GraphExecutor()

    async def create_claim(self, claim: Claim) -> None:
        """MERGE on claim_id — claim_id is itself content-derived (assertion_id),
        so re-persisting the identical Claim is a no-op write, not a duplicate.

        When source_segment_id is set, also materializes (seg)-[:HAS_CLAIM]->
        (cl) — §10's routing principle ('route evidence through Conversation,
        TranscriptSegment, and Opportunity') requires evidence to be reachable
        FROM the segment, not just carry the segment id as a bare property; the
        P4.5 recommendation use case traverses this edge to scope Claims to one
        conversation.
        """
        match = scoped_match("Claim", "cl", claim_id="claim_id")
        params = _claim_params(claim)
        if claim.source_segment_id:
            segment_match = scoped_match("TranscriptSegment", "seg", segment_id="source_segment_id")
            await self._executor.tenant_query(
                f"""
                MATCH {segment_match}
                MERGE {match}
                ON CREATE SET cl.created_at = $created_at,
                              cl.adjudication_status = $adjudication_status,
                              cl.adjudication_reason = $adjudication_reason,
                              cl.adjudication_decided_by = $adjudication_decided_by,
                              cl.adjudication_decided_at = $adjudication_decided_at
                SET cl.subject_id = $subject_id,
                    cl.predicate = $predicate,
                    cl.object_id = $object_id,
                    cl.object_value = $object_value,
                    cl.polarity = $polarity,
                    cl.source_type = $source_type,
                    cl.source_record_id = $source_record_id,
                    cl.source_segment_id = $source_segment_id,
                    cl.source_system = $source_system,
                    cl.evidence_char_start = $evidence_char_start,
                    cl.evidence_char_end = $evidence_char_end,
                    cl.source_timestamp = $source_timestamp,
                    cl.speaker_id = $speaker_id,
                    cl.speaker_role = $speaker_role,
                    cl.confidence = $confidence,
                    cl.valid_from = $valid_from,
                    cl.valid_to = $valid_to,
                    cl.transaction_from = $transaction_from,
                    cl.transaction_to = $transaction_to,
                    cl.is_superseded = $is_superseded,
                    cl.retention_class = $retention_class,
                    cl.erasure_status = $erasure_status,
                    cl.resolved_entity_id = $resolved_entity_id,
                    cl.resolved_entity_type = $resolved_entity_type,
                    cl.resolution_status = $resolution_status,
                    cl.resolution_score = $resolution_score,
                    cl.extraction_run_id = $extraction_run_id
                MERGE (seg)-[:HAS_CLAIM]->(cl)
                """,
                **params,
            )
            # "Claims created" (docs/plan.md Sec 14). MERGE makes this call
            # idempotent (docstring above), and GraphExecutor.tenant_query
            # doesn't surface Neo4j's created-vs-matched result-summary
            # counters, so this counts write attempts, not strictly
            # distinct new nodes -- an honest approximation, not exact.
            CLAIMS_TOTAL.labels(event="created").inc()
            return
        await self._executor.tenant_query(
            f"""
            MERGE {match}
            ON CREATE SET cl.created_at = $created_at,
                          cl.adjudication_status = $adjudication_status,
                          cl.adjudication_reason = $adjudication_reason,
                          cl.adjudication_decided_by = $adjudication_decided_by,
                          cl.adjudication_decided_at = $adjudication_decided_at
            SET cl.subject_id = $subject_id,
                cl.predicate = $predicate,
                cl.object_id = $object_id,
                cl.object_value = $object_value,
                cl.polarity = $polarity,
                cl.source_type = $source_type,
                cl.source_record_id = $source_record_id,
                cl.source_segment_id = $source_segment_id,
                cl.source_system = $source_system,
                cl.evidence_char_start = $evidence_char_start,
                cl.evidence_char_end = $evidence_char_end,
                cl.source_timestamp = $source_timestamp,
                cl.speaker_id = $speaker_id,
                cl.speaker_role = $speaker_role,
                cl.confidence = $confidence,
                cl.valid_from = $valid_from,
                cl.valid_to = $valid_to,
                cl.transaction_from = $transaction_from,
                cl.transaction_to = $transaction_to,
                cl.is_superseded = $is_superseded,
                cl.retention_class = $retention_class,
                cl.erasure_status = $erasure_status,
                cl.resolved_entity_id = $resolved_entity_id,
                cl.resolved_entity_type = $resolved_entity_type,
                cl.resolution_status = $resolution_status,
                cl.resolution_score = $resolution_score,
                cl.extraction_run_id = $extraction_run_id
            """,
            **params,
        )
        CLAIMS_TOTAL.labels(event="created").inc()  # see note above

    async def get_claim(self, workspace_id: str, claim_id: str) -> Claim | None:
        match = scoped_match("Claim", "cl", claim_id="claim_id")
        rows = await self._executor.tenant_query(
            f"MATCH {match} RETURN {_CLAIM_RETURN}",
            workspace_id=workspace_id,
            claim_id=claim_id,
        )
        return Claim(**rows[0]) if rows else None

    async def list_claims_by_subject(
        self, workspace_id: str, subject_id: str, *, limit: int = 1000, offset: int = 0
    ) -> list[Claim]:
        # 1000, not the repository-package norm of 100 -- Claims are
        # evidence (docs/ontology.md), and this feeds ContextGraphBuilder
        # directly; silently truncating evidence is a correctness bug, not
        # only a performance one (same reasoning as
        # conversation_repository.py::list_segments).
        match = scoped_match("Claim", "cl", subject_id="subject_id")
        rows = await self._executor.tenant_query(
            f"MATCH {match} RETURN {_CLAIM_RETURN} ORDER BY cl.claim_id SKIP $offset LIMIT $limit",
            workspace_id=workspace_id,
            subject_id=subject_id,
            offset=offset,
            limit=limit,
        )
        return [Claim(**row) for row in rows]

    async def list_claims_recorded_since(
        self, workspace_id: str, subject_id: str, since: datetime, *, limit: int = 1000, offset: int = 0
    ) -> list[Claim]:
        """Increment 14 — 'what's new on this subject since <date>', filtering
        on transaction_from (populated at ingest, real — see
        src/ingestion/transcript_pipeline.py) rather than attempting true
        point-in-time ('as of') reconstruction. valid_to/transaction_to are
        never set by anything in this vertical slice (no supersession-closes-
        the-interval wiring exists), so a genuine 'what did we believe before
        X' query would silently return wrong answers for every Claim that
        predates that wiring — deliberately not built; see
        docs/evaluation.md's Known measurement gaps.
        """
        match = scoped_match("Claim", "cl", subject_id="subject_id")
        rows = await self._executor.tenant_query(
            f"""
            MATCH {match}
            WHERE cl.transaction_from >= $since
            RETURN {_CLAIM_RETURN}
            ORDER BY cl.claim_id SKIP $offset LIMIT $limit
            """,
            workspace_id=workspace_id,
            subject_id=subject_id,
            since=since.isoformat(),
            offset=offset,
            limit=limit,
        )
        return [Claim(**row) for row in rows]

    async def list_claims_for_conversation(
        self, workspace_id: str, conversation_id: str, *, limit: int = 1000, offset: int = 0
    ) -> list[Claim]:
        """Routed through Conversation -[:HAS_SEGMENT]-> TranscriptSegment
        -[:HAS_CLAIM]-> Claim (§10) — never a bare property-match on
        source_segment_id, which would bypass the workspace-scoped traversal
        chain."""
        conv_match = scoped_match("Conversation", "c", conversation_id="conversation_id")
        rows = await self._executor.tenant_query(
            f"""
            MATCH {conv_match}
            MATCH (c)-[:HAS_SEGMENT]->(:TranscriptSegment)-[:HAS_CLAIM]->(cl:Claim)
            RETURN {_CLAIM_RETURN}
            ORDER BY cl.claim_id SKIP $offset LIMIT $limit
            """,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            offset=offset,
            limit=limit,
        )
        return [Claim(**row) for row in rows]

    async def list_claims_for_conversations(
        self, workspace_id: str, conversation_ids: list[str]
    ) -> dict[str, list[Claim]]:
        """Batched sibling of list_claims_for_conversation (Phase 3,
        docs/evaluation.md's "buying committee — three levels deep" N+1:
        BuyingCommitteeUseCase._gather_evidence previously called this once
        per participant). One round trip for every conversation_id in the
        given (already-bounded) list, grouped by conversation_id in Python.
        No per-conversation limit -- same reasoning as
        ConversationRepository.list_participants_for_conversations."""
        if not conversation_ids:
            return {}
        rows = await self._executor.tenant_query(
            """
            MATCH (c:Conversation {workspace_id: $workspace_id})
            WHERE c.conversation_id IN $conversation_ids
            MATCH (c)-[:HAS_SEGMENT]->(:TranscriptSegment)-[:HAS_CLAIM]->(cl:Claim)
            RETURN c.conversation_id AS _batch_conversation_id, """ + _CLAIM_RETURN,
            workspace_id=workspace_id,
            conversation_ids=conversation_ids,
        )
        grouped: dict[str, list[Claim]] = {cid: [] for cid in conversation_ids}
        for row in rows:
            conversation_id = row.pop("_batch_conversation_id")
            grouped[conversation_id].append(Claim(**row))
        return grouped

    async def list_claims_by_predicate_for_seller(
        self, workspace_id: str, seller_id: str, predicate: str, *, limit: int = 1000, offset: int = 0
    ) -> list[Claim]:
        """Increment 13 — cross-Opportunity aggregation for one seller's OPEN
        deals (e.g. 'top objections in this seller's pipeline'). Routes
        Opportunity(seller_id) -> Conversation(opportunity_id) ->
        TranscriptSegment -> Claim — the same routing shape
        list_claims_by_opportunity already relies on (Conversation carries
        opportunity_id as a direct property), just aggregated over every open
        Opportunity one seller owns instead of one at a time. Never a direct
        Account/Opportunity -> Claim fan-out (§10).
        """
        opp_match = scoped_match("Opportunity", "o", seller_id="seller_id")
        rows = await self._executor.tenant_query(
            f"""
            MATCH {opp_match}
            WHERE o.is_open = true
            MATCH (c:Conversation {{workspace_id: $workspace_id, opportunity_id: o.opportunity_id}})
            MATCH (c)-[:HAS_SEGMENT]->(:TranscriptSegment)-[:HAS_CLAIM]->(cl:Claim {{predicate: $predicate}})
            RETURN {_CLAIM_RETURN}
            ORDER BY cl.claim_id SKIP $offset LIMIT $limit
            """,
            workspace_id=workspace_id,
            seller_id=seller_id,
            predicate=predicate,
            offset=offset,
            limit=limit,
        )
        return [Claim(**row) for row in rows]

    async def list_claims_by_opportunity(
        self, workspace_id: str, opportunity_id: str, *, limit: int = 1000, offset: int = 0
    ) -> list[Claim]:
        """Same routing as list_claims_by_opportunity_and_predicate, without
        the predicate filter — every Claim across every Conversation
        belonging to one Opportunity. Backs Increment 11's conflict-detection
        route, which needs to compare Claims across all predicates, not one."""
        conv_match = scoped_match("Conversation", "c", opportunity_id="opportunity_id")
        rows = await self._executor.tenant_query(
            f"""
            MATCH {conv_match}
            MATCH (c)-[:HAS_SEGMENT]->(:TranscriptSegment)-[:HAS_CLAIM]->(cl:Claim)
            RETURN {_CLAIM_RETURN}
            ORDER BY cl.claim_id SKIP $offset LIMIT $limit
            """,
            workspace_id=workspace_id,
            opportunity_id=opportunity_id,
            offset=offset,
            limit=limit,
        )
        return [Claim(**row) for row in rows]

    async def list_claims_by_opportunity_and_predicate(
        self, workspace_id: str, opportunity_id: str, predicate: str, *, limit: int = 1000, offset: int = 0
    ) -> list[Claim]:
        """Every Claim with the given predicate across every Conversation
        belonging to one Opportunity — Conversation carries opportunity_id as
        a direct property (see conversation_repository.py's
        list_conversations_by_opportunity), so this routes through Conversation
        -[:HAS_SEGMENT]->TranscriptSegment-[:HAS_CLAIM]->Claim (§10) exactly
        like list_claims_for_conversation, just scoped by opportunity_id
        instead of conversation_id and additionally filtered by predicate.
        Backs the Q&A layer's account_objections/open_commitments intents and
        Increment 13's cross-deal aggregation (called once per Opportunity).
        """
        conv_match = scoped_match("Conversation", "c", opportunity_id="opportunity_id")
        rows = await self._executor.tenant_query(
            f"""
            MATCH {conv_match}
            MATCH (c)-[:HAS_SEGMENT]->(:TranscriptSegment)-[:HAS_CLAIM]->(cl:Claim {{predicate: $predicate}})
            RETURN {_CLAIM_RETURN}
            ORDER BY cl.claim_id SKIP $offset LIMIT $limit
            """,
            workspace_id=workspace_id,
            opportunity_id=opportunity_id,
            predicate=predicate,
            offset=offset,
            limit=limit,
        )
        return [Claim(**row) for row in rows]

    async def close_claim_interval(
        self, workspace_id: str, claim_id: str, *, valid_to: datetime, transaction_to: datetime
    ) -> None:
        """Increment 19 — marks a Claim superseded and closes both bitemporal
        intervals. A narrow SET on the existing node (mirroring
        conflict_repository.py's resolve_conflict style) rather than a full
        re-persist through create_claim — this only ever runs as the losing
        side of ConflictsUseCase.resolve(), and touching only the three fields
        that actually change keeps the write's blast radius obvious.

        Before this method, nothing in the codebase ever set valid_to/
        transaction_to/is_superseded (see docs/evaluation.md's prior "Known
        measurement gaps" entry) — this is the first and only writer.
        """
        match = scoped_match("Claim", "cl", claim_id="claim_id")
        await self._executor.tenant_query(
            f"""
            MATCH {match}
            SET cl.valid_to = $valid_to,
                cl.transaction_to = $transaction_to,
                cl.is_superseded = true
            """,
            workspace_id=workspace_id,
            claim_id=claim_id,
            valid_to=valid_to.isoformat(),
            transaction_to=transaction_to.isoformat(),
        )
        CLAIMS_TOTAL.labels(event="superseded").inc()

    async def transition_adjudication_status(
        self,
        workspace_id: str,
        claim_id: str,
        new_status: AdjudicationStatus,
        *,
        reason: str | None,
        decided_by: str | None,
        decided_at: datetime,
    ) -> bool:
        """Record a human adjudication decision as history, not an overwrite.

        Reuses reconcile_claim_subject's snapshot mechanism (ClaimRevision +
        HAS_REVISION) rather than introducing a new IS_STATUS-edge-to-status-
        node scheme -- this codebase already has one graph-native audit-trail
        convention, and a second, inconsistent one would cost more than it's
        worth.

        Deliberately does NOT touch transaction_from/transaction_to/
        is_superseded the way reconcile_claim_subject and close_claim_interval
        do: those three fields carry the bitemporal "is this assertion still
        the live belief" story, which is orthogonal to "has a human judged
        whether to trust this claim." Reusing them here would let an
        adjudication decision accidentally un-supersede a conflict-losing
        Claim, or invert an already-closed interval.

        The revision's own transaction_to is pinned equal to its
        transaction_from (a degenerate, always-empty interval) specifically
        so list_claims_as_of's open-interval union query never picks up an
        adjudication-history revision as a duplicate of the live Claim --
        these revisions are audit trail only, read back via
        list_adjudication_history, not list_claims_as_of.
        """
        transition_id = hashlib.sha256(
            f"{workspace_id}\x1f{claim_id}\x1f{decided_at.isoformat()}\x1f{new_status.value}".encode("utf-8")
        ).hexdigest()
        match = scoped_match("Claim", "cl", claim_id="claim_id")
        rows = await self._executor.tenant_query(
            f"""
            MATCH {match}
            MERGE (revision:ClaimRevision {{workspace_id: $workspace_id, revision_id: $revision_id}})
            ON CREATE SET revision = properties(cl),
                          revision.revision_id = $revision_id,
                          revision.revised_claim_id = cl.claim_id,
                          revision.adjudication_transition_id = $revision_id,
                          revision.adjudication_status_before = cl.adjudication_status,
                          revision.adjudication_status_after = $new_status,
                          revision.adjudication_reason = $reason,
                          revision.adjudication_decided_by = $decided_by,
                          revision.adjudication_decided_at = $decided_at,
                          revision.transaction_to = revision.transaction_from
            MERGE (cl)-[:HAS_REVISION]->(revision)
            SET cl.adjudication_status = $new_status,
                cl.adjudication_reason = $reason,
                cl.adjudication_decided_by = $decided_by,
                cl.adjudication_decided_at = $decided_at
            RETURN cl.claim_id AS claim_id
            """,
            workspace_id=workspace_id,
            claim_id=claim_id,
            new_status=new_status.value,
            reason=reason,
            decided_by=decided_by,
            decided_at=decided_at.isoformat(),
            revision_id=transition_id,
        )
        if rows:
            CLAIMS_TOTAL.labels(event="adjudicated").inc()
        return bool(rows)

    async def list_adjudication_history(
        self, workspace_id: str, claim_id: str, *, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        """Read only the adjudication-specific revision properties written by
        transition_adjudication_status, oldest first. These revisions carry a
        degenerate transaction interval (see above) and must never be
        confused with reconcile_claim_subject's reconstructable historical
        Claims sharing the same ClaimRevision label."""
        match = scoped_match("Claim", "cl", claim_id="claim_id")
        rows = await self._executor.tenant_query(
            f"""
            MATCH {match}
            MATCH (cl)-[:HAS_REVISION]->(revision:ClaimRevision)
            WHERE revision.adjudication_transition_id IS NOT NULL
            RETURN revision.adjudication_transition_id AS transition_id,
                   revision.adjudication_status_before AS status_before,
                   revision.adjudication_status_after AS status_after,
                   revision.adjudication_reason AS reason,
                   revision.adjudication_decided_by AS decided_by,
                   revision.adjudication_decided_at AS decided_at
            ORDER BY revision.adjudication_decided_at ASC
            SKIP $offset LIMIT $limit
            """,
            workspace_id=workspace_id,
            claim_id=claim_id,
            offset=offset,
            limit=limit,
        )
        return rows

    async def reconcile_claim_subject(
        self,
        workspace_id: str,
        claim_id: str,
        *,
        subject_id: str,
        decided_at: datetime,
        review_decision_id: str,
    ) -> bool:
        """Apply a human identity correction without erasing Claim history."""
        revision_id = hashlib.sha256(
            f"{workspace_id}\x1f{claim_id}\x1f{review_decision_id}\x1f{subject_id}".encode("utf-8")
        ).hexdigest()
        match = scoped_match("Claim", "cl", claim_id="claim_id")
        rows = await self._executor.tenant_query(
            f"""
            MATCH {match}
            MERGE (revision:ClaimRevision {{workspace_id: $workspace_id, revision_id: $revision_id}})
            ON CREATE SET revision = properties(cl),
                          revision.revision_id = $revision_id,
                          revision.revised_claim_id = cl.claim_id,
                          revision.review_decision_id = $review_decision_id,
                          revision.transaction_to = $decided_at,
                          revision.is_superseded = true
            MERGE (cl)-[:HAS_REVISION]->(revision)
            SET cl.subject_id = $subject_id,
                cl.transaction_from = $decided_at,
                cl.transaction_to = NULL,
                cl.is_superseded = false,
                cl.review_reconciled = true
            RETURN cl.claim_id AS claim_id
            """,
            workspace_id=workspace_id,
            claim_id=claim_id,
            subject_id=subject_id,
            decided_at=decided_at.isoformat(),
            review_decision_id=review_decision_id,
            revision_id=revision_id,
        )
        return bool(rows)

    async def list_claims_as_of(
        self, workspace_id: str, subject_id: str, as_of: datetime, *, limit: int = 1000, offset: int = 0
    ) -> list[Claim]:
        """True point-in-time reconstruction: every Claim whose transaction
        interval was open at `as_of` — recorded by then
        (transaction_from <= as_of) and not yet superseded by then
        (transaction_to IS NULL OR transaction_to > as_of). This is only
        honest as of Increment 19 because close_claim_interval above is now a
        real writer of transaction_to; before this increment every Claim's
        interval looked permanently open (see docs/evaluation.md) and this
        query would have silently returned every Claim regardless of `as_of`.

        Still narrower than "everything ever superseded": a Claim reconciled
        via ReviewService's subject_id rewrite (src/review/service.py) does
        not go through close_claim_interval and so does not close its
        interval — documented explicitly in docs/evaluation.md, not silently
        assumed away.
        """
        current_match = scoped_match("Claim", "cl", subject_id="subject_id")
        revision_match = scoped_match("ClaimRevision", "revision", subject_id="subject_id")
        rows = await self._executor.tenant_query(
            f"""
            MATCH {current_match}
            WHERE cl.transaction_from <= $as_of
              AND (cl.transaction_to IS NULL OR cl.transaction_to > $as_of)
            RETURN {_CLAIM_RETURN}
            UNION ALL
            MATCH {revision_match}
            WHERE revision.transaction_from <= $as_of
              AND (revision.transaction_to IS NULL OR revision.transaction_to > $as_of)
            RETURN {_claim_return("revision")}
            ORDER BY claim_id SKIP $offset LIMIT $limit
            """,
            workspace_id=workspace_id,
            subject_id=subject_id,
            as_of=as_of.isoformat(),
            offset=offset,
            limit=limit,
        )
        return [Claim(**row) for row in rows]

    async def erase_claims_for_subject(self, workspace_id: str, subject_id: str) -> list[tuple[str, str | None]]:
        """GDPR Art. 17 execution (docs/evaluation.md's Showpad engineering-
        rigor assessment, 2026-08-08, Band 3: "ErasureEvent is defined and
        never constructed anywhere... GDPR Art. 17 is modeled, not
        implemented"). Called only by src/usecases/erasure.py, itself only
        reachable via an explicit, authenticated erasure request -- never
        part of any routine read/write path.

        Sets erasure_status=ERASED (Claim already had this field, unused
        until now) and redacts object_value -- the one piece of free text a
        Claim owns directly (a literal like "wants a 20% discount", as
        opposed to object_id, which references another entity and carries
        no personal text of its own). Returns (claim_id,
        source_segment_id) pairs so the caller can also redact the
        TranscriptSegment text those Claims' evidence spans point into --
        a Claim doesn't duplicate that text itself
        (src/redaction/pii.py's locked-in design keeps raw transcript text
        on TranscriptSegment only).

        This is a deliberate, audited EXCEPTION to that same locked-in
        design ("raw text stays verbatim at rest... required by the
        evidence model"), not a reversal of it: the default path still
        preserves evidence; this path only ever runs for one specific,
        already-erasure-requested subject, and exists precisely because a
        valid erasure request overrides "keep everything for evidence" for
        that subject going forward.

        Does NOT touch the Neo4j vector-embedding property
        (src/embedding/backfill.py) or the optional Qdrant backend
        (src/embedding/qdrant_backend.py). ErasureEvent.erasure_scope's own
        docstring names "embeddings" as an example of what a real erasure
        scope might include; this MVP's completed event lists only what it
        actually erased so it never overclaims coverage it doesn't have --
        see docs/evaluation.md.
        """
        match = scoped_match("Claim", "cl", subject_id="subject_id")
        rows = await self._executor.tenant_query(
            f"""
            MATCH {match}
            WHERE cl.erasure_status <> $erased
            SET cl.erasure_status = $erased,
                cl.object_value = CASE WHEN cl.object_value IS NOT NULL THEN $redacted ELSE NULL END
            RETURN cl.claim_id AS claim_id, cl.source_segment_id AS source_segment_id
            """,
            workspace_id=workspace_id,
            subject_id=subject_id,
            erased=ErasureStatus.ERASED.value,
            redacted="[erased]",
        )
        if rows:
            # "erased" was already an anticipated CLAIMS_TOTAL label value
            # (see this module's import of CLAIMS_TOTAL and the metric's
            # own docstring in src/core/telemetry.py) -- this is the first
            # code path that actually produces it.
            CLAIMS_TOTAL.labels(event="erased").inc(len(rows))
        return [(row["claim_id"], row["source_segment_id"]) for row in rows]
