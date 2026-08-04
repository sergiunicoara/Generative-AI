"""Tenant-safe repository for Claim — same pattern as crm_repository.py.

Per docs/plan.md §10 ("no direct Claim or Mention fan-out from Account; route
evidence through Conversation, TranscriptSegment, and Opportunity"), this
repository never accepts an Account id — only subject_id (whatever entity the
Claim is actually about) and workspace_id.
"""

from __future__ import annotations

from src.domain.assertion import Claim
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
    "cl.retention_class AS retention_class, cl.erasure_status AS erasure_status, "
    "cl.created_at AS created_at"
)


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
        "retention_class": claim.retention_class,
        "erasure_status": claim.erasure_status.value,
        "created_at": claim.created_at.isoformat(),
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
                ON CREATE SET cl.created_at = $created_at
                SET cl.subject_id = $subject_id,
                    cl.predicate = $predicate,
                    cl.object_id = $object_id,
                    cl.object_value = $object_value,
                    cl.polarity = $polarity,
                    cl.source_type = $source_type,
                    cl.source_record_id = $source_record_id,
                    cl.source_segment_id = $source_segment_id,
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
                    cl.adjudication_status = $adjudication_status,
                    cl.retention_class = $retention_class,
                    cl.erasure_status = $erasure_status
                MERGE (seg)-[:HAS_CLAIM]->(cl)
                """,
                **params,
            )
            return
        await self._executor.tenant_query(
            f"""
            MERGE {match}
            ON CREATE SET cl.created_at = $created_at
            SET cl.subject_id = $subject_id,
                cl.predicate = $predicate,
                cl.object_id = $object_id,
                cl.object_value = $object_value,
                cl.polarity = $polarity,
                cl.source_type = $source_type,
                cl.source_record_id = $source_record_id,
                cl.source_segment_id = $source_segment_id,
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
                cl.adjudication_status = $adjudication_status,
                cl.retention_class = $retention_class,
                cl.erasure_status = $erasure_status
            """,
            **params,
        )

    async def get_claim(self, workspace_id: str, claim_id: str) -> Claim | None:
        match = scoped_match("Claim", "cl", claim_id="claim_id")
        rows = await self._executor.tenant_query(
            f"MATCH {match} RETURN {_CLAIM_RETURN}",
            workspace_id=workspace_id,
            claim_id=claim_id,
        )
        return Claim(**rows[0]) if rows else None

    async def list_claims_by_subject(self, workspace_id: str, subject_id: str) -> list[Claim]:
        match = scoped_match("Claim", "cl", subject_id="subject_id")
        rows = await self._executor.tenant_query(
            f"MATCH {match} RETURN {_CLAIM_RETURN}",
            workspace_id=workspace_id,
            subject_id=subject_id,
        )
        return [Claim(**row) for row in rows]

    async def list_claims_for_conversation(self, workspace_id: str, conversation_id: str) -> list[Claim]:
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
            """,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
        return [Claim(**row) for row in rows]
