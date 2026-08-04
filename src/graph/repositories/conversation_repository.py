"""Tenant-safe repository for Conversation, TranscriptSegment, Participant, and
SpeakerResolution — same pattern as crm_repository.py.

Segments and participants are routed through Conversation (MERGE (c)-[:HAS_
SEGMENT]->(seg), (c)-[:HAS_PARTICIPANT]->(p)) — §10's 'no direct Claim or
Mention fan-out from Account; route evidence through Conversation,
TranscriptSegment, and Opportunity' principle applies just as much to segments
and participants themselves, which are the routing path for everything else.
"""

from __future__ import annotations

from src.domain.conversation import Conversation, Participant, SpeakerResolution, TranscriptSegment
from src.graph.execution import GraphExecutor, scoped_match

_CONVERSATION_RETURN = (
    "c.conversation_id AS conversation_id, c.workspace_id AS workspace_id, "
    "c.source_record_id AS source_record_id, c.source_system AS source_system, "
    "c.external_call_id AS external_call_id, c.occurred_at AS occurred_at, "
    "c.opportunity_id AS opportunity_id, c.account_id AS account_id"
)

_SEGMENT_RETURN = (
    "seg.segment_id AS segment_id, seg.workspace_id AS workspace_id, "
    "seg.conversation_id AS conversation_id, seg.source_segment_index AS source_segment_index, "
    "seg.speaker_label AS speaker_label, seg.text AS text, "
    "seg.start_ms AS start_ms, seg.end_ms AS end_ms"
)

_PARTICIPANT_RETURN = (
    "p.participant_id AS participant_id, p.workspace_id AS workspace_id, "
    "p.conversation_id AS conversation_id, p.speaker_label AS speaker_label, "
    "p.contact_id AS contact_id, p.seller_id AS seller_id, p.role AS role"
)


class ConversationRepository:
    def __init__(self, executor: GraphExecutor | None = None):
        self._executor = executor or GraphExecutor()

    async def upsert_conversation(self, conversation: Conversation) -> None:
        match = scoped_match("Conversation", "c", conversation_id="conversation_id")
        await self._executor.tenant_query(
            f"""
            MERGE {match}
            ON CREATE SET c.created_at = datetime()
            SET c.source_record_id = $source_record_id,
                c.source_system = $source_system,
                c.external_call_id = $external_call_id,
                c.occurred_at = $occurred_at,
                c.opportunity_id = $opportunity_id,
                c.account_id = $account_id
            """,
            workspace_id=conversation.workspace_id,
            conversation_id=conversation.conversation_id,
            source_record_id=conversation.source_record_id,
            source_system=conversation.source_system,
            external_call_id=conversation.external_call_id,
            occurred_at=conversation.occurred_at.isoformat(),
            opportunity_id=conversation.opportunity_id,
            account_id=conversation.account_id,
        )

    async def get_conversation(self, workspace_id: str, conversation_id: str) -> Conversation | None:
        match = scoped_match("Conversation", "c", conversation_id="conversation_id")
        rows = await self._executor.tenant_query(
            f"MATCH {match} RETURN {_CONVERSATION_RETURN}",
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
        return Conversation(**rows[0]) if rows else None

    async def list_conversations_by_opportunity(self, workspace_id: str, opportunity_id: str) -> list[Conversation]:
        match = scoped_match("Conversation", "c", opportunity_id="opportunity_id")
        rows = await self._executor.tenant_query(
            f"MATCH {match} RETURN {_CONVERSATION_RETURN}",
            workspace_id=workspace_id,
            opportunity_id=opportunity_id,
        )
        return [Conversation(**row) for row in rows]

    async def upsert_segment(self, segment: TranscriptSegment) -> None:
        conv_match = scoped_match("Conversation", "c", conversation_id="conversation_id")
        await self._executor.tenant_query(
            f"""
            MATCH {conv_match}
            MERGE (seg:TranscriptSegment {{segment_id: $segment_id}})
            SET seg.workspace_id = $workspace_id,
                seg.conversation_id = $conversation_id,
                seg.source_segment_index = $source_segment_index,
                seg.speaker_label = $speaker_label,
                seg.text = $text,
                seg.start_ms = $start_ms,
                seg.end_ms = $end_ms
            MERGE (c)-[:HAS_SEGMENT]->(seg)
            """,
            workspace_id=segment.workspace_id,
            conversation_id=segment.conversation_id,
            segment_id=segment.segment_id,
            source_segment_index=segment.source_segment_index,
            speaker_label=segment.speaker_label,
            text=segment.text,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
        )

    async def get_segment(self, workspace_id: str, segment_id: str) -> TranscriptSegment | None:
        # conversation_id isn't recoverable from segment_id alone (it's a hash),
        # so this searches across the workspace's conversations for the one
        # whose HAS_SEGMENT edge matches — still workspace-scoped throughout.
        conv_match = scoped_match("Conversation", "c")
        rows = await self._executor.tenant_query(
            f"""
            MATCH {conv_match}
            MATCH (c)-[:HAS_SEGMENT]->(seg:TranscriptSegment {{segment_id: $segment_id}})
            RETURN {_SEGMENT_RETURN}
            """,
            workspace_id=workspace_id,
            segment_id=segment_id,
        )
        return TranscriptSegment(**rows[0]) if rows else None

    async def list_segments(self, workspace_id: str, conversation_id: str) -> list[TranscriptSegment]:
        conv_match = scoped_match("Conversation", "c", conversation_id="conversation_id")
        rows = await self._executor.tenant_query(
            f"""
            MATCH {conv_match}
            MATCH (c)-[:HAS_SEGMENT]->(seg:TranscriptSegment)
            RETURN {_SEGMENT_RETURN}
            ORDER BY seg.source_segment_index
            """,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
        return [TranscriptSegment(**row) for row in rows]

    async def upsert_participant(self, participant: Participant) -> None:
        conv_match = scoped_match("Conversation", "c", conversation_id="conversation_id")
        await self._executor.tenant_query(
            f"""
            MATCH {conv_match}
            MERGE (p:Participant {{participant_id: $participant_id}})
            SET p.workspace_id = $workspace_id,
                p.conversation_id = $conversation_id,
                p.speaker_label = $speaker_label,
                p.contact_id = $contact_id,
                p.seller_id = $seller_id,
                p.role = $role
            MERGE (c)-[:HAS_PARTICIPANT]->(p)
            """,
            workspace_id=participant.workspace_id,
            conversation_id=participant.conversation_id,
            participant_id=participant.participant_id,
            speaker_label=participant.speaker_label,
            contact_id=participant.contact_id,
            seller_id=participant.seller_id,
            role=participant.role.value,
        )

    async def list_participants(self, workspace_id: str, conversation_id: str) -> list[Participant]:
        conv_match = scoped_match("Conversation", "c", conversation_id="conversation_id")
        rows = await self._executor.tenant_query(
            f"""
            MATCH {conv_match}
            MATCH (c)-[:HAS_PARTICIPANT]->(p:Participant)
            RETURN {_PARTICIPANT_RETURN}
            """,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
        return [Participant(**row) for row in rows]

    async def upsert_speaker_resolution(self, resolution: SpeakerResolution) -> None:
        conv_match = scoped_match("Conversation", "c", conversation_id="conversation_id")
        await self._executor.tenant_query(
            f"""
            MATCH {conv_match}
            MERGE (r:SpeakerResolution {{resolution_id: $resolution_id}})
            SET r.workspace_id = $workspace_id,
                r.conversation_id = $conversation_id,
                r.speaker_label = $speaker_label,
                r.resolved_contact_id = $resolved_contact_id,
                r.resolved_seller_id = $resolved_seller_id,
                r.role = $role,
                r.evidence = $evidence
            MERGE (c)-[:HAS_SPEAKER_RESOLUTION]->(r)
            """,
            workspace_id=resolution.workspace_id,
            conversation_id=resolution.conversation_id,
            resolution_id=resolution.resolution_id,
            speaker_label=resolution.speaker_label,
            resolved_contact_id=resolution.resolved_contact_id,
            resolved_seller_id=resolution.resolved_seller_id,
            role=resolution.role.value,
            evidence=resolution.evidence,
        )
