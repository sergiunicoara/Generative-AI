"""Gong-shaped transcript adapter (docs/plan.md §4 initial adapters).

Parses a raw Gong-style call export (`id`, `started`, `parties[]`, `transcript[]`
of speaker turns each with `sentences[]`) into Conversation + Participant +
TranscriptSegment. Every source segment is persisted even when extraction is
skipped — this adapter only parses; the ingestion pipeline (Increment 5's
src/ingestion/pipeline.py) is responsible for actually persisting segments
before any extraction runs (§7).
"""

from __future__ import annotations

from src.domain.conversation import Conversation, Participant, TranscriptSegment
from src.domain.enums import SpeakerRole
from src.domain.identity import conversation_id as _conversation_id, participant_id, segment_id
from src.ingestion.adapters.base import ParsedRecord, compute_content_hash


class GongAdapter:
    source_system = "gong"
    # This fixture models Gong calls carrying an explicit `deleted` flag on
    # re-export — a trustworthy per-call deletion signal (§6), unlike a partial
    # incremental batch with no way to distinguish "not in this batch" from
    # "actually deleted".
    supports_deletion_signal = True

    def parse_conversation(
        self,
        workspace_id: str,
        raw: dict,
        *,
        opportunity_id: str | None = None,
        account_id: str | None = None,
    ) -> ParsedRecord:
        external_id = raw["id"]
        conv_id = _conversation_id(workspace_id, self.source_system, external_id)
        conversation = Conversation(
            conversation_id=conv_id,
            workspace_id=workspace_id,
            source_record_id=conv_id,
            source_system=self.source_system,
            external_call_id=external_id,
            occurred_at=raw["started"],
            opportunity_id=opportunity_id,
            account_id=account_id,
        )
        return ParsedRecord(
            entity=conversation,
            external_id=external_id,
            object_type="Conversation",
            content_hash=compute_content_hash(raw),
            extra={"is_deleted": bool(raw.get("deleted"))},
        )

    def parse_participants(self, workspace_id: str, conversation_id_: str, raw: dict) -> list[Participant]:
        participants = []
        for party in raw.get("parties", []):
            speaker_label = party["speakerId"]
            participants.append(Participant(
                participant_id=participant_id(conversation_id_, speaker_label),
                workspace_id=workspace_id,
                conversation_id=conversation_id_,
                speaker_label=speaker_label,
                role=SpeakerRole.UNKNOWN,  # resolved separately — src/resolution/speaker.py
            ))
        return participants

    def parse_party_emails(self, raw: dict) -> dict[str, str | None]:
        """speaker_label -> raw email (or None for a genuinely opaque speaker —
        the required 'opaque speaker IDs' fixture has parties entries with no
        emailAddress at all)."""
        return {party["speakerId"]: party.get("emailAddress") for party in raw.get("parties", [])}

    def parse_segments(self, workspace_id: str, conversation_id_: str, raw: dict) -> list[TranscriptSegment]:
        segments: list[TranscriptSegment] = []
        index = 0
        for turn in raw.get("transcript", []):
            speaker_label = turn["speakerId"]
            for sentence in turn.get("sentences", []):
                segments.append(TranscriptSegment(
                    segment_id=segment_id(conversation_id_, index),
                    workspace_id=workspace_id,
                    conversation_id=conversation_id_,
                    source_segment_index=index,
                    speaker_label=speaker_label,
                    text=sentence["text"],
                    start_ms=sentence.get("start"),
                    end_ms=sentence.get("end"),
                ))
                index += 1
        return segments
