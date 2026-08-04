"""§8 speaker resolution — 'Resolve opaque speaker IDs before calculating Claim
authority, using participant email, seller directory, meeting invitees,
self-introduction evidence, and relationship context. Failed speaker resolution
produces speaker_role=UNKNOWN; it does not discard the Claim.'

This vertical slice implements the deterministic subset that's provable without
an LLM: participant-email-against-known-Contact/Seller-directory matching. The
other named signals (meeting invitees, self-introduction evidence, relationship
context) are real future extensions, not simulated here — an unmatched speaker
resolves to UNKNOWN honestly rather than guessing.
"""

from __future__ import annotations

import hashlib

from src.domain.conversation import SpeakerResolution
from src.domain.enums import SpeakerRole

_SEP = "\x1f"


def _resolution_id(conversation_id: str, speaker_label: str) -> str:
    joined = _SEP.join((conversation_id, speaker_label))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def resolve_speaker(
    *,
    workspace_id: str,
    conversation_id: str,
    speaker_label: str,
    raw_email: str | None,
    email_to_contact_id: dict[str, str],
    email_to_seller_id: dict[str, str],
) -> SpeakerResolution:
    normalized_email = raw_email.strip().lower() if raw_email else None
    resolved_contact_id = email_to_contact_id.get(normalized_email) if normalized_email else None
    resolved_seller_id = email_to_seller_id.get(normalized_email) if normalized_email else None

    if resolved_contact_id:
        role = SpeakerRole.BUYER
    elif resolved_seller_id:
        role = SpeakerRole.SELLER
    else:
        role = SpeakerRole.UNKNOWN

    return SpeakerResolution(
        resolution_id=_resolution_id(conversation_id, speaker_label),
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        speaker_label=speaker_label,
        resolved_contact_id=resolved_contact_id,
        resolved_seller_id=resolved_seller_id,
        role=role,
        evidence="participant_email_match" if role != SpeakerRole.UNKNOWN else None,
    )
