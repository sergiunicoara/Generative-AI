"""Deterministic source/assertion identity (docs/plan.md §6).

All IDs are sha256 hex digests over their component fields, joined with the ASCII
Unit Separator (\\x1f) rather than a printable delimiter like "|" — printable
delimiters that could plausibly appear inside a field value create adjacency
collisions (e.g. crm_entity_id("a|b", "c", ...) == crm_entity_id("a", "b|c", ...)
if "|" were used and a field legitimately contained "|"). \\x1f cannot appear in
any of these fields (workspace/system/type identifiers, external IDs, predicates,
normalized text), so the join is unambiguous — recovering the exact field
boundaries from the digest is impossible either way (it's a hash), but the digest
itself is guaranteed not to collide purely because two adjacent fields' content
shifted across a boundary.

Full (untruncated) sha256 hex digests are used throughout — no length truncation.
Truncating trades collision resistance for shorter IDs with no functional benefit
here (these are opaque keys, never displayed to end users), so the full digest is
kept.

These are pure functions: same input always produces the same output, in any
process, with no I/O — this is the whole point of "stable source identities" (§6)
and is what makes "second identical ingest changes zero graph counts" (§15)
possible without a lookup table.
"""

from __future__ import annotations

import hashlib

_SEP = "\x1f"


def _hash(*parts: object) -> str:
    joined = _SEP.join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def crm_entity_id(workspace: str, source_system: str, object_type: str, external_id: str) -> str:
    """§6 — crm_entity_id = hash(workspace | source_system | object_type | external_id)."""
    return _hash(workspace, source_system, object_type, external_id)


def conversation_id(workspace: str, source_system: str, external_call_id: str) -> str:
    """§6 — conversation_id = hash(workspace | source_system | external_call_id)."""
    return _hash(workspace, source_system, external_call_id)


def segment_id(conversation_id_: str, source_segment_index: int) -> str:
    """§6 — segment_id = hash(conversation_id | source_segment_index)."""
    return _hash(conversation_id_, source_segment_index)


def participant_id(conversation_id_: str, speaker_label: str) -> str:
    """Not in §6's explicit list, same deterministic-derivation reasoning as
    snapshot_id() — a participant is identified by its (conversation, opaque
    speaker label) pair, so re-parsing the same call never creates duplicates."""
    return _hash(conversation_id_, speaker_label)


def snapshot_id(source_record_id: str, source_version: int) -> str:
    """Not in §6's explicit list, but derived the same deterministic way for the
    same reason as everything else here: a retried ingestion run recomputing the
    same (source_record_id, source_version) pair must land on the same
    SourceSnapshot id, not a fresh random one — that's what makes reconciliation
    idempotent under retry, not just under a single successful run."""
    return _hash(source_record_id, source_version)


def mention_id(
    segment_id_: str,
    char_start: int,
    char_end: int,
    normalized_surface: str,
    entity_type: str,
) -> str:
    """§6 — mention_id = hash(segment_id | char_start | char_end | normalized_surface | entity_type)."""
    return _hash(segment_id_, char_start, char_end, normalized_surface, entity_type)


def assertion_id(
    workspace: str,
    source_segment_id: str,
    evidence_char_start: int,
    evidence_char_end: int,
    canonical_subject: str,
    predicate: str,
    normalized_object: str,
    polarity: str,
) -> str:
    """§6 — assertion_id deliberately excludes extractor version, provider, model,
    prompt_version, and run_nonce (see extraction_run_id below) — those describe
    HOW a Claim was found, not WHAT was found. An identical assertion re-found by
    a newer extractor must link to the existing Claim, not create a duplicate; the
    exclusion is structural here (extraction execution details are simply not
    parameters of this function), not a convention callers must remember to honor.
    """
    return _hash(
        workspace,
        source_segment_id,
        evidence_char_start,
        evidence_char_end,
        canonical_subject,
        predicate,
        normalized_object,
        polarity,
    )


def extraction_run_id(
    provider: str,
    model: str,
    prompt_version: str,
    extractor_version: str,
    run_nonce: str,
) -> str:
    """§6 — extraction_run_id = hash(provider | model | prompt_version | extractor_version | run_nonce).

    Identifies one extraction EXECUTION, not the resulting assertion — see
    assertion_id's docstring for why the two are deliberately split.
    """
    return _hash(provider, model, prompt_version, extractor_version, run_nonce)
