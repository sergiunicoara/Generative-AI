"""One definition of "this fact may be used as current", shared by every read path.

Before this module each retrieval query inlined its own variant: only one path
checked edge expiry (and only when ``as_of`` was passed), none read
``confidence_state``, quarantine was spelled two ways. Every graph read that
returns facts now uses these fragments (docs/trust-metadata.md).

All fragments take the Cypher variable name. ``AT`` is the instant facts are
judged at: the caller's ``$as_of`` when given, otherwise now. A query using it
must always pass ``as_of`` (``None`` is fine).
"""
from __future__ import annotations

AT = "coalesce(datetime($as_of), datetime())"


def edge_is_current(var: str = "r", at: str = AT) -> str:
    """Not retracted and not expired at ``at``. DISPUTED edges stay usable but are
    flagged and down-weighted (trust.py), never silently treated as settled."""
    return (
        f"coalesce({var}.confidence_state, 'ASSERTED') <> 'RETRACTED' "
        f"AND ({var}.valid_from IS NULL OR {var}.valid_from <= {at}) "
        f"AND ({var}.valid_to IS NULL OR {var}.valid_to > {at})"
    )


def entity_is_active(var: str = "e") -> str:
    return f"coalesce({var}.quarantined, false) = false"


def edge_trust_fields(var: str = "r") -> str:
    """RETURN items exposing an edge's trust metadata. ``verification_status`` is
    UNVERIFIED unless a reviewer set it: having a source is not verification."""
    return (
        f"coalesce({var}.confidence_state, 'ASSERTED') AS confidence_state, "
        f"coalesce({var}.origin, CASE coalesce({var}.source_type, 'document') "
        f"WHEN 'inferred' THEN 'INFERRED' WHEN 'manual' THEN 'MANUAL' "
        f"WHEN 'llm' THEN 'GENERATED' ELSE 'EXTRACTED' END) AS origin, "
        f"coalesce({var}.verification_status, 'UNVERIFIED') AS verification_status, "
        f"{var}.verified_by AS verified_by, {var}.generated_by AS generated_by, "
        f"toString({var}.valid_from) AS valid_from, toString({var}.valid_to) AS valid_to, "
        f"toString({var}.stale_after) AS stale_after, {var}.schema_version AS schema_version"
    )


def document_trust_fields(var: str = "d") -> str:
    """RETURN items for the document behind a chunk (null-safe when ``var`` is null)."""
    return (
        f"{var}.id AS document_id, {var}.authority_level AS authority_level, "
        f"({var}.superseded_by IS NOT NULL) AS superseded, {var}.superseded_by AS superseded_by, "
        f"toString({var}.valid_from) AS doc_valid_from, toString({var}.valid_to) AS doc_valid_to, "
        f"toString({var}.stale_after) AS doc_stale_after, {var}.schema_version AS schema_version"
    )
