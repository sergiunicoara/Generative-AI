"""Permission-aware retrieval helpers.

All document and chunk searches use the same predicate.  This is important:
filtering only after retrieval would expose protected text to rerankers, graph
expansion or the answer cache before it was eventually discarded.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os

from graphrag.enterprise.models import AccessContext, DocumentAccessPolicy

#: Permission-freshness bound.  A restricted document whose ACL was last read
#: from its source longer ago than this is denied until the source is read
#: again.  Answers cached while it was readable stop being served at the next
#: ACL epoch (see ``acl_epoch``), so a revocation that never reaches the
#: platform stops affecting answers within at most twice this bound; one that
#: is re-synced takes effect on the next request (the re-ingest advances the
#: corpus revision every cache key includes).
ACL_MAX_AGE = timedelta(seconds=int(os.getenv("ACL_MAX_AGE_SECONDS", "3600")))


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def acl_fresh_after(now: datetime | None = None) -> str:
    """Oldest acceptable ACL observation time, in the stored string form."""
    return _utc_iso((now or datetime.now(timezone.utc)) - ACL_MAX_AGE)


def acl_epoch(now: datetime | None = None) -> int:
    """Index of the current freshness window, part of the answer-cache key."""
    seconds = max(1, int(ACL_MAX_AGE.total_seconds()))
    return int((now or datetime.now(timezone.utc)).timestamp()) // seconds


def acl_is_fresh(policy: DocumentAccessPolicy, now: datetime | None = None) -> bool:
    """Python mirror of the Cypher freshness clause."""
    if policy.mode != "restricted" or policy.observed_at is None:
        return True
    return _utc_iso(policy.observed_at) >= acl_fresh_after(now)


def access_params(context: AccessContext | None, *, enabled: bool) -> dict:
    context = context or AccessContext()
    return {
        "acl_enabled": bool(enabled),
        "acl_principals": context.principals,
        "acl_groups_resolved": context.groups_resolved,
        "acl_fresh_after": acl_fresh_after(),
    }


def document_access_predicate(alias: str = "d") -> str:
    """Cypher predicate for fail-closed document access enforcement.

    Existing corpora remain available while ``acl_enabled`` is false.  Once a
    deployment enables ACL enforcement, every document must have a known policy;
    restricted documents additionally require a resolved group claim when their
    ACL depends on groups.  Explicit deny always wins over allow.
    """

    return f"""
      AND (
        NOT $acl_enabled
        OR (
          coalesce({alias}.acl_state, 'unknown') = 'known'
          AND NOT any(principal IN $acl_principals
                      WHERE principal IN coalesce({alias}.deny_principals, []))
          AND (
            coalesce({alias}.access_mode, 'restricted') = 'tenant'
            OR (
              any(principal IN $acl_principals
                  WHERE principal IN coalesce({alias}.allow_principals, []))
              AND (
                NOT coalesce({alias}.requires_group_resolution, false)
                OR $acl_groups_resolved
              )
              AND ({alias}.acl_observed_at IS NULL OR {alias}.acl_observed_at >= $acl_fresh_after)
            )
          )
        )
      )
    """


def link_access_predicate(alias: str = "link") -> str:
    """The same fail-closed ACL rule applied to a persisted link snapshot.

    A document-link traversal must be authorised at three levels: source
    document, the source ACL captured when the explicit link was observed, and
    target document.  Keeping this predicate independent avoids assuming that
    a later source ACL update retroactively authorises a stale edge.
    """

    return f"""
      AND (
        NOT $acl_enabled
        OR (
          coalesce({alias}.acl_state, 'unknown') = 'known'
          AND NOT any(principal IN $acl_principals
                      WHERE principal IN coalesce({alias}.deny_principals, []))
          AND (
            coalesce({alias}.access_mode, 'restricted') = 'tenant'
            OR (
              any(principal IN $acl_principals
                  WHERE principal IN coalesce({alias}.allow_principals, []))
              AND (
                NOT coalesce({alias}.requires_group_resolution, false)
                OR $acl_groups_resolved
              )
            )
          )
        )
      )
    """


def normalise_policy(policy: DocumentAccessPolicy) -> dict:
    """Return Neo4j-safe scalar/list fields for a document node."""

    state = policy.state.value
    # A tenant-scoped policy is fully known even though it has no external ACL.
    if policy.mode == "tenant" and policy.state.value == "not_applicable":
        state = "known"
    return {
        "access_mode": policy.mode,
        "acl_state": state,
        "allow_principals": policy.allow_principals,
        "deny_principals": policy.deny_principals,
        "requires_group_resolution": policy.requires_group_resolution,
        "acl_observed_at": _utc_iso(policy.observed_at) if policy.observed_at else None,
    }
