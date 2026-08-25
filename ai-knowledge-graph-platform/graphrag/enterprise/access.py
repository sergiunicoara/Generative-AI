"""Permission-aware retrieval helpers.

All document and chunk searches use the same predicate.  This is important:
filtering only after retrieval would expose protected text to rerankers, graph
expansion or the answer cache before it was eventually discarded.
"""

from __future__ import annotations

from graphrag.enterprise.models import AccessContext, DocumentAccessPolicy


def access_params(context: AccessContext | None, *, enabled: bool) -> dict:
    context = context or AccessContext()
    return {
        "acl_enabled": bool(enabled),
        "acl_principals": context.principals,
        "acl_groups_resolved": context.groups_resolved,
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
    }
