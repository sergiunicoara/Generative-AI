from __future__ import annotations

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from graphrag.enterprise.access import access_params, document_access_predicate, normalise_policy
from graphrag.enterprise.models import ACLState, AccessContext, DocumentAccessPolicy, MetadataEnvelope
from graphrag.retrieval.query_cache import QueryCacheContext, build_cache_key


def _cache_context(**overrides) -> QueryCacheContext:
    return QueryCacheContext(
        corpus_revision=1,
        requested_mode="hybrid",
        effective_mode="hybrid",
        model_route={"provider": "test"},
        prompt_version="v1",
        retrieval_config={},
        ontology_version="v1",
        **overrides,
    )


def test_access_context_requires_explicit_group_resolution_claim() -> None:
    unresolved = AccessContext.from_claims({"sub": "alice"})
    resolved = AccessContext.from_claims({"sub": "alice", "groups": ["legal", "legal"]})

    assert unresolved.principals == ["user:alice"]
    assert unresolved.groups_resolved is False
    assert resolved.principals == ["group:legal", "user:alice"]
    assert resolved.groups_resolved is True
    assert unresolved.fingerprint != resolved.fingerprint


def test_restricted_acl_requires_an_allowed_principal() -> None:
    with pytest.raises(ValidationError):
        DocumentAccessPolicy(mode="restricted", state=ACLState.KNOWN)


def test_tenant_policy_is_normalised_to_known_acl() -> None:
    policy = normalise_policy(DocumentAccessPolicy())
    assert policy["access_mode"] == "tenant"
    assert policy["acl_state"] == "known"


def test_acl_predicate_is_fail_closed_when_enforced() -> None:
    predicate = document_access_predicate("document")
    params = access_params(AccessContext.from_claims({"sub": "alice"}), enabled=True)

    assert "coalesce(document.acl_state, 'unknown') = 'known'" in predicate
    assert "deny_principals" in predicate
    assert params["acl_enabled"] is True
    assert params["acl_groups_resolved"] is False


def test_answer_cache_is_partitioned_by_entitlement() -> None:
    public_key = build_cache_key("question", "tenant-a", _cache_context(access_fingerprint="a"))
    restricted_key = build_cache_key("question", "tenant-a", _cache_context(access_fingerprint="b"))
    assert public_key != restricted_key


def test_metadata_envelope_rejects_an_invalid_effective_interval() -> None:
    with pytest.raises(ValidationError):
        MetadataEnvelope(
            effective_from=datetime(2026, 2, 1, tzinfo=timezone.utc),
            effective_to=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
