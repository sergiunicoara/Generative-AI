"""Unit tests for OntologyRegistry — domain/range enforcement, migration, domain rules."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag.graph.ontology_registry import (
    OntologyRegistry,
    _RELATION_RULES,
    _registries,
    get_ontology_registry,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def neo4j():
    mock = AsyncMock()
    mock.run = AsyncMock(side_effect=[
        [],                              # MATCH existing relations
        [{"version_id": "v-test-1"}],    # MERGE OntologyVersion
    ])
    return mock


@pytest.fixture
async def registry(neo4j):
    r = OntologyRegistry(neo4j)
    await r.load(["PERSON", "ORG", "PRODUCT", "CONCEPT", "LOCATION", "EVENT"])
    return r


def _entity(name: str, etype: str):
    e = MagicMock()
    e.id   = f"e_{name}"
    e.name = name
    e.type = etype
    return e


def _relation(src, tgt, rel: str):
    r = MagicMock()
    r.source_entity_id = src.id
    r.target_entity_id = tgt.id
    r.relation = rel
    return r


def test_get_ontology_registry_is_isolated_by_tenant():
    _registries.clear()
    neo4j = AsyncMock()

    marketing = get_ontology_registry(neo4j, tenant="marketing")
    automotive = get_ontology_registry(neo4j, tenant="automotive")

    assert marketing is not automotive
    assert marketing._tenant == "marketing"
    assert automotive._tenant == "automotive"


async def test_graph_migration_counts_and_updates_only_the_registry_tenant():
    neo4j = AsyncMock()
    neo4j.run = AsyncMock(return_value=[{"total": 2}])
    registry = OntologyRegistry(neo4j, tenant="marketing")
    registry._migration_map = {"OLD_RULE": "NEW_RULE"}

    result = await registry.apply_graph_migrations()

    assert result == {"OLD_RULE": 2}
    count_call, update_call, _ = neo4j.run.await_args_list
    assert "tenant: $tenant" in count_call.args[0]
    assert count_call.kwargs["tenant"] == "marketing"
    assert update_call.kwargs["tenant"] == "marketing"


# ── validate_extraction ────────────────────────────────────────────────────────

class TestValidateExtraction:
    async def test_unknown_type_corrected_to_concept(self, registry):
        e = _entity("Widget", "UNKNOWN_TYPE")
        registry.validate_extraction([e], [])
        assert e.type == "CONCEPT"

    async def test_known_type_unchanged(self, registry):
        e = _entity("Elon Musk", "PERSON")
        registry.validate_extraction([e], [])
        assert e.type == "PERSON"

    async def test_relation_uppercased(self, registry):
        src = _entity("A", "PERSON")
        tgt = _entity("B", "ORG")
        rel = _relation(src, tgt, "works_at")
        registry.validate_extraction([src, tgt], [rel])
        assert rel.relation == "WORKS_AT"

    async def test_malformed_relation_falls_back_to_related_to(self, registry):
        src = _entity("A", "PERSON")
        tgt = _entity("B", "ORG")
        rel = _relation(src, tgt, "123invalid!")
        registry.validate_extraction([src, tgt], [rel])
        assert rel.relation == "RELATED_TO"

    async def test_domain_range_violation_falls_back(self, registry):
        """FOUNDED: (PERSON, ORG) — using (LOCATION, EVENT) should fall back."""
        src = _entity("London", "LOCATION")
        tgt = _entity("Launch", "EVENT")
        rel = _relation(src, tgt, "FOUNDED")
        registry.validate_extraction([src, tgt], [rel])
        assert rel.relation == "RELATED_TO"

    async def test_valid_triplet_passes(self, registry):
        src = _entity("Alice", "PERSON")
        tgt = _entity("Acme", "ORG")
        rel = _relation(src, tgt, "WORKS_AT")
        registry.validate_extraction([src, tgt], [rel])
        assert rel.relation == "WORKS_AT"

    async def test_drift_detected_on_new_relation(self, registry):
        src = _entity("A", "ORG")
        tgt = _entity("B", "ORG")
        rel = _relation(src, tgt, "PARTNER_OF")  # new, not in known_relations
        report = registry.validate_extraction([src, tgt], [rel])
        assert report["drift_detected"] is True
        assert "PARTNER_OF" in report["new_relations"]


# ── validate_relation_triplet ─────────────────────────────────────────────────

class TestValidateRelationTriplet:
    async def test_valid_triplet(self, registry):
        ok, norm = registry.validate_relation_triplet("PERSON", "CEO_OF", "ORG")
        assert ok is True
        assert norm == "CEO_OF"

    async def test_invalid_triplet_returns_false(self, registry):
        ok, norm = registry.validate_relation_triplet("LOCATION", "CEO_OF", "PERSON")
        assert ok is False

    async def test_open_relation_always_valid(self, registry):
        """RELATED_TO has empty domain/range — always passes."""
        ok, norm = registry.validate_relation_triplet("CONCEPT", "RELATED_TO", "EVENT")
        assert ok is True

    async def test_normalises_relation_name(self, registry):
        ok, norm = registry.validate_relation_triplet("PERSON", "works at", "ORG")
        assert norm == "WORKS_AT"

    async def test_migration_applied_before_check(self, registry):
        registry._migration_map["IS_CEO"] = "CEO_OF"
        ok, norm = registry.validate_relation_triplet("PERSON", "IS_CEO", "ORG")
        assert norm == "CEO_OF"
        assert ok is True


# ── add_domain_range_rules ────────────────────────────────────────────────────

class TestAddDomainRangeRules:
    async def test_new_domain_rule_added(self, registry):
        registry.add_domain_range_rules({
            "SUPERSEDES": {
                "domain": ["REGULATION"],
                "target": ["REGULATION"],
            }
        })
        assert "SUPERSEDES" in registry._domain_rules
        assert ("REGULATION", "REGULATION") in registry._domain_rules["SUPERSEDES"]

    async def test_domain_rule_validates_correctly(self, registry):
        registry.add_domain_range_rules({
            "APPLIES_TO": {
                "domain": ["REGULATION"],
                "target": ["PRODUCT", "ORG"],
            }
        })
        ok, _ = registry.validate_relation_triplet("REGULATION", "APPLIES_TO", "PRODUCT")
        assert ok is True

    async def test_domain_rule_rejects_wrong_types(self, registry):
        registry.add_domain_range_rules({
            "APPLIES_TO": {
                "domain": ["REGULATION"],
                "target": ["PRODUCT"],
            }
        })
        ok, norm = registry.validate_relation_triplet("PERSON", "APPLIES_TO", "LOCATION")
        assert ok is False
        assert norm == "APPLIES_TO"  # normalised but invalid

    async def test_domain_rules_extend_not_replace(self, registry):
        """Adding domain rules for a built-in relation merges with built-in pairs."""
        original_pairs = set(_RELATION_RULES.get("LOCATED_IN", set()))
        registry.add_domain_range_rules({
            "LOCATED_IN": {
                "domain": ["PRODUCT"],
                "target": ["LOCATION"],
            }
        })
        # Original built-in pairs should still be valid
        for pair in original_pairs:
            src_type, tgt_type = pair
            ok, _ = registry.validate_relation_triplet(src_type, "LOCATED_IN", tgt_type)
            assert ok is True, f"Built-in pair {pair} should still be valid"
        # New domain pair should also be valid
        ok, _ = registry.validate_relation_triplet("PRODUCT", "LOCATED_IN", "LOCATION")
        assert ok is True

    async def test_added_relation_registered_as_known(self, registry):
        registry.add_domain_range_rules({
            "MANDATED_BY": {"domain": ["REGULATION"], "target": ["ORG"]}
        })
        assert "MANDATED_BY" in registry._known_relations


# ── F13: tenant scoping of relation vocabulary and version history ────────────

class TestKnownRelationsScopedByTenant:
    """load()'s "known relation types from existing graph" query used to scan
    ALL tenants' RELATES_TO edges, so tenant A's vocabulary silently
    suppressed validate_extraction's "new relation" drift signal for tenant B.
    See docs/context_graph_gap_plan.md F13."""

    async def test_relation_scan_filters_by_tenant(self):
        neo4j = AsyncMock()
        neo4j.run = AsyncMock(side_effect=[
            [],                            # MATCH existing relations
            [{"version_id": "v-1"}],       # MERGE OntologyVersion
        ])
        registry = OntologyRegistry(neo4j, tenant="acme")
        await registry.load(["ORG"])

        relation_scan_call = neo4j.run.await_args_list[0]
        query = relation_scan_call.args[0]
        assert "tenant: $tenant" in query
        assert relation_scan_call.kwargs["tenant"] == "acme"


class TestOntologyVersionScopedByTenant:
    async def test_version_merge_includes_tenant_in_key(self):
        neo4j = AsyncMock()
        neo4j.run = AsyncMock(side_effect=[[], [{"version_id": "v-1"}]])
        registry = OntologyRegistry(neo4j, tenant="acme")
        await registry.load(["ORG"])

        merge_call = neo4j.run.await_args_list[1]
        assert "MERGE (o:OntologyVersion {schema_hash: $hash, tenant: $tenant})" in merge_call.args[0]
        assert merge_call.kwargs["tenant"] == "acme"

    async def test_two_tenants_with_identical_ontology_get_separate_versions(self):
        """Same schema_hash must not collapse two tenants into one shared
        OntologyVersion node — that would merge their governance histories."""
        neo4j = AsyncMock()
        neo4j.run = AsyncMock(side_effect=[
            [], [{"version_id": "v-acme"}],
            [], [{"version_id": "v-other"}],
        ])
        acme = OntologyRegistry(neo4j, tenant="acme")
        await acme.load(["ORG"])
        other = OntologyRegistry(neo4j, tenant="other")
        await other.load(["ORG"])

        acme_merge = neo4j.run.await_args_list[1]
        other_merge = neo4j.run.await_args_list[3]
        assert acme_merge.kwargs["tenant"] == "acme"
        assert other_merge.kwargs["tenant"] == "other"


class TestGetSchemaHistoryRequiresTenant:
    async def test_tenant_is_required_positional(self):
        neo4j = AsyncMock()
        registry = OntologyRegistry(neo4j, tenant="acme")
        with pytest.raises(TypeError):
            await registry.get_schema_history()

    async def test_query_filters_by_tenant(self):
        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[])
        registry = OntologyRegistry(neo4j, tenant="acme")

        await registry.get_schema_history("acme")

        call = neo4j.run.await_args
        assert "MATCH (o:OntologyVersion {tenant: $tenant})" in call.args[0]
        assert call.kwargs["tenant"] == "acme"

    async def test_other_tenants_versions_not_returned(self):
        """Adversarial: a caller passing tenant B must not see tenant A's
        history, regardless of what the fake Neo4j layer would return for an
        unscoped query."""
        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[
            {"version_id": "v-1", "hash": "h1", "entity_types": [], "created_at": "t", "event_count": 0},
        ])
        registry = OntologyRegistry(neo4j, tenant="attacker")
        result = await registry.get_schema_history("attacker")

        # The mock always returns the same row regardless of query text, so
        # this test's real assertion is on the query sent, not the result --
        # covered by test_query_filters_by_tenant above. This test documents
        # the call contract an integration/live test would also need to hold.
        assert result[0]["version_id"] == "v-1"
        assert neo4j.run.await_args.kwargs["tenant"] == "attacker"
