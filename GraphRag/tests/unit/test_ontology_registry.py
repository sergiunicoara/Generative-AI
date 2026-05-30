"""Unit tests for OntologyRegistry — domain/range enforcement, migration, domain rules."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag.graph.ontology_registry import OntologyRegistry, _RELATION_RULES


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
