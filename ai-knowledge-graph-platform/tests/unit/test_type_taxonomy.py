"""Unit tests for TypeTaxonomy — hierarchy, subtype expansion, ancestors, LCA."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graphrag.graph.type_taxonomy import DEFAULT_HIERARCHY, TypeTaxonomy


# ── Fixture ───────────────────────────────────────────────────────────────────

def _make_taxonomy(pairs: list[tuple[str, str]]) -> TypeTaxonomy:
    """Build an in-memory TypeTaxonomy without Neo4j by directly populating dicts."""
    neo4j = AsyncMock()
    t = TypeTaxonomy(neo4j)
    for child, parent in pairs:
        t._children.setdefault(parent, set()).add(child)
        t._parents.setdefault(child, set()).add(parent)
    t._loaded = True
    return t


BASE_PAIRS = list(DEFAULT_HIERARCHY)   # default: PERSON/ORG/PRODUCT/LOCATION/EVENT/CONCEPT
AERO_PAIRS = [
    ("REGULATION",               "CONCEPT"),
    ("AIRWORTHINESS_DIRECTIVE",  "REGULATION"),
    ("SERVICE_BULLETIN",         "REGULATION"),
    ("TYPE_CERTIFICATE",         "REGULATION"),
    ("AIRCRAFT_TYPE",            "PRODUCT"),
    ("AIRCRAFT_COMPONENT",       "PRODUCT"),
    ("REGULATOR",                "ORG"),
]


# ── get_subtypes ──────────────────────────────────────────────────────────────

class TestGetSubtypes:
    def test_direct_subtypes(self):
        t = _make_taxonomy(BASE_PAIRS)
        subtypes = t.get_subtypes("AGENT", transitive=False)
        assert set(subtypes) == {"PERSON", "ORG"}

    def test_transitive_subtypes(self):
        t = _make_taxonomy(BASE_PAIRS + AERO_PAIRS)
        # CONCEPT → REGULATION → AIRWORTHINESS_DIRECTIVE, SERVICE_BULLETIN, TYPE_CERTIFICATE
        all_subs = t.get_subtypes("CONCEPT", transitive=True)
        assert "REGULATION" in all_subs
        assert "AIRWORTHINESS_DIRECTIVE" in all_subs
        assert "SERVICE_BULLETIN" in all_subs

    def test_leaf_node_has_no_subtypes(self):
        t = _make_taxonomy(BASE_PAIRS)
        assert t.get_subtypes("PERSON") == []

    def test_unknown_type_returns_empty(self):
        t = _make_taxonomy(BASE_PAIRS)
        assert t.get_subtypes("NONEXISTENT") == []


# ── get_ancestors ─────────────────────────────────────────────────────────────

class TestGetAncestors:
    def test_direct_parent(self):
        t = _make_taxonomy(BASE_PAIRS)
        assert "AGENT" in t.get_ancestors("PERSON")

    def test_transitive_ancestors(self):
        t = _make_taxonomy(BASE_PAIRS + AERO_PAIRS)
        ancestors = t.get_ancestors("AIRWORTHINESS_DIRECTIVE", transitive=True)
        assert "REGULATION" in ancestors
        assert "CONCEPT" in ancestors

    def test_root_has_no_ancestors(self):
        t = _make_taxonomy(BASE_PAIRS)
        assert t.get_ancestors("AGENT") == []

    def test_unknown_type_returns_empty(self):
        t = _make_taxonomy(BASE_PAIRS)
        assert t.get_ancestors("GHOST") == []


# ── expand_type ───────────────────────────────────────────────────────────────

class TestExpandType:
    def test_expand_includes_self(self):
        t = _make_taxonomy(BASE_PAIRS)
        expanded = t.expand_type("AGENT")
        assert "AGENT" in expanded

    def test_expand_includes_subtypes(self):
        t = _make_taxonomy(BASE_PAIRS)
        expanded = t.expand_type("AGENT")
        assert "PERSON" in expanded
        assert "ORG" in expanded

    def test_expand_leaf_returns_only_self(self):
        t = _make_taxonomy(BASE_PAIRS)
        expanded = t.expand_type("PERSON")
        assert expanded == ["PERSON"]

    def test_expand_transitive(self):
        t = _make_taxonomy(BASE_PAIRS + AERO_PAIRS)
        expanded = t.expand_type("CONCEPT")
        assert "AIRWORTHINESS_DIRECTIVE" in expanded
        assert "REGULATION" in expanded
        assert "CONCEPT" in expanded


# ── least_common_ancestor ─────────────────────────────────────────────────────

class TestLeastCommonAncestor:
    def test_sibling_types_share_parent(self):
        t = _make_taxonomy(BASE_PAIRS)
        lca = t.least_common_ancestor("PERSON", "ORG")
        assert lca == "AGENT"

    def test_same_type_returns_itself(self):
        t = _make_taxonomy(BASE_PAIRS)
        assert t.least_common_ancestor("PERSON", "PERSON") == "PERSON"

    def test_aero_siblings(self):
        t = _make_taxonomy(BASE_PAIRS + AERO_PAIRS)
        lca = t.least_common_ancestor("AIRWORTHINESS_DIRECTIVE", "SERVICE_BULLETIN")
        assert lca == "REGULATION"

    def test_no_common_ancestor_returns_none(self):
        t = _make_taxonomy(BASE_PAIRS)
        # PERSON (under AGENT) and LOCATION (under PLACE) — no common parent
        lca = t.least_common_ancestor("PERSON", "LOCATION")
        assert lca is None


# ── register_subclass ─────────────────────────────────────────────────────────

class TestRegisterSubclass:
    async def test_register_new_subclass(self):
        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[])
        t = _make_taxonomy(BASE_PAIRS)
        t._neo4j = neo4j

        await t.register_subclass("REGULATOR", "ORG")

        assert "REGULATOR" in t._children.get("ORG", set())
        assert "ORG" in t._parents.get("REGULATOR", set())

    async def test_register_normalises_to_upper(self):
        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[])
        t = _make_taxonomy(BASE_PAIRS)
        t._neo4j = neo4j

        await t.register_subclass("regulator", "org")
        assert "REGULATOR" in t._children.get("ORG", set())


# ── domain_ontology integration ───────────────────────────────────────────────

class TestDomainOntologyIntegration:
    def test_get_type_hierarchy_pairs_parses_list_format(self):
        from graphrag.graph.domain_ontology import get_type_hierarchy_pairs
        ontology = {
            "type_hierarchy": [
                ["AIRWORTHINESS_DIRECTIVE", "REGULATION"],
                ["REGULATION", "CONCEPT"],
            ]
        }
        pairs = get_type_hierarchy_pairs(ontology)
        assert ("AIRWORTHINESS_DIRECTIVE", "REGULATION") in pairs
        assert ("REGULATION", "CONCEPT") in pairs

    def test_get_type_hierarchy_pairs_uppercases(self):
        from graphrag.graph.domain_ontology import get_type_hierarchy_pairs
        ontology = {"type_hierarchy": [["airworthiness_directive", "regulation"]]}
        pairs = get_type_hierarchy_pairs(ontology)
        assert ("AIRWORTHINESS_DIRECTIVE", "REGULATION") in pairs

    def test_get_inference_rules_returns_list(self):
        from graphrag.graph.domain_ontology import get_inference_rules
        ontology = {
            "inference_rules": [
                {"name": "test_rule", "relation": "SUPERSEDES",
                 "rule_type": "transitivity", "max_depth": 5,
                 "confidence_decay": 0.95}
            ]
        }
        rules = get_inference_rules(ontology)
        assert len(rules) == 1
        assert rules[0]["name"] == "test_rule"

    def test_build_inference_rules_returns_inference_rule_objects(self):
        from graphrag.graph.domain_ontology import build_inference_rules_from_ontology
        from graphrag.graph.inference_engine import InferenceRule
        ontology = {
            "inference_rules": [
                {"name": "supersedes_transitivity",
                 "relation": "SUPERSEDES",
                 "rule_type": "transitivity",
                 "max_depth": 5,
                 "confidence_decay": 0.95}
            ]
        }
        rules = build_inference_rules_from_ontology(ontology)
        assert len(rules) == 1
        assert isinstance(rules[0], InferenceRule)
        assert rules[0].relation == "SUPERSEDES"
        assert rules[0].max_depth == 5
