"""Unit tests for domain_ontology.get_entity_types_for_tenant.

Covers the entity-type-collision fix: extraction should be offered
domain-specific types (e.g. AIRWORTHINESS_DIRECTIVE for aerospace) on top of
the generic base list, so same-name-different-meaning entities are more
likely to land on distinct (name, type) MERGE keys instead of colliding.
"""

from __future__ import annotations

from graphrag.graph.domain_ontology import get_entity_types_for_tenant

_BASE_TYPES = ["PERSON", "ORG", "PRODUCT", "CONCEPT", "LOCATION", "EVENT"]


class TestGetEntityTypesForTenant:
    def test_unknown_tenant_returns_base_types_unchanged(self):
        """Tenant with no config/ontologies/{tenant}*.yml file — pure fallback."""
        result = get_entity_types_for_tenant("no_such_tenant_xyz", _BASE_TYPES)
        assert result == _BASE_TYPES

    def test_aerospace_tenant_adds_domain_specific_types(self):
        """Real ontology file — domain child types are appended to the base list."""
        result = get_entity_types_for_tenant("aerospace", _BASE_TYPES)

        # Base types are preserved, in order, at the front.
        assert result[: len(_BASE_TYPES)] == _BASE_TYPES

        # Domain-specific types from aerospace_regulatory.yml's type_hierarchy
        # are present — these are what let "AD-2024-01-02" extract as
        # AIRWORTHINESS_DIRECTIVE rather than the generic CONCEPT/REGULATION,
        # and are the actual disambiguation mechanism for name collisions.
        assert "AIRWORTHINESS_DIRECTIVE" in result
        assert "REGULATOR" in result

    def test_no_duplicate_types(self):
        result = get_entity_types_for_tenant("aerospace", _BASE_TYPES)
        assert len(result) == len(set(result))

    def test_domain_types_are_sorted_for_deterministic_prompt(self):
        """Order must be stable across calls — extraction is cache-keyed on
        the full prompt text, so a shuffled type list would break cache hits."""
        result_a = get_entity_types_for_tenant("aerospace", _BASE_TYPES)
        result_b = get_entity_types_for_tenant("aerospace", _BASE_TYPES)
        assert result_a == result_b

        domain_part = result_a[len(_BASE_TYPES):]
        assert domain_part == sorted(domain_part)
