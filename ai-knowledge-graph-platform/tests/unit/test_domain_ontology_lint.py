"""Unit tests for the ontology-lint additions to graphrag.graph.domain_ontology:
find_orphaned_types, find_duplicate_relations, and their wiring into
validate_ontology_document's warnings.

No YAML files, no Neo4j -- all fixture ontology dicts.
"""
from __future__ import annotations

from graphrag.graph.domain_ontology import (
    find_duplicate_relations,
    find_orphaned_types,
    validate_ontology_document,
)

BASE_META = {
    "id": "test-ontology",
    "version": "1.0.0",
    "status": "active",
    "compatible_with": "1.x",
}


def _ontology(**overrides) -> dict:
    doc = {"ontology": dict(BASE_META), "type_hierarchy": [], "relation_rules": {}}
    doc.update(overrides)
    return doc


class TestFindOrphanedTypes:
    def test_type_used_only_as_domain_is_not_orphaned(self):
        ontology = _ontology(
            type_hierarchy=[["SUPPLIER", "ORG"], ["ORG", "ENTITY"]],
            relation_rules={"SUPPLIES": {"domain": ["SUPPLIER"], "target": ["PRODUCT"]}},
        )
        assert "SUPPLIER" not in find_orphaned_types(ontology)

    def test_type_never_referenced_in_any_relation_is_orphaned(self):
        ontology = _ontology(
            type_hierarchy=[["WIDGET", "PRODUCT"], ["SUPPLIER", "ORG"]],
            relation_rules={
                "SUPPLIES": {"domain": ["SUPPLIER"], "target": ["PRODUCT"]},
                "PARENT_OF": {"domain": ["ORG"], "target": ["ORG"]},
            },
        )
        # WIDGET is declared but never a domain/target anywhere; ORG and the
        # other three types are all used somewhere above.
        assert find_orphaned_types(ontology) == ["WIDGET"]

    def test_pure_taxonomy_parent_can_still_be_orphaned(self):
        ontology = _ontology(
            type_hierarchy=[["SUPPLIER", "ORG"]],
            relation_rules={"SUPPLIES": {"domain": ["SUPPLIER"], "target": ["PRODUCT"]}},
        )
        # ORG is only ever a parent, never used as a relation endpoint.
        assert find_orphaned_types(ontology) == ["ORG"]

    def test_no_type_hierarchy_is_empty(self):
        ontology = _ontology()
        assert find_orphaned_types(ontology) == []

    def test_string_domain_target_are_tolerated_not_just_lists(self):
        ontology = _ontology(
            type_hierarchy=[["SUPPLIER", "ORG"]],
            relation_rules={"SUPPLIES": {"domain": "SUPPLIER", "target": "PRODUCT"}},
        )
        assert "SUPPLIER" not in find_orphaned_types(ontology)


class TestFindDuplicateRelations:
    def test_identical_domain_range_across_files_is_not_a_conflict(self):
        a = _ontology(
            ontology={**BASE_META, "id": "a"},
            relation_rules={"OWNS": {"domain": ["PERSON"], "target": ["ORG"]}},
        )
        b = _ontology(
            ontology={**BASE_META, "id": "b"},
            relation_rules={"OWNS": {"domain": ["PERSON"], "target": ["ORG"]}},
        )
        assert find_duplicate_relations([a, b]) == []

    def test_conflicting_domain_range_across_files_is_flagged(self):
        a = _ontology(
            ontology={**BASE_META, "id": "a"},
            relation_rules={"OWNS": {"domain": ["PERSON"], "target": ["ORG"]}},
        )
        b = _ontology(
            ontology={**BASE_META, "id": "b"},
            relation_rules={"OWNS": {"domain": ["ORG"], "target": ["PRODUCT"]}},
        )
        conflicts = find_duplicate_relations([a, b])
        assert len(conflicts) == 1
        assert "OWNS" in conflicts[0]
        assert "'a'" in conflicts[0] and "'b'" in conflicts[0]

    def test_domain_order_does_not_cause_a_false_conflict(self):
        a = _ontology(relation_rules={"OWNS": {"domain": ["PERSON", "ORG"], "target": ["ORG"]}})
        b = _ontology(relation_rules={"OWNS": {"domain": ["ORG", "PERSON"], "target": ["ORG"]}})
        assert find_duplicate_relations([a, b]) == []

    def test_single_ontology_has_no_cross_file_conflicts(self):
        a = _ontology(relation_rules={"OWNS": {"domain": ["PERSON"], "target": ["ORG"]}})
        assert find_duplicate_relations([a]) == []

    def test_unrelated_relations_across_files_are_not_flagged(self):
        a = _ontology(relation_rules={"OWNS": {"domain": ["PERSON"], "target": ["ORG"]}})
        b = _ontology(relation_rules={"SUPPLIES": {"domain": ["SUPPLIER"], "target": ["PRODUCT"]}})
        assert find_duplicate_relations([a, b]) == []


class TestValidateOntologyDocumentWiresOrphanedTypes:
    def test_orphaned_type_appears_as_a_warning_not_an_error(self):
        ontology = _ontology(
            type_hierarchy=[["WIDGET", "PRODUCT"]],
            relation_rules={},
        )
        report = validate_ontology_document(ontology)
        assert report["valid"] is True  # warnings never fail validation
        assert any("WIDGET" in w for w in report["warnings"])

    def test_no_orphaned_types_produces_no_such_warning(self):
        ontology = _ontology(
            type_hierarchy=[["SUPPLIER", "ORG"]],
            relation_rules={"SUPPLIES": {"domain": ["SUPPLIER"], "target": ["ORG"]}},
        )
        report = validate_ontology_document(ontology)
        assert not any("never used as a relation domain or target" in w for w in report["warnings"])
