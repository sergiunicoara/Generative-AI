"""Lifecycle and activation gates for YAML ontology definitions."""

from pathlib import Path

import pytest

from graphrag.graph.domain_ontology import (
    OntologyValidationError,
    assert_valid_ontology,
    validate_ontology_yaml,
)

_ONTOLOGIES_DIR = Path(__file__).resolve().parents[2] / "config" / "ontologies"


def _ontology(type_hierarchy=None, **metadata):
    return {
        "ontology": {
            "id": "demo",
            "version": "1.0.0",
            "status": "active",
            "compatible_with": ">=1.0.0",
            "deprecated_types": [],
            "deprecated_relations": [],
            **metadata,
        },
        "type_hierarchy": type_hierarchy or [["WIDGET", "CONCEPT"]],
        "relation_rules": {
            "USES": {"domain": ["ORG"], "target": ["WIDGET"]},
        },
        "inference_rules": [],
    }


def test_all_shipped_ontologies_pass_lifecycle_gate():
    """Globs config/ontologies/*.yml rather than a hand-maintained list — the
    hardcoded list had silently omitted sustainability_supply_chain.yml, so a
    newly added ontology was never actually gated by CI (F9,
    docs/context_graph_gap_plan.md). A glob can't go stale the same way."""
    files = sorted(_ONTOLOGIES_DIR.glob("*.yml"))
    assert files, "found no ontology files — the scan is broken, not the fixtures"
    for path in files:
        report = validate_ontology_yaml(str(path))
        assert report["valid"] is True, f"{path.name}: {report}"


def test_invalid_hierarchy_cycle_is_rejected():
    ontology = _ontology(type_hierarchy=[["A", "B"], ["B", "A"]])
    with pytest.raises(OntologyValidationError, match="cycle"):
        assert_valid_ontology(ontology)


def test_deprecated_relation_requires_migration():
    ontology = _ontology(deprecated_relations=["OLD_USES"])
    with pytest.raises(OntologyValidationError, match="migration_map"):
        assert_valid_ontology(ontology)


def test_major_version_change_is_incompatible():
    current = _ontology()
    current["ontology"]["version"] = "2.0.0"
    with pytest.raises(OntologyValidationError, match="incompatible major"):
        assert_valid_ontology(current, previous=_ontology())
