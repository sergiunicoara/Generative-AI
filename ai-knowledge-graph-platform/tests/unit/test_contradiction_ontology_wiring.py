"""Unit tests for _ontology_lists() — per-tenant contradiction vocabulary.

Each domain ontology defines `exclusive_state_pairs` / `functional_relations`
annotated with the golden-set contradictions they power (C01-C05 automotive,
T01/T02 telecom, WPP01/WPP02 marketing), and documents itself as "Extends the
default pairs in contradiction_strategies.py". Those lists used to be hardcoded
and the ontology ignored, so a domain's contradiction vocabulary never reached
the detector.
"""

from __future__ import annotations

from unittest.mock import patch

from graphrag.graph.contradiction_strategies import (
    _DEFAULT_EXCLUSIVE_PAIRS,
    _DEFAULT_FUNCTIONAL_RELATIONS,
    _ontology_lists,
)


class TestDefaults:
    def test_no_tenant_returns_defaults_only(self):
        # scan-all-tenants mode can't resolve a single ontology
        pairs, functional = _ontology_lists(None)
        assert pairs == list(_DEFAULT_EXCLUSIVE_PAIRS)
        assert functional == list(_DEFAULT_FUNCTIONAL_RELATIONS)

    def test_defaults_always_present_for_a_tenant(self):
        pairs, functional = _ontology_lists("automotive")
        for default in _DEFAULT_EXCLUSIVE_PAIRS:
            assert default in pairs
        for default in _DEFAULT_FUNCTIONAL_RELATIONS:
            assert default in functional


class TestOntologyExtension:
    """The ontology's pairs must be layered on top of the defaults."""

    def test_ontology_pairs_and_relations_are_added(self):
        fake = {
            "exclusive_state_pairs": [["CI_STATUS_DECOMMISSIONED", "CI_STATUS_ACTIVE"]],
            "functional_relations": ["HAS_STATUS"],
        }
        with (
            patch("graphrag.graph.domain_ontology.get_ontology_path_for_tenant",
                  return_value="fake.yml"),
            patch("graphrag.graph.domain_ontology.load_domain_ontology",
                  return_value=fake),
        ):
            pairs, functional = _ontology_lists("telecom")

        assert ("CI_STATUS_DECOMMISSIONED", "CI_STATUS_ACTIVE") in pairs
        assert "HAS_STATUS" in functional
        # defaults still there
        assert ("IS_ACTIVE", "IS_DEPRECATED") in pairs
        assert "CEO_OF" in functional

    def test_pairs_are_tuples_not_lists(self):
        # YAML gives lists; the query loop unpacks `for rel_a, rel_b in ...`,
        # and downstream membership checks compare against tuples.
        fake = {"exclusive_state_pairs": [["A", "B"]], "functional_relations": []}
        with (
            patch("graphrag.graph.domain_ontology.get_ontology_path_for_tenant",
                  return_value="fake.yml"),
            patch("graphrag.graph.domain_ontology.load_domain_ontology",
                  return_value=fake),
        ):
            pairs, _ = _ontology_lists("x")
        assert ("A", "B") in pairs
        for p in pairs:
            assert isinstance(p, tuple) and len(p) == 2

    def test_duplicate_of_a_default_is_not_added_twice(self):
        fake = {
            "exclusive_state_pairs": [["IS_ACTIVE", "IS_DEPRECATED"]],
            "functional_relations": ["CEO_OF"],
        }
        with (
            patch("graphrag.graph.domain_ontology.get_ontology_path_for_tenant",
                  return_value="fake.yml"),
            patch("graphrag.graph.domain_ontology.load_domain_ontology",
                  return_value=fake),
        ):
            pairs, functional = _ontology_lists("x")
        assert pairs.count(("IS_ACTIVE", "IS_DEPRECATED")) == 1
        assert functional.count("CEO_OF") == 1

    def test_malformed_pair_is_skipped(self):
        fake = {
            "exclusive_state_pairs": [["ONLY_ONE"], ["A", "B", "C"], ["OK_A", "OK_B"]],
            "functional_relations": [],
        }
        with (
            patch("graphrag.graph.domain_ontology.get_ontology_path_for_tenant",
                  return_value="fake.yml"),
            patch("graphrag.graph.domain_ontology.load_domain_ontology",
                  return_value=fake),
        ):
            pairs, _ = _ontology_lists("x")
        assert ("OK_A", "OK_B") in pairs
        assert all(len(p) == 2 for p in pairs)


class TestFailOpen:
    """Contradiction detection is a quality signal, not a gate — a broken
    ontology must never break a scan."""

    def test_ontology_load_exception_falls_back_to_defaults(self):
        with patch("graphrag.graph.domain_ontology.get_ontology_path_for_tenant",
                   side_effect=RuntimeError("ontology exploded")):
            pairs, functional = _ontology_lists("automotive")
        assert pairs == list(_DEFAULT_EXCLUSIVE_PAIRS)
        assert functional == list(_DEFAULT_FUNCTIONAL_RELATIONS)

    def test_no_ontology_path_falls_back_to_defaults(self):
        with patch("graphrag.graph.domain_ontology.get_ontology_path_for_tenant",
                   return_value=None):
            pairs, functional = _ontology_lists("tenant-with-no-ontology")
        assert pairs == list(_DEFAULT_EXCLUSIVE_PAIRS)
        assert functional == list(_DEFAULT_FUNCTIONAL_RELATIONS)


class TestRealOntologiesResolve:
    """Against the actual shipped ontologies — the vocabularies annotated with
    golden-question IDs must now reach the detector."""

    def test_automotive_c01_c03_pairs_present(self):
        pairs, functional = _ontology_lists("automotive")
        assert ("REQUIRES_THREE_OFFERS", "REQUIRES_TWO_OFFERS") in pairs      # C01
        assert ("REEVALUATED_SEMESTRIAL", "REEVALUATED_ANNUAL") in pairs      # C03/C05
        assert "CLASSIFIED_AS" in functional

    def test_aerospace_pairs_present(self):
        pairs, functional = _ontology_lists("aerospace")
        assert ("IS_AIRWORTHY", "IS_UNAIRWORTHY") in pairs
        assert "TYPE_CERTIFICATE_OF" in functional

    def test_marketing_wpp_pairs_present(self):
        pairs, _ = _ontology_lists("marketing")
        assert ("CATEGORY_EXCLUDED", "CATEGORY_LOCALLY_APPROVED") in pairs    # WPP01
        assert ("INFERENCE_PROHIBITED", "INFERENCE_PERMITTED") in pairs       # WPP02
