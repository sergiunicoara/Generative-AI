"""Entity-resolution evaluator: error taxonomy, per-category counts, and the labelled case file."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from graphrag.evaluation.entity_resolution import ERROR_KINDS, EntityResolutionCase, evaluate_entity_resolution
from graphrag.graph.alias_registry import AmbiguousMatch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import eval_entity_resolution as script  # noqa: E402

A = ("Alpha", "ORG")
B = ("Beta", "ORG")
AMBIGUOUS = AmbiguousMatch(A, 0.8, "fuzzy")
ALPHA = {"cn": "Alpha", "ct": "ORG"}


def _run(expected, outcome, **kw):
    case = EntityResolutionCase("x", "ORG", expected, kw.get("cn", ""), kw.get("ct", ""), kw.get("cat", "c"))
    return evaluate_entity_resolution([case], lambda _n: outcome)["results"][0]["error"]


@pytest.mark.parametrize("expected,outcome,kw,error", [
    ("matched", A, ALPHA, None),
    ("matched", B, ALPHA, "wrong_link"),
    ("matched", None, ALPHA, "missed_match"),
    ("matched", AMBIGUOUS, ALPHA, "over_quarantine"),
    ("new", None, {}, None),
    ("new", A, {}, "over_merge"),
    ("new", AMBIGUOUS, {}, "over_quarantine"),
    ("quarantined", AMBIGUOUS, {}, None),
    ("quarantined", A, {}, "missed_review"),
    ("quarantined", None, {}, "missed_review"),
])
def test_error_kind_per_disposition(expected, outcome, kw, error):
    assert _run(expected, outcome, **kw) == error


def test_report_counts_errors_categories_and_f1_and_keeps_original_keys():
    cases = [EntityResolutionCase("a", "ORG", "matched", "Alpha", "ORG", "x"),
             EntityResolutionCase("b", "ORG", "new", category="y"),
             EntityResolutionCase("c", "ORG", "matched", "Alpha", "ORG", "x")]
    outcomes = {"a": A, "b": A, "c": None}
    report = evaluate_entity_resolution(cases, outcomes.get)
    assert set(report["errors"]) == set(ERROR_KINDS)
    assert report["errors"]["over_merge"] == 1 and report["errors"]["missed_match"] == 1
    assert report["by_category"] == {"x": {"correct": 1, "total": 2}, "y": {"correct": 0, "total": 1}}
    assert report["auto_match_precision"] == 0.5 and report["auto_match_recall"] == 0.5
    assert report["auto_match_f1"] == 0.5
    assert {"total", "correct", "accuracy", "dispositions", "results"} <= set(report)


def test_labelled_case_file_is_well_formed():
    canonical, cases = script.load_cases()
    names = set(canonical)
    assert len(names) == len(canonical) >= 5 and len(cases) >= 20
    assert {c.expected for c in cases} == {"matched", "quarantined", "new"}
    for c in cases:
        if c.expected == "matched":
            assert (c.canonical_name, c.canonical_type) in names, c.raw_name
        else:
            assert not c.canonical_name
    assert len({c.raw_name for c in cases}) == len(cases)


def test_script_run_is_deterministic_and_reports_unmeasured_tiers():
    first, second = script.run(), script.run()
    assert first == second
    assert "embedding" in first["tiers_not_measured"]
    assert first["total"] == len(first["results"])


def test_exact_and_case_variants_always_resolve_correctly():
    """Floor, not a score: the simplest tiers must never regress. Messier categories are recorded, not asserted."""
    for r in script.run()["results"]:
        if r["category"] in {"exact", "case"}:
            assert r["correct"], r
