"""Subgroup breakdown with Wilson intervals and conservative flagging."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from graphrag.evaluation.subgroups import subgroup_report, wilson_interval

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import eval_subgroups as script  # noqa: E402


def _items(group, passed, failed):
    return [{"g": group, "passed": True}] * passed + [{"g": group, "passed": False}] * failed


def test_wilson_interval_known_values_and_bounds():
    assert wilson_interval(0, 0) is None
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-3) and high == pytest.approx(0.9433, abs=1e-3)
    assert wilson_interval(0, 5)[0] == 0.0 and wilson_interval(5, 5)[1] == 1.0
    small, large = wilson_interval(8, 10), wilson_interval(80, 100)
    assert (small[1] - small[0]) > (large[1] - large[0])


def test_flags_only_big_enough_groups_clearly_below_overall():
    items = _items("easy", 9, 1) + _items("hard", 3, 3) + _items("tiny", 0, 2)
    report = subgroup_report(items, group_key="g")
    rows = {r["group"]: r for r in report["groups"]}
    assert report["overall_rate"] == pytest.approx(12 / 18, abs=1e-4)
    assert rows["hard"]["below_overall"] is True
    assert rows["tiny"]["insufficient_n"] is True and rows["tiny"]["below_overall"] is False
    assert rows["easy"]["below_overall"] is False
    assert report["flagged"] == ["hard"]


def test_gap_threshold_is_inclusive_and_configurable():
    items = _items("a", 8, 2) + _items("b", 6, 4)       # overall 0.7, b is 0.1 below
    assert subgroup_report(items, group_key="g", min_gap=0.1)["flagged"] == ["b"]
    assert subgroup_report(items, group_key="g", min_gap=0.11)["flagged"] == []


def test_empty_and_missing_keys():
    empty = subgroup_report([], group_key="g")
    assert empty["overall_rate"] is None and empty["groups"] == [] and empty["max_gap"] is None
    report = subgroup_report([{"passed": True}, {"g": "", "passed": False}], group_key="g", min_n=1)
    assert [r["group"] for r in report["groups"]] == ["(none)"] and report["groups"][0]["n"] == 2


def test_custom_passed_key():
    report = subgroup_report([{"g": "a", "correct": True}, {"g": "a", "correct": False}], group_key="g",
                             passed_key="correct", min_n=1)
    assert report["overall_rate"] == 0.5


def test_script_builds_from_recorded_results_without_network():
    report = script.build()
    assert report["golden_set_by_type"]["n"] > 0
    assert report["routing"]["by_expected_class"]["n"] > 0
    assert report["entity_resolution_by_category"]["n"] > 0
