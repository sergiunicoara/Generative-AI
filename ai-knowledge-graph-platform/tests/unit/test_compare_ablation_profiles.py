"""Paired statistics of scripts/compare_ablation_profiles.py (synthetic result files, no model, no database)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import compare_ablation_profiles as cmp  # noqa: E402


def _row(qid, qtype, passed, faith=None, status=None):
    return {"id": qid, "type": qtype, "golden_pass": passed,
            "status": status or ("scored" if faith is not None else "refused"), "faithfulness": faith}


def _write(directory: Path, profile: str, stamp: str, rows, *, n_errors=0):
    report = {"run": {"retrieval_profile": profile}, "summary": {"n_errors": n_errors}, "results": rows}
    (directory / f"aerospace__{profile}__{stamp}.json").write_text(json.dumps(report), encoding="utf-8")


def test_mcnemar_exact_matches_hand_computed_values():
    assert cmp.mcnemar_exact(0, 0) == 1.0
    assert cmp.mcnemar_exact(5, 0) == pytest.approx(2 * 0.5 ** 5)                  # 0.0625
    assert cmp.mcnemar_exact(8, 1) == pytest.approx(2 * (1 + 9) / 512)             # 0.0390625
    assert cmp.mcnemar_exact(4, 4) == 1.0                                          # capped at 1
    assert cmp.mcnemar_exact(1, 8) == cmp.mcnemar_exact(8, 1)                      # symmetric


def test_holm_is_monotone_and_never_below_the_raw_p():
    adjusted = cmp.holm([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert all(a >= p for a, p in zip(adjusted, [0.01, 0.04, 0.03]))


def test_bootstrap_ci_excludes_zero_for_a_consistent_gain_and_includes_it_for_noise():
    lo, hi = cmp.paired_bootstrap_ci([0.1] * 20)
    assert lo == pytest.approx(0.1) and hi == pytest.approx(0.1)
    lo, hi = cmp.paired_bootstrap_ci([0.2, -0.2] * 10)
    assert lo < 0 < hi
    assert cmp.paired_bootstrap_ci([]) is None
    assert cmp.paired_bootstrap_ci([0.1, 0.3, -0.1], seed=1) == cmp.paired_bootstrap_ci([0.1, 0.3, -0.1], seed=1)


def test_a_question_passes_when_it_passes_in_most_runs_and_faithfulness_averages_scored_runs(tmp_path):
    _write(tmp_path, "full", "1", [_row("Q1", "single_hop", True, 0.8), _row("Q2", "multi_hop", False, None)])
    _write(tmp_path, "full", "2", [_row("Q1", "single_hop", False, 0.6), _row("Q2", "multi_hop", False, None)])
    _write(tmp_path, "full", "3", [_row("Q1", "single_hop", True, None), _row("Q2", "multi_hop", False, None)])
    data = cmp.per_question(cmp.load_runs(tmp_path, "full"))
    assert data["Q1"]["passed"] is True and data["Q1"]["faithfulness"] == pytest.approx(0.7)
    assert data["Q2"]["passed"] is False and data["Q2"]["faithfulness"] is None


def test_runs_with_errors_and_the_invalid_folder_are_ignored(tmp_path):
    _write(tmp_path, "full", "1", [_row("Q1", "single_hop", True, 0.9)])
    _write(tmp_path, "full", "2", [_row("Q1", "single_hop", False, 0.1)], n_errors=3)
    (tmp_path / "invalid").mkdir()
    _write(tmp_path / "invalid", "full", "3", [_row("Q1", "single_hop", False, 0.0)])
    assert len(cmp.load_runs(tmp_path, "full")) == 1


def test_compare_counts_only_one_sided_passes_and_pairs_faithfulness_on_questions_scored_in_both():
    base = {f"Q{i}": {"type": "single_hop", "passed": True, "faithfulness": 0.9, "n_runs": 1} for i in range(8)}
    other = {f"Q{i}": {"type": "single_hop", "passed": False, "faithfulness": 0.7, "n_runs": 1} for i in range(8)}
    base["Q8"] = {"type": "multi_hop", "passed": False, "faithfulness": None, "n_runs": 1}
    other["Q8"] = {"type": "multi_hop", "passed": True, "faithfulness": 0.5, "n_runs": 1}
    result = cmp.compare(base, other)
    assert (result["only_base_passed"], result["only_other_passed"]) == (8, 1)
    assert result["mcnemar_p"] == pytest.approx(0.0390625)
    assert result["n_faithfulness_pairs"] == 8                      # Q8 is unscored on one side
    assert result["faithfulness_mean_diff"] == pytest.approx(0.2)
    assert result["faithfulness_distinguishable"] is True
    multi = cmp.compare(base, other, only_type="multi_hop")
    assert multi["n_questions"] == 1 and multi["only_other_ids"] == ["Q8"] and multi["faithfulness_ci95"] is None


def test_analyse_and_render_end_to_end(tmp_path):
    ids = [f"Q{i}" for i in range(10)]
    _write(tmp_path, "full", "1", [_row(q, "multi_hop" if i < 3 else "single_hop", i != 9, 0.9) for i, q in enumerate(ids)])
    _write(tmp_path, "vector_only", "1", [_row(q, "multi_hop" if i < 3 else "single_hop", i > 4, 0.8) for i, q in enumerate(ids)])
    report = cmp.analyse(tmp_path, "full", ["vector_only"])
    table = cmp.render(report)
    assert report["comparisons"]["vector_only"]["all"]["only_base_passed"] == 5
    assert "| vector_only | all | 10 |" in table and "| vector_only | multi_hop | 3 |" in table
    assert "Holm" in table


def test_a_profile_without_valid_runs_is_reported_not_silently_skipped(tmp_path):
    _write(tmp_path, "full", "1", [_row("Q1", "single_hop", True, 0.9)])
    with pytest.raises(SystemExit, match="vector_only"):
        cmp.analyse(tmp_path, "full", ["vector_only"])


def test_citation_recall_is_compared_per_question_with_a_paired_bootstrap(tmp_path):
    def row(qid, answer_recall, context_recall):
        entry = _row(qid, "single_hop", True, 0.9)
        entry["citation_metrics"] = {"answer": {"recall_citable": answer_recall},
                                     "context": {"recall_raw": context_recall}}
        return entry

    ids = [f"Q{i}" for i in range(12)]
    _write(tmp_path, "full", "1", [row(q, 1.0, 1.0) for q in ids])
    _write(tmp_path, "vector_only", "1", [row(q, 0.5, 0.5) for q in ids])
    _write(tmp_path, "old", "1", [_row(q, "single_hop", True, 0.9) for q in ids])       # no citation_metrics
    report = cmp.analyse(tmp_path, "full", ["vector_only", "old"])
    answer = report["comparisons"]["vector_only"]["all"]["recall_answer"]
    assert answer["n_pairs"] == 12 and answer["mean_diff"] == pytest.approx(0.5) and answer["distinguishable"] is True
    assert report["comparisons"]["old"]["all"]["recall_answer"]["n_pairs"] == 0      # old files are skipped, not zeroed
    assert "Paired bootstrap of per-question citation recall" in cmp.render(report)
