"""Failure classes of scripts/classify_golden_misses.py (synthetic result files, no model, no database)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import classify_golden_misses as cls  # noqa: E402

CIT = "insufficient citation recall — missing: ['FAA-AD-2022-03-07']"
TERM = "answer missing required term: 'supersedes'"


def _row(qid="Q1", passed=False, failures=(), citable=None, status="scored", context_recall=1.0, qtype="single_hop"):
    failures = list(failures)
    return {"id": qid, "type": qtype, "golden_pass": passed, "golden_failures": failures,
            "golden_failures_citable": failures if citable is None else list(citable), "status": status,
            "citation_metrics": {"context": {"recall_raw": context_recall}, "answer": {}}}


def _write(directory: Path, profile: str, stamp: str, rows, *, n_errors=0):
    report = {"run": {"retrieval_profile": profile}, "summary": {"n_errors": n_errors}, "results": rows}
    (directory / f"aerospace__{profile}__{stamp}.json").write_text(json.dumps(report), encoding="utf-8")


def test_a_passing_row_has_no_class():
    assert cls.classify(_row(passed=True)) is None


@pytest.mark.parametrize("row, expected", [
    # only the citation check failed
    (_row(failures=[CIT], citable=[]), "citation_uncitable_artifact"),        # nothing citable was missing
    (_row(failures=[CIT], context_recall=0.5), "retrieval_miss_citation"),    # a citable document never reached the context
    (_row(failures=[CIT], context_recall=1.0), "citation_only"),
    # a content check failed
    (_row(failures=[TERM], context_recall=1.0), "answer_error"),
    (_row(failures=[TERM], context_recall=0.5), "retrieval_miss"),
    (_row(failures=[TERM], status="refused", context_recall=1.0), "refused_with_evidence"),
    (_row(failures=[TERM], status="refused", context_recall=0.0), "refused_no_evidence"),
    (_row(failures=[TERM, CIT], context_recall=0.5), "retrieval_miss"),       # content decides; citation is secondary
    (_row(failures=["answer contains forbidden term: 'x'"], context_recall=1.0), "answer_error"),
])
def test_ordered_rule(row, expected):
    assert cls.classify(row) == expected


def test_full_context_has_no_retrieval_step_so_a_content_failure_is_an_answer_error():
    row = _row(failures=[TERM])
    row["citation_metrics"]["context"] = {"recall_raw": None}                 # full_context: no context set
    assert cls.classify(row) == "answer_error"
    refused = {**row, "status": "refused"}
    assert cls.classify(refused) == "refused_no_evidence"                     # no evidence can be shown without a context set


def test_an_unscorable_judge_status_does_not_hide_the_golden_cause():
    assert cls.classify(_row(failures=[TERM], status="unscorable", context_recall=1.0)) == "answer_error"


def test_classes_are_listed_and_unique():
    assert len(cls.CLASSES) == len(set(cls.CLASSES)) == 7


def test_invalid_runs_and_n_errors_are_ignored_and_counts_and_majority_class_are_computed(tmp_path):
    (tmp_path / "invalid").mkdir()
    rows = [_row("Q1", failures=[TERM]), _row("Q2", passed=True), _row("Q3", failures=[CIT], citable=[])]
    _write(tmp_path, "full", "1", rows)
    _write(tmp_path, "full", "2", [_row("Q1", failures=[TERM], context_recall=0.5), _row("Q2", passed=True),
                                   _row("Q3", passed=True)])
    _write(tmp_path / "invalid", "full", "3", rows)                            # never read: the glob is not recursive
    _write(tmp_path, "full", "4", rows, n_errors=1)                            # not valid
    runs = {"full": cls.load(tmp_path, "full")}
    assert len(runs["full"]) == 2
    result = cls.classify_runs(runs)
    assert result["by_profile"]["full"] == {"answer_error": 1, "citation_uncitable_artifact": 1, "retrieval_miss": 1}
    assert result["by_question"]["full"]["Q3"] == "citation_uncitable_artifact"
    assert result["by_question"]["full"]["Q1"] in {"answer_error", "retrieval_miss"}   # a tie keeps the first seen
    assert result["n_runs"] == {"full": 2}


def test_render_states_the_limitation_and_the_failed_row_totals(tmp_path):
    _write(tmp_path, "full", "1", [_row("Q1", failures=[TERM])])
    text = cls.render(cls.classify_runs({"full": cls.load(tmp_path, "full")}))
    assert "Limitation:" in text and "graph or schema gap" in text and "| answer_error | 1 |" in text


def test_main_writes_json_and_markdown_and_rejects_a_profile_without_runs(tmp_path):
    for profile in cls.PROFILES:
        _write(tmp_path, profile, "1", [_row("Q1", failures=[TERM])])
    out = tmp_path / "out"
    cls.main([str(tmp_path), "--output-dir", str(out)])
    assert json.loads((out / "miss_classes.json").read_text(encoding="utf-8"))["limitation"]
    assert (out / "miss_classes.md").read_text(encoding="utf-8").startswith("# Golden-set miss classes")
    with pytest.raises(SystemExit):
        cls.main([str(tmp_path / "out"), "--output-dir", str(tmp_path / "x")])
