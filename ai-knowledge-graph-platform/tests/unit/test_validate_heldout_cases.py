"""Offline structural validation of labelled golden cases against the automotive corpus."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_heldout_cases as v  # noqa: E402

CORPUS = {"doc-a": v._norm("Înregistrările se păstrează 3 ani. Rata țintă 95 %."), "doc-b": v._norm("lunar")}


def test_answerable_case_passes_with_accent_and_percent_spacing_differences():
    case = {"id": "x", "type": "single_hop", "expected_citations": ["doc-a"],
            "required_answer_terms": ["inregistrarile", "95%"], "required_answer_any_of": [["nope", "3 ani"]]}
    assert v.validate([case], CORPUS) == []


def test_unanswerable_terms_unknown_documents_and_duplicates_are_reported():
    cases = [
        {"id": "a", "type": "single_hop", "expected_citations": ["doc-b"], "required_answer_terms": ["3 ani"]},
        {"id": "a", "type": "single_hop", "expected_citations": ["doc-z"], "required_answer_terms": []},
        {"id": "c", "type": "multi_hop", "expected_citations": [], "required_answer_terms": ["x"]},
        {"id": "d", "type": "single_hop", "expected_citations": ["doc-a"], "required_answer_any_of": [["zz", "yy"]]},
    ]
    problems = v.validate(cases, CORPUS)
    assert any("duplicate id a" in p for p in problems)
    assert any("'3 ani' not found" in p for p in problems)
    assert any("not in corpus" in p for p in problems)
    assert any("without citations" in p for p in problems)
    assert any("none of" in p for p in problems)


def test_negative_case_fails_when_its_forbidden_term_exists_in_the_corpus():
    ok = {"id": "n1", "type": "negative", "expected_citations": [], "forbidden_terms": ["euro"]}
    bad = {"id": "n2", "type": "negative", "expected_citations": [], "forbidden_terms": ["lunar"]}
    assert v.validate([ok], CORPUS) == []
    assert v.validate([bad], CORPUS) and "occurs in the corpus" in v.validate([bad], CORPUS)[0]


def test_shipped_heldout_and_original_automotive_sets_are_valid_and_heldout_is_marked_unrun():
    corpus = v.load_corpus_text()
    for name in ("queries_automotive_heldout.json", "queries_automotive.json"):
        data = json.loads((ROOT / "data" / "eval_golden" / name).read_text(encoding="utf-8"))
        assert v.validate(data["questions"], corpus) == [], name
    heldout = json.loads(v.DEFAULT_CASES.read_text(encoding="utf-8"))
    assert heldout["status"] == "unrun" and len(heldout["questions"]) >= 10
    original = json.loads((ROOT / "data" / "eval_golden" / "queries_automotive.json").read_text(encoding="utf-8"))
    assert not {q["id"] for q in heldout["questions"]} & {q["id"] for q in original["questions"]}
