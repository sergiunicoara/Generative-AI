"""Unit tests for scripts/run_golden_eval.py's _check() scoring logic."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from run_golden_eval import _check  # noqa: E402


class TestRequiredAnswerTerms:
    def test_all_present_passes(self):
        spec = {"required_answer_terms": ["prevails", "SOW"]}
        passed, failures = _check(spec, {"answer": "The SOW prevails in a conflict."})
        assert passed
        assert failures == []

    def test_missing_term_fails(self):
        spec = {"required_answer_terms": ["prevails"]}
        passed, failures = _check(spec, {"answer": "The SOW takes precedence."})
        assert not passed
        assert "prevails" in failures[0]


class TestRequiredAnswerAnyOf:
    """Any-of groups: at least one alternative phrasing per group must match."""

    def test_first_alternative_matches(self):
        spec = {"required_answer_any_of": [["prohibited", "does not permit", "not allowed"]]}
        passed, _ = _check(spec, {"answer": "This is prohibited under the policy."})
        assert passed

    def test_second_alternative_matches(self):
        spec = {"required_answer_any_of": [["prohibited", "does not permit", "not allowed"]]}
        passed, _ = _check(spec, {"answer": "The policy does not permit this."})
        assert passed

    def test_no_alternative_matches_fails(self):
        spec = {"required_answer_any_of": [["prohibited", "does not permit", "not allowed"]]}
        passed, failures = _check(spec, {"answer": "This is totally fine and permitted."})
        assert not passed
        assert "missing any of required terms" in failures[0]

    def test_multiple_groups_all_must_have_a_match(self):
        spec = {"required_answer_any_of": [
            ["prohibited", "does not permit"],
            ["SOW", "Statement of Work"],
        ]}
        # Satisfies group 1 but not group 2
        passed, failures = _check(spec, {"answer": "This is prohibited."})
        assert not passed
        assert len(failures) == 1
        assert "SOW" in failures[0]

    def test_combines_with_required_answer_terms(self):
        spec = {
            "required_answer_terms": ["gambling"],
            "required_answer_any_of": [["prohibited", "does not permit"]],
        }
        passed, _ = _check(spec, {"answer": "Gambling targeting does not permit this."})
        assert passed

    def test_empty_any_of_is_noop(self):
        spec = {"required_answer_any_of": []}
        passed, failures = _check(spec, {"answer": "anything at all"})
        assert passed
        assert failures == []


class TestForbiddenTerms:
    def test_forbidden_term_present_fails(self):
        spec = {"forbidden_terms": ["unknown"]}
        passed, failures = _check(spec, {"answer": "The value is unknown."})
        assert not passed

    def test_word_boundary_avoids_false_positive_on_inflection(self):
        # "american" is a substring of the Romanian inflection "americană" —
        # the word-boundary check must not flag it as the forbidden term.
        spec = {"forbidden_terms": ["american"]}
        passed, _ = _check(spec, {"answer": "piața nord-americană nu este menționată"})
        assert passed


class TestCitationRecall:
    def test_sufficient_recall_passes(self):
        spec = {"expected_citations": ["doc-a", "doc-b"], "min_citation_recall": 0.5}
        passed, _ = _check(spec, {"answer": "x", "citations": ["doc-a"]})
        assert passed

    def test_insufficient_recall_fails(self):
        spec = {"expected_citations": ["doc-a", "doc-b", "doc-c"], "min_citation_recall": 0.9}
        passed, failures = _check(spec, {"answer": "x", "citations": ["doc-a"]})
        assert not passed
        assert "citation recall" in failures[0]
