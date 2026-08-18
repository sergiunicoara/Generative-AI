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


class TestCitationRecallCanonicalMatching:
    """Regression tests for the citation naming-system mismatch found
    2026-08-17 (see docs/audit-2026-08-13.md).

    The pipeline returns a document under two different names depending on how
    it was reached: chunk-derived citations use the source filename stem
    ("FAA-AD-2024-01-02"), entity-derived ones use the surface form the corpus
    text writes ("AD 2024-01-02"). Under the original raw-substring check the
    second form could never satisfy an expected citation written in the first,
    so a transitively-referenced document was structurally unable to pass
    citation recall regardless of retrieval or synthesis quality.
    """

    def test_entity_form_citation_satisfies_filename_form_expectation(self):
        # The exact AUT-01 case: only the bridging chunk was retrieved, so the
        # AD is cited under its in-text surface form, not its filename stem.
        spec = {"expected_citations": ["FAA-AD-2024-01-02"], "min_citation_recall": 1.0}
        passed, failures = _check(
            spec,
            {"answer": "x", "citations": ["AD 2024-01-02", "737MAX_CMM_Engine_Mount"]},
        )
        assert passed, failures

    def test_hyphenated_and_spaced_prefix_forms_both_match(self):
        spec = {"expected_citations": ["FAA-AD-2022-03-07"], "min_citation_recall": 1.0}
        for form in ("FAA AD 2022-03-07", "AD-2022-03-07", "AD 2022-03-07"):
            passed, failures = _check(spec, {"answer": "x", "citations": [form]})
            assert passed, f"{form!r} should satisfy the expectation: {failures}"

    def test_filename_form_still_matches_unchanged(self):
        # The original substring path must keep working exactly as before.
        spec = {"expected_citations": ["FAA-AD-2024-01-02"], "min_citation_recall": 1.0}
        passed, _ = _check(spec, {"answer": "x", "citations": ["FAA-AD-2024-01-02"]})
        assert passed

    def test_non_regulatory_document_names_unaffected(self):
        # Prefix stripping must not change how ordinary document names match.
        spec = {"expected_citations": ["SWA_fleet_registry_2024"], "min_citation_recall": 1.0}
        passed, _ = _check(spec, {"answer": "x", "citations": ["SWA_fleet_registry_2024"]})
        assert passed

    def test_genuinely_absent_citation_still_fails(self):
        # The check must not become a rubber stamp: an unrelated citation set
        # must still fail, otherwise this "fix" would mask real recall gaps.
        spec = {"expected_citations": ["FAA-AD-2024-01-02"], "min_citation_recall": 1.0}
        passed, failures = _check(
            spec, {"answer": "x", "citations": ["Boeing_company_profile", "AD 2020-05-11"]}
        )
        assert not passed
        assert "citation recall" in failures[0]
