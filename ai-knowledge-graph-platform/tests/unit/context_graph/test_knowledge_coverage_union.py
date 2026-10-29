"""Joint coverage: several partial waivers or partial supersessions that
together cover a requirement's whole bound context.

Coverage is over combinations of context values, not per dimension: two
exceptions for (A, P1) and (B, P2) leave (A, P2) and (B, P1) uncovered, so the
requirement stays in force.  Remaining in-force context is reported on the
result.
"""

from __future__ import annotations

from itertools import permutations

from graphrag.context_graph.knowledge import Fact

from .test_contextual_knowledge_audit import EV, evaluate, exception, item, outcome, scope

TRUE = {"f.x": Fact(value=True, evidence=EV)}


class TestJointWaivers:
    def test_two_partial_waivers_that_cover_every_market_waive_the_requirement(self):
        decision = evaluate([item("M")], [
            exception("WA", "waiver", ["M"], applicability=scope(market=["A"])),
            exception("WB", "waiver", ["M"], applicability=scope(market=["B"])),
        ], market=["A", "B"])
        result = outcome(decision, "M")
        assert result.outcome_reason == "waived_by_exception"
        assert result.exception_refs == ("WA@v1", "WB@v1")
        assert result.partial_exception_refs == ()
        assert decision.overall.value == "compliant"
        assert {a.exception_key for a in decision.exceptions_applied} == {"WA@v1", "WB@v1"}

    def test_partial_waivers_that_leave_a_market_uncovered_keep_the_requirement(self):
        decision = evaluate([item("M")], [
            exception("WA", "waiver", ["M"], applicability=scope(market=["A"])),
            exception("WB", "waiver", ["M"], applicability=scope(market=["B"])),
        ], market=["A", "B", "C"])
        result = outcome(decision, "M")
        assert result.outcome.value == "fail"
        assert result.partial_exception_refs == ("WA@v1", "WB@v1")
        assert result.in_force_context == ({"market": ("C",)},)
        assert decision.exceptions_applied == ()
        assert decision.overall.value == "non_compliant"

    def test_diagonal_waivers_do_not_cover_the_cross_combinations(self):
        decision = evaluate([item("M")], [
            exception("W1", "waiver", ["M"], applicability=scope(market=["A"], product=["P1"])),
            exception("W2", "waiver", ["M"], applicability=scope(market=["B"], product=["P2"])),
        ], market=["A", "B"], product=["P1", "P2"])
        result = outcome(decision, "M")
        assert result.outcome.value == "fail", "(A, P2) and (B, P1) are not waived"
        assert set(result.partial_exception_refs) == {"W1@v1", "W2@v1"}
        assert result.in_force_context == (
            {"market": ("A",), "product": ("P2",)}, {"market": ("B",), "product": ("P1",)},
        )

    def test_overlapping_waivers_covering_every_combination_waive(self):
        decision = evaluate([item("M")], [
            exception("W1", "waiver", ["M"], applicability=scope(market=["A"])),
            exception("W2", "waiver", ["M"], applicability=scope(product=["P2"])),
            exception("W3", "waiver", ["M"], applicability=scope(market=["B"], product=["P1"])),
        ], market=["A", "B"], product=["P1", "P2"])
        assert outcome(decision, "M").outcome_reason == "waived_by_exception"

    def test_joint_coverage_is_independent_of_exception_order(self):
        waivers = [
            exception("W1", "waiver", ["M"], applicability=scope(market=["A"])),
            exception("W2", "waiver", ["M"], applicability=scope(product=["P2"])),
            exception("W3", "waiver", ["M"], applicability=scope(market=["B"], product=["P1"])),
        ]
        seen = set()
        for order in permutations(waivers):
            decision = evaluate([item("M")], list(order), market=["A", "B"], product=["P1", "P2"])
            seen.add((outcome(decision, "M").outcome_reason, decision.decision_id))
        assert len(seen) == 1

    def test_unrelated_applying_waiver_is_not_reported_as_partial(self):
        decision = evaluate([item("M", applicability=scope(market=["B"]))],
                            [exception("WA", "waiver", ["M"], applicability=scope(market=["A"]))],
                            market=["A", "B"])
        result = outcome(decision, "M")
        assert result.outcome.value == "fail"
        assert result.partial_exception_refs == ()
        assert result.in_force_context == ()

    def test_unknown_waiver_scope_still_escalates_even_beside_covering_waivers(self):
        decision = evaluate([item("M")], [
            exception("WA", "waiver", ["M"], applicability=scope(market=["A"])),
            exception("WB", "waiver", ["M"], applicability=scope(market=["B"])),
            exception("WX", "waiver", ["M"], applicability=scope(brand=["X"])),  # brand not stated
        ], market=["A", "B"])
        assert outcome(decision, "M").outcome_reason == "waived_by_exception"
        assert decision.human_review_required
        assert any("WX@v1" in reason for reason in decision.human_review_reasons)


class TestJointSupersession:
    def _old(self):
        return item("OLD", series="s.rev", subject_key="s.rev")

    def _rev(self, rev_id, **restricted):
        # regional revisions: no shared series, so two of them in force for
        # different markets are not an ambiguous revision of each other
        return item(rev_id, supersedes=("OLD",), applicability=scope(**restricted))

    def test_two_partial_revisions_that_cover_every_market_supersede(self):
        decision = evaluate([self._old(), self._rev("NA", market=["A"]), self._rev("NB", market=["B"])],
                            facts=TRUE, market=["A", "B"])
        old = outcome(decision, "OLD")
        assert old.applicability.value == "does_not_apply"
        assert old.applicability_reasons[0].code == "superseded"
        assert "NA@v1" in old.applicability_reasons[0].detail and "NB@v1" in old.applicability_reasons[0].detail
        assert set(decision.applicable) == {"NA@v1", "NB@v1"}

    def test_partially_superseded_revision_reports_the_remaining_context(self):
        decision = evaluate([self._old(), self._rev("NA", market=["A"])],
                            facts=TRUE, market=["A", "B", "C"])
        old = outcome(decision, "OLD")
        assert old.applicability.value == "applies"
        assert old.in_force_context == ({"market": ("B", "C")},)
        assert old.matched_context == {}  # own scope is unrestricted; the remainder is reported separately
        assert any(r.code == "partially_superseded" for r in old.applicability_reasons)

    def test_partial_supersession_and_a_waiver_on_the_remainder_together_retire_the_item(self):
        decision = evaluate(
            [self._old(), self._rev("NA", market=["A"])],
            [exception("WB", "waiver", ["OLD"], applicability=scope(market=["B"]))],
            facts={"f.x": Fact(value=False, evidence=EV)}, market=["A", "B"],
        )
        old = outcome(decision, "OLD")
        assert old.outcome_reason == "waived_by_exception"
        assert old.exception_refs == ("WB@v1",)
        # the revision still binds market A and is evaluated there
        assert outcome(decision, "NA").outcome.value == "fail"

    def test_an_unknown_revision_still_makes_an_uncovered_item_unknown(self):
        decision = evaluate([self._old(), self._rev("NA", market=["A"]), self._rev("NX", brand=["X"])],
                            facts=TRUE, market=["A", "B"])
        assert "OLD@v1" in decision.unknown_applicability

    def test_jointly_covered_item_is_superseded_even_beside_an_unknown_revision(self):
        decision = evaluate(
            [self._old(), self._rev("NA", market=["A"]), self._rev("NB", market=["B"]),
             self._rev("NX", brand=["X"])],
            facts=TRUE, market=["A", "B"],
        )
        assert outcome(decision, "OLD").applicability.value == "does_not_apply"
