"""Several partial waivers / supersessions that together cover a requirement's
whole bound context (roadmap item, 2026-10-29).

Coverage is judged on *combinations* of context values, not per dimension:
two waivers that each cover one (market, product) pair must not be read as
covering the other two pairs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import permutations

from graphrag.context_graph.knowledge import (
    Applicability, BusinessContext, Constraint, DimensionScope, EvidenceRef, Fact, KnowledgeException,
    KnowledgeItem, Predicate, PublicationStatus, SourceAuthority,
)
from graphrag.context_graph.knowledge_evaluation import KnowledgeReleaseRef, evaluate_knowledge
from graphrag.enterprise.models import AccessContext

NOW = datetime(2026, 8, 1, tzinfo=timezone.utc)
REF = KnowledgeReleaseRef(release_id="union", domain="union", version=0)
DIMS = ("customer", "brand", "market", "product", "product_family", "workflow")
EV = (EvidenceRef(source_id="EV-1"),)
USER = AccessContext.from_claims({"sub": "u"})


def scope(**restricted) -> Applicability:
    values = {d: DimensionScope.unrestricted() for d in DIMS}
    for dim, vals in restricted.items():
        values[dim] = DimensionScope(mode="in", values=vals)
    return Applicability(**values)


def item(item_id: str, *, applicability=None, **extra) -> KnowledgeItem:
    return KnowledgeItem.model_validate(dict(
        id=item_id, version=1, tenant="t", title=item_id, statement=item_id,
        subject_key=extra.pop("subject_key", f"s.{item_id.lower()}"), modality="mandatory",
        authority=SourceAuthority.REGULATORY, status=PublicationStatus.PUBLISHED,
        applicability=applicability or scope(), recorded_at=NOW,
        constraint=Constraint(all_of=(Predicate(fact="f.x", operator="eq", value=True),)),
        constraint_provenance={"authored_by": "user:c", "approval_ref": "A"},
        sources=(EvidenceRef(source_id=f"SRC-{item_id}"),), reviewed_by="user:c", review_ref="R", **extra,
    ))


def waiver(exc_id: str, targets, **restricted) -> KnowledgeException:
    return KnowledgeException.model_validate(dict(
        id=exc_id, version=1, tenant="t", kind="waiver", target_item_ids=targets, rationale="union",
        applicability=scope(**restricted), granted_by="user:g", approval_ref="AP",
        sources=(EvidenceRef(source_id=f"SRC-{exc_id}"),), status="published", recorded_at=NOW,
    ))


def evaluate(items, exceptions=(), **context):
    return evaluate_knowledge(
        items=items, exceptions=exceptions, context=BusinessContext(as_of=NOW, **context),
        facts={"f.x": Fact(value=False, evidence=EV)}, tenant="t", access=USER, domain="union", release=REF)


def result(decision, item_id):
    return next(r for r in decision.results if r.item_id == item_id)


class TestWaiverUnion:
    def test_two_partial_waivers_together_waive_the_requirement(self):
        decision = evaluate([item("M")], [waiver("WA", ["M"], market=["A"]), waiver("WB", ["M"], market=["B"])],
                            market=["A", "B"])
        waived = result(decision, "M")
        assert waived.outcome_reason == "waived_by_exception"
        assert set(waived.exception_refs) == {"WA@v1", "WB@v1"} and waived.partial_exception_refs == ()
        assert decision.overall.value == "compliant"
        assert {e.exception_key for e in decision.exceptions_applied} == {"WA@v1", "WB@v1"}

    def test_union_leaving_one_value_uncovered_keeps_the_requirement(self):
        decision = evaluate([item("M")], [waiver("WA", ["M"], market=["A"]), waiver("WB", ["M"], market=["B"])],
                            market=["A", "B", "C"])
        kept = result(decision, "M")
        assert kept.outcome.value == "fail" and set(kept.partial_exception_refs) == {"WA@v1", "WB@v1"}
        assert decision.overall.value == "non_compliant"

    def test_coverage_is_per_combination_not_per_dimension(self):
        # A+P1 and B+P2 are waived; A+P2 and B+P1 are not, although each
        # dimension's values are all mentioned by some waiver.
        waivers = [waiver("W1", ["M"], market=["A"], product=["P1"]),
                   waiver("W2", ["M"], market=["B"], product=["P2"])]
        decision = evaluate([item("M")], waivers, market=["A", "B"], product=["P1", "P2"])
        assert result(decision, "M").outcome.value == "fail"
        full = waivers + [waiver("W3", ["M"], market=["A"], product=["P2"]),
                          waiver("W4", ["M"], market=["B"], product=["P1"])]
        decision = evaluate([item("M")], full, market=["A", "B"], product=["P1", "P2"])
        assert result(decision, "M").outcome_reason == "waived_by_exception"

    def test_result_does_not_depend_on_exception_order(self):
        waivers = [waiver("WA", ["M"], market=["A"]), waiver("WB", ["M"], market=["B"]),
                   waiver("WC", ["M"], market=["C"])]
        seen = {evaluate([item("M")], list(order), market=["A", "B", "C"]).model_dump_json(exclude={"decision_id"})
                for order in permutations(waivers)}
        assert len(seen) == 1


class TestSupersessionUnion:
    def _revisions(self):
        old = item("OLD", series="s.rev", subject_key="s.rev")
        new_a = item("NEWA", subject_key="s.a", supersedes=("OLD",), applicability=scope(market=["A"]))
        new_b = item("NEWB", subject_key="s.b", supersedes=("OLD",), applicability=scope(market=["B"]))
        return old, new_a, new_b

    def test_two_partial_revisions_together_supersede_the_old_one(self):
        old, new_a, new_b = self._revisions()
        decision = evaluate([old, new_a, new_b], market=["A", "B"])
        assert decision.applicable == ("NEWA@v1", "NEWB@v1")
        assert result(decision, "OLD").applicability.value == "does_not_apply"
        assert result(decision, "OLD").applicability_reasons[0].code == "superseded"

    def test_union_that_leaves_a_market_uncovered_reports_the_remaining_context(self):
        old, new_a, new_b = self._revisions()
        decision = evaluate([old, new_a, new_b], market=["A", "B", "C"])
        remaining = result(decision, "OLD")
        assert remaining.applicability.value == "applies"
        assert remaining.matched_context["market"] == ("C",)
        assert "partially_superseded" in {r.code for r in remaining.applicability_reasons}

    def test_remaining_context_is_what_later_exceptions_must_cover(self):
        old = item("OLD", series="s.rev", subject_key="s.rev")
        new_a = item("NEWA", subject_key="s.a", supersedes=("OLD",), applicability=scope(market=["A"]))
        decision = evaluate([old, new_a], [waiver("W", ["OLD"], market=["B"])], market=["A", "B"])
        assert result(decision, "OLD").outcome_reason == "waived_by_exception", \
            "market A is superseded, so a waiver for the remaining market B covers the requirement"

    def test_supersession_union_is_independent_of_input_order(self):
        old, new_a, new_b = self._revisions()
        seen = {evaluate(list(order), market=["A", "B"]).model_dump_json(exclude={"decision_id"})
                for order in permutations([old, new_a, new_b])}
        assert len(seen) == 1
