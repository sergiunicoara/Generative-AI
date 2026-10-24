"""Regression tests for the 2026-10-24 audit of contextual-knowledge evaluation.

Each class pins one reported defect with an observable decision or stored
state.  They were written and run against the pre-fix code first (see
tasks/todo.md for which ones failed there).
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import permutations
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from graphrag.context_graph.knowledge import (
    Applicability, BusinessContext, Constraint, DimensionScope, EvidenceRef, Fact, KnowledgeException,
    KnowledgeItem, NormativeModality, Predicate, PublicationStatus, SourceAuthority,
)
from graphrag.context_graph.knowledge_evaluation import KnowledgeReleaseRef, evaluate_knowledge
from graphrag.context_graph.knowledge_release import (
    KnowledgePublicationError, check_pack, load_cases, load_pack, publish_pack,
)
from graphrag.context_graph.models import ConditionOperator
from graphrag.enterprise.models import AccessContext, DocumentAccessPolicy

ROOT = Path(__file__).resolve().parents[3]
PACK = ROOT / "config" / "knowledge_packs" / "packaging_labeling.yml"
CASES = ROOT / "evals" / "knowledge_eval" / "packaging_labeling_cases.json"
NOW = datetime(2026, 8, 1, tzinfo=timezone.utc)
REF = KnowledgeReleaseRef(release_id="audit", domain="audit", version=0)
DIMS = ("customer", "brand", "market", "product", "product_family", "workflow")
EV = (EvidenceRef(source_id="EV-1"),)
USER = AccessContext.from_claims({"sub": "u"})


def scope(**restricted) -> Applicability:
    values = {d: DimensionScope.unrestricted() for d in DIMS}
    for dim, vals in restricted.items():
        values[dim] = DimensionScope(mode="in", values=vals)
    return Applicability(**values)


def item(item_id: str, *, modality="mandatory", applicability=None, predicate=None, **extra) -> KnowledgeItem:
    predicate = predicate or Predicate(fact="f.x", operator=ConditionOperator.EQ, value=True)
    return KnowledgeItem.model_validate(dict(
        id=item_id, version=extra.pop("version", 1), tenant="t", title=item_id, statement=item_id,
        subject_key=extra.pop("subject_key", f"s.{item_id.lower().replace('-', '_')}"),
        modality=modality, authority=SourceAuthority.REGULATORY, status=PublicationStatus.PUBLISHED,
        applicability=applicability or scope(), recorded_at=NOW,
        constraint=Constraint(all_of=(predicate,)),
        constraint_provenance={"authored_by": "user:c", "approval_ref": "A"},
        sources=(EvidenceRef(source_id=f"SRC-{item_id}"),), reviewed_by="user:c", review_ref="R", **extra,
    ))


def exception(exc_id: str, kind: str, targets, *, winner="", applicability=None, version=1, **extra):
    return KnowledgeException.model_validate(dict(
        id=exc_id, version=version, tenant="t", kind=kind, target_item_ids=targets, winner_item_id=winner,
        rationale="audit", applicability=applicability or scope(), granted_by="user:g", approval_ref="AP",
        sources=(EvidenceRef(source_id=f"SRC-{exc_id}"),), status="published", recorded_at=NOW, **extra,
    ))


def evaluate(items, exceptions=(), facts=None, access=USER, **context):
    return evaluate_knowledge(
        items=items, exceptions=exceptions, context=BusinessContext(as_of=NOW, **context),
        facts=facts if facts is not None else {"f.x": Fact(value=False, evidence=EV)},
        tenant="t", access=access, domain="audit", release=REF)


def outcome(decision, item_id):
    return next(r for r in decision.results if r.item_id == item_id)


# -- 1. partial-context exceptions ------------------------------------------


class TestPartialExceptionCoverage:
    def test_waiver_for_one_market_does_not_waive_the_other(self):
        decision = evaluate([item("M")], [exception("W", "waiver", ["M"], applicability=scope(market=["A"]))],
                            market=["A", "B"])
        result = outcome(decision, "M")
        assert result.outcome.value == "fail", "market B is not covered by the waiver"
        assert decision.overall.value == "non_compliant"
        assert "W@v1" in result.partial_exception_refs

    def test_waiver_covering_every_bound_value_still_waives(self):
        decision = evaluate([item("M", applicability=scope(market=["A"]))],
                            [exception("W", "waiver", ["M"], applicability=scope(market=["A"]))],
                            market=["A", "B"])
        assert outcome(decision, "M").outcome_reason == "waived_by_exception"
        assert decision.overall.value == "compliant"

    def test_partial_precedence_keeps_the_target(self):
        winner = item("WIN", subject_key="s.shared", predicate=Predicate(fact="f.y", operator="eq", value="a"))
        loser = item("LOS", subject_key="s.shared", predicate=Predicate(fact="f.y", operator="eq", value="b"))
        decision = evaluate([winner, loser],
                            [exception("P", "precedence", ["LOS"], winner="WIN", applicability=scope(market=["A"]))],
                            facts={"f.y": Fact(value="a", evidence=EV)}, market=["A", "B"])
        assert outcome(decision, "LOS").outcome_reason != "resolved_by_precedence"
        assert decision.human_review_required


# -- 2. precedence needs an effective winner ---------------------------------


class TestPrecedenceNeedsEffectiveWinner:
    def _pair(self, winner_scope=None):
        winner = item("WIN", subject_key="s.shared", applicability=winner_scope or scope(),
                      predicate=Predicate(fact="f.y", operator="eq", value="a"))
        loser = item("LOS", subject_key="s.shared", predicate=Predicate(fact="f.y", operator="eq", value="b"))
        return winner, loser

    def test_non_applicable_winner_does_not_suppress_the_target(self):
        winner, loser = self._pair(scope(market=["A"]))
        decision = evaluate([winner, loser], [exception("P", "precedence", ["LOS"], winner="WIN")],
                            facts={"f.y": Fact(value="a", evidence=EV)}, market=["B"])
        assert outcome(decision, "LOS").outcome.value == "fail"
        assert decision.overall.value == "non_compliant"

    def test_unknown_winner_escalates_instead_of_suppressing(self):
        winner, loser = self._pair(scope(market=["A"]))
        decision = evaluate([winner, loser], [exception("P", "precedence", ["LOS"], winner="WIN")],
                            facts={"f.y": Fact(value="b", evidence=EV)})  # market not stated
        assert outcome(decision, "LOS").outcome_reason != "resolved_by_precedence"
        assert decision.human_review_required and decision.overall.value == "needs_review"

    def test_waived_winner_does_not_suppress_the_target(self):
        winner, loser = self._pair()
        decision = evaluate([winner, loser], [
            exception("P", "precedence", ["LOS"], winner="WIN"), exception("W", "waiver", ["WIN"]),
        ], facts={"f.y": Fact(value="a", evidence=EV)})
        assert outcome(decision, "LOS").outcome.value == "fail"

    def test_superseded_winner_escalates(self):
        winner, loser = self._pair()
        newer = item("WIN2", subject_key="s.other", supersedes=("WIN",))
        decision = evaluate([winner, loser, newer], [exception("P", "precedence", ["LOS"], winner="WIN")],
                            facts={"f.y": Fact(value="b", evidence=EV), "f.x": Fact(value=True, evidence=EV)})
        assert outcome(decision, "LOS").outcome_reason != "resolved_by_precedence"
        assert decision.human_review_required

    def test_inaccessible_winner_escalates_without_naming_it(self):
        winner, loser = self._pair()
        hidden = winner.model_copy(update={"access": DocumentAccessPolicy(
            mode="restricted", state="known", allow_principals=["group:secret"])})
        decision = evaluate([hidden, loser], [exception("P", "precedence", ["LOS"], winner="WIN")],
                            facts={"f.y": Fact(value="b", evidence=EV)})
        assert outcome(decision, "LOS").outcome_reason != "resolved_by_precedence"
        assert decision.human_review_required
        assert "WIN" not in " ".join(decision.human_review_reasons)

    def test_contradictory_precedences_do_not_depend_on_order(self):
        winner, loser = self._pair()
        p1 = exception("P1", "precedence", ["LOS"], winner="WIN")
        p2 = exception("P2", "precedence", ["WIN"], winner="LOS")
        facts = {"f.y": Fact(value="a", evidence=EV)}
        results = {evaluate([winner, loser], order, facts=facts).model_dump_json(exclude={"decision_id"})
                   for order in ([p1, p2], [p2, p1])}
        assert len(results) == 1
        decision = evaluate([winner, loser], [p1, p2], facts=facts)
        assert decision.overall.value == "needs_review"


# -- 3. absence-is-violation on prohibitions ---------------------------------


class TestProhibitionAbsenceSemantics:
    def test_missing_fact_declared_as_violation_never_passes_a_prohibition(self):
        rule = item("P", modality="prohibited",
                    predicate=Predicate(fact="f.claim", operator="eq", value=True, when_missing="violation"))
        decision = evaluate([rule], facts={})
        result = outcome(decision, "P")
        assert result.outcome.value == "fail" and result.outcome_reason == "absence_is_violation"
        assert decision.overall.value == "non_compliant"

    def test_ordinary_falsity_still_passes_a_prohibition(self):
        rule = item("P", modality="prohibited",
                    predicate=Predicate(fact="f.claim", operator="eq", value=True, when_missing="violation"))
        decision = evaluate([rule], facts={"f.claim": Fact(value=False, evidence=EV)})
        assert outcome(decision, "P").outcome.value == "pass"


# -- 4. fact types -------------------------------------------------------------


class TestFactTypes:
    rule = staticmethod(lambda op="gte", value=6: item("N", predicate=Predicate(fact="f.n", operator=op, value=value)))

    @pytest.mark.parametrize("value", ["7", [7], True, {"a": 1}.get("missing", "x")])
    def test_incompatible_fact_types_are_structured_not_crashes(self, value):
        decision = evaluate([self.rule()], facts={"f.n": Fact(value=value, evidence=EV)})
        result = outcome(decision, "N")
        assert result.outcome.value == "insufficient_evidence"
        assert result.predicates[0].reason == "fact_type_mismatch"
        assert decision.overall.value == "needs_review"

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_non_finite_numbers_are_rejected_by_the_contract(self, value):
        with pytest.raises(ValidationError, match="finite"):
            Fact(value=value, evidence=EV)
        with pytest.raises(ValidationError, match="finite"):
            Fact(value=[1, value], evidence=EV)

    def test_bool_is_not_a_number_and_number_is_not_a_bool(self):
        flag = item("B", predicate=Predicate(fact="f.n", operator="eq", value=True))
        decision = evaluate([flag], facts={"f.n": Fact(value=1, evidence=EV)})
        assert outcome(decision, "B").predicates[0].reason == "fact_type_mismatch"

    def test_compatible_types_still_evaluate(self):
        decision = evaluate([self.rule()], facts={"f.n": Fact(value=6.5, evidence=EV)})
        assert outcome(decision, "N").outcome.value == "pass"
        listed = item("L", predicate=Predicate(fact="f.l", operator="contains", value="sl"))
        decision = evaluate([listed], facts={"f.l": Fact(value=["sl", "en"], evidence=EV)})
        assert outcome(decision, "L").outcome.value == "pass"
        within = item("I", predicate=Predicate(fact="f.s", operator="in", value=["a", "b"]))
        decision = evaluate([within], facts={"f.s": Fact(value=3, evidence=EV)})
        assert outcome(decision, "I").predicates[0].reason == "fact_type_mismatch"


# -- 5. supersession chains -----------------------------------------------------


class TestSupersessionChains:
    def _chain(self):
        a = item("A", series="s.rev", subject_key="s.rev")
        b = item("B", series="s.rev", subject_key="s.rev", supersedes=("A",))
        c = item("C", series="s.rev", subject_key="s.rev", supersedes=("B",))
        return [a, b, c]

    def test_chain_result_is_independent_of_input_order(self):
        seen = set()
        for order in permutations(self._chain()):
            decision = evaluate(list(order), facts={"f.x": Fact(value=True, evidence=EV)})
            seen.add((decision.applicable, tuple(c.kind for c in decision.conflicts)))
        assert seen == {(("C@v1",), ())}

    def test_transitive_supersession_when_the_middle_revision_is_not_in_force(self):
        a, b, c = self._chain()
        b = b.model_copy(update={"applicability": scope().model_copy(
            update={"valid_from": datetime(2027, 1, 1, tzinfo=timezone.utc)})})
        decision = evaluate([a, b, c], facts={"f.x": Fact(value=True, evidence=EV)})
        assert decision.applicable == ("C@v1",)

    def test_partial_scope_supersession_keeps_the_older_revision_for_uncovered_values(self):
        old = item("OLD", series="s.rev", subject_key="s.rev")
        new = item("NEW", series="s.rev", subject_key="s.rev", supersedes=("OLD",),
                   applicability=scope(market=["A"]))
        decision = evaluate([old, new], facts={"f.x": Fact(value=True, evidence=EV)}, market=["A", "B"])
        assert set(decision.applicable) == {"OLD@v1", "NEW@v1"}
        assert decision.conflicts == ()  # ordered by supersession, not ambiguous

    def test_unknown_superseder_makes_the_older_revision_unknown(self):
        old = item("OLD", series="s.rev", subject_key="s.rev")
        new = item("NEW", series="s.rev", subject_key="s.rev", supersedes=("OLD",),
                   applicability=scope(market=["A"]))
        decision = evaluate([old, new], facts={"f.x": Fact(value=True, evidence=EV)})
        assert "OLD@v1" in decision.unknown_applicability and decision.human_review_required

    def test_supersession_cycle_is_unknown_and_escalates(self):
        a = item("A", series="s.rev", subject_key="s.rev", supersedes=("B",))
        b = item("B", series="s.rev", subject_key="s.rev", supersedes=("A",))
        decision = evaluate([a, b], facts={"f.x": Fact(value=True, evidence=EV)})
        assert set(decision.unknown_applicability) == {"A@v1", "B@v1"}
        assert decision.overall.value == "needs_review"


# -- 6/7/8. release gate and identity -------------------------------------------


@pytest.fixture
async def store(tmp_path):
    from graphrag.domains.energy.governance_store import GovernanceStore

    governance = GovernanceStore(f"sqlite+aiosqlite:///{(tmp_path / 'gov.sqlite').as_posix()}",
                                 blob_root=tmp_path / "blobs")
    await governance.open()
    yield governance
    await governance.close()


class TestReleaseGate:
    async def test_publishing_without_regression_cases_is_refused(self, store):
        with pytest.raises(KnowledgePublicationError, match="regression"):
            await publish_pack(store, load_pack(PACK), published_by="user:p")
        assert await store.active_knowledge_release("packaging_demo", "packaging_labeling") is None

    async def test_refusal_preserves_an_existing_active_release(self, store):
        first = await publish_pack(store, load_pack(PACK), published_by="user:p", cases=load_cases(CASES))
        changed = load_pack(PACK).model_copy(update={"disclaimer": "changed"})
        with pytest.raises(KnowledgePublicationError):
            await publish_pack(store, changed, published_by="user:p")
        active = await store.active_knowledge_release("packaging_demo", "packaging_labeling")
        assert active["release_id"] == first["release_id"]


class TestSingleEffectiveExceptionVersion:
    def _pack_with_two_waiver_versions(self):
        pack = load_pack(PACK).model_copy(deep=True)
        old = next(e for e in pack.exceptions if e.id == "EXC-001")
        newer = old.model_copy(update={"version": 2, "applicability": old.applicability.model_copy(
            update={"valid_to": datetime(2026, 7, 15, tzinfo=timezone.utc)})})
        pack.exceptions.append(newer)
        return pack, old, newer

    def test_gate_rejects_two_published_versions(self):
        pack, _, _ = self._pack_with_two_waiver_versions()
        assert any("EXC-001" in failure for failure in check_pack(pack))

    def test_evaluator_uses_only_the_latest_version(self):
        mandatory = item("M")
        old = exception("W", "waiver", ["M"], version=1)
        newer = exception("W", "waiver", ["M"], version=2, applicability=scope(market=["Z"]))
        for order in ([old, newer], [newer, old]):
            decision = evaluate([mandatory], order, market=["A"])
            assert outcome(decision, "M").outcome.value == "fail"
            assert [e.exception_key for e in decision.exceptions_applied] == []

    def test_evaluator_uses_only_the_latest_item_version(self):
        old = item("V", version=1, predicate=Predicate(fact="f.n", operator="gte", value=6))
        new = item("V", version=2, predicate=Predicate(fact="f.n", operator="gte", value=8))
        for order in ([old, new], [new, old]):
            decision = evaluate(order, facts={"f.n": Fact(value=7, evidence=EV)})
            assert decision.applicable == ("V@v2",)
            assert outcome(decision, "V").outcome.value == "fail"


class TestReleaseIdentity:
    async def test_disclaimer_change_is_a_new_audited_release(self, store, tmp_path):
        cases = load_cases(CASES)
        pack = load_pack(PACK)
        first = await publish_pack(store, pack, published_by="user:p", cases=cases)
        again = await publish_pack(store, pack, published_by="user:p", cases=cases)
        assert again["release_id"] == first["release_id"]
        updated = pack.model_copy(update={"disclaimer": "Updated demonstration disclaimer."})
        second = await publish_pack(store, updated, published_by="user:p", cases=cases)
        assert second["release_id"] != first["release_id"] and second["version"] == 2
        assert second["content_hash"] != first["content_hash"]

        from graphrag.domains.energy.governance_store import GovernanceStore

        reopened = GovernanceStore(store.db_url, blob_root=tmp_path / "blobs")
        await reopened.open()
        try:
            active = await reopened.active_knowledge_release("packaging_demo", "packaging_labeling")
            assert active["disclaimer"] == "Updated demonstration disclaimer."
            history = await reopened.knowledge_release_history("packaging_demo", "packaging_labeling")
            assert [h["version"] for h in history] == [1, 2]
            restored = await reopened.rollback_knowledge_release(
                "packaging_demo", "packaging_labeling", first["release_id"], rolled_back_by="user:p")
            assert restored["disclaimer"] == pack.disclaimer and restored["version"] == 3
        finally:
            await reopened.close()
