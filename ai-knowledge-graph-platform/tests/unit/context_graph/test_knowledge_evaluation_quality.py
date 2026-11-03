"""Metamorphic relations, the generated split, the conflict index, stage
metrics and constraint rendering (roadmap candidates, 2026-11-03)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

import graphrag.context_graph.knowledge_evaluation as evaluation
from graphrag.context_graph.knowledge import (
    BusinessContext, Constraint, DimensionScope, Fact, KnowledgeException, KnowledgeItem, NormativeModality,
    Predicate,
)
from graphrag.context_graph.knowledge_evaluation import KnowledgeReleaseRef, detect_conflicts, evaluate_knowledge
from graphrag.context_graph.knowledge_oracle import oracle_applicability
from graphrag.context_graph.knowledge_release import load_cases, load_pack, review_queue
from graphrag.context_graph.knowledge_render import render_constraint, render_item, render_predicate
from graphrag.context_graph.knowledge_service import KnowledgeEvaluationService
from graphrag.context_graph.knowledge_synthetic import synthetic_context, synthetic_facts, synthetic_items
from graphrag.context_graph.models import ConditionOperator
from graphrag.enterprise.models import AccessContext

ROOT = Path(__file__).resolve().parents[3]
PACK = ROOT / "config" / "knowledge_packs" / "packaging_labeling.yml"
SETS = [ROOT / "evals" / "knowledge_eval" / name
        for name in ("packaging_labeling_cases.json", "packaging_labeling_heldout.json")]
REF = KnowledgeReleaseRef(release_id="m", domain="packaging_labeling", version=0)
ALL_CASES = [case for path in SETS for case in load_cases(path)]


def _run(case, items, exceptions, **overrides):
    inputs = dict(context=BusinessContext.model_validate(case.context),
                  facts={k: Fact.model_validate(v) for k, v in case.facts.items()},
                  access=AccessContext.from_claims(case.principal), tenant=case.tenant,
                  expect_applicable_knowledge=case.expect_applicable_knowledge,
                  semantic_candidates=case.semantic_candidates)
    inputs.update(overrides)
    return evaluate_knowledge(items=items, exceptions=exceptions, domain="packaging_labeling", release=REF,
                              **inputs)


def _dump(decision):
    return decision.model_dump(mode="json")


@pytest.fixture(scope="module")
def pack():
    return load_pack(PACK)


# -- metamorphic relations ---------------------------------------------------------


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda c: c.id)
class TestMetamorphicRelations:
    def test_input_order_does_not_matter(self, case, pack):
        items, exceptions = pack.published_items(), pack.published_exceptions()
        assert _dump(_run(case, list(reversed(items)), list(reversed(exceptions)))) == \
            _dump(_run(case, items, exceptions))
        shuffled = items[:]
        random.Random(case.id).shuffle(shuffled)
        assert _dump(_run(case, shuffled, exceptions)) == _dump(_run(case, items, exceptions))

    def test_another_tenants_knowledge_changes_nothing(self, case, pack):
        items, exceptions = pack.published_items(), pack.published_exceptions()
        foreign = [i.model_copy(update={"tenant": "intruder", "id": "X" + i.id}) for i in items]
        foreign_exc = [e.model_copy(update={"tenant": "intruder", "id": "X" + e.id}) for e in exceptions]
        assert _dump(_run(case, items + foreign, exceptions + foreign_exc)) == _dump(_run(case, items, exceptions))

    def test_unpublished_and_inaccessible_items_change_nothing(self, case, pack):
        items, exceptions = pack.published_items(), pack.published_exceptions()
        draft = items[0].model_copy(update={"id": "PKG-DRAFT", "status": "draft"})
        hidden = items[0].model_copy(update={"id": "PKG-HIDDEN", "access": items[0].access.model_copy(
            update={"mode": "restricted", "state": "known", "allow_principals": ["group:nobody"]})})
        assert _dump(_run(case, items + [draft, hidden], exceptions)) == _dump(_run(case, items, exceptions))

    def test_an_out_of_scope_item_changes_nothing_but_its_own_row(self, case, pack):
        items, exceptions = pack.published_items(), pack.published_exceptions()
        scope = items[0].applicability.model_copy(update={"market": DimensionScope(mode="in", values=["ZZ_NOWHERE"])})
        stray = items[0].model_copy(update={"id": "PKG-STRAY", "applicability": scope})
        base, extended = _run(case, items, exceptions), _run(case, items + [stray], exceptions)
        if case.tenant != stray.tenant:
            assert _dump(extended) == _dump(base), "another tenant never sees the stray item"
            return
        assert (extended.overall, extended.human_review_required) == (base.overall, base.human_review_required)
        assert extended.applicable == base.applicable
        stray_row = next(r for r in extended.results if r.item_id == "PKG-STRAY")
        assert stray_row.applicability.value in ("does_not_apply", "unknown")
        if stray_row.applicability.value == "does_not_apply":
            assert [r for r in extended.results if r.item_id != "PKG-STRAY"] == list(base.results)

    def test_consistent_renaming_preserves_every_outcome(self, case, pack):
        items, exceptions = pack.published_items(), pack.published_exceptions()
        name = {i.id: "R" + i.id for i in items}

        def rename_item(item: KnowledgeItem) -> KnowledgeItem:
            return item.model_copy(update={
                "id": name[item.id], "supersedes": tuple(name.get(t, t) for t in item.supersedes),
                "overrides": tuple(name.get(t, t) for t in item.overrides)})

        def rename_exc(exc: KnowledgeException) -> KnowledgeException:
            return exc.model_copy(update={
                "id": "R" + exc.id, "target_item_ids": tuple(name.get(t, t) for t in exc.target_item_ids),
                "winner_item_id": name.get(exc.winner_item_id, exc.winner_item_id)})

        semantic = [name.get(c, c) for c in case.semantic_candidates] if case.semantic_candidates else None
        base = _run(case, items, exceptions)
        renamed = _run(case, [rename_item(i) for i in items], [rename_exc(e) for e in exceptions],
                       semantic_candidates=semantic)
        assert (renamed.overall, renamed.human_review_required) == (base.overall, base.human_review_required)
        assert {name[r.item_id]: (r.applicability, r.outcome) for r in base.results} == \
            {r.item_id: (r.applicability, r.outcome) for r in renamed.results}
        assert sorted(c.kind for c in renamed.conflicts) == sorted(c.kind for c in base.conflicts)


# -- the generated split and the conflict index ----------------------------------------


def test_generated_contexts_agree_with_the_independent_oracle():
    items = synthetic_items(150, seed=5)
    rng = random.Random(5)
    access = AccessContext.from_claims({"sub": "u"})
    compared = 0
    for _ in range(60):
        context, facts = synthetic_context(rng), synthetic_facts(rng)
        decision = evaluate_knowledge(items=items, context=context, facts=facts, tenant="syn", access=access,
                                      domain="synthetic", release=REF)
        by_item = {r.item_id: r.applicability.value for r in decision.results}
        for item in items:
            expected = oracle_applicability(item, context)
            if expected is not None:
                compared += 1
                assert by_item[item.id] == expected, (item.id, context)
    assert compared >= 150 * 60


def test_the_conflict_index_gives_identical_decisions(monkeypatch, pack):
    items = synthetic_items(300, seed=9)
    rng = random.Random(9)
    access = AccessContext.from_claims({"sub": "u"})
    contexts = [(synthetic_context(rng), synthetic_facts(rng)) for _ in range(25)]

    def run_all():
        return [evaluate_knowledge(items=items, context=c, facts=f, tenant="syn", access=access,
                                   domain="synthetic", release=REF).model_dump(mode="json") for c, f in contexts]

    indexed = run_all()
    reference = evaluation._candidate_pairs
    monkeypatch.setattr(evaluation, "_candidate_pairs", lambda xs, indexed=True: reference(xs, indexed=False))
    assert run_all() == indexed
    assert detect_conflicts(pack.published_items(), indexed=True) == detect_conflicts(pack.published_items(),
                                                                                       indexed=False)
    assert any(c.kind for c in detect_conflicts(items, indexed=True)), "the synthetic pack does produce conflicts"


async def test_service_reports_cache_hit_rate_and_stage_times(tmp_path, pack):
    from graphrag.context_graph.knowledge_release import publish_pack
    from graphrag.domains.energy.governance_store import GovernanceStore

    store = GovernanceStore(f"sqlite+aiosqlite:///{(tmp_path / 'g.sqlite').as_posix()}", blob_root=tmp_path / "b")
    await store.open()
    try:
        await publish_pack(store, pack, published_by="user:p", cases=load_cases(SETS[0]))
        service = KnowledgeEvaluationService(store)
        case = ALL_CASES[0]
        args = dict(tenant=case.tenant, domain="packaging_labeling",
                    access=AccessContext.from_claims(case.principal),
                    context=BusinessContext.model_validate(case.context), record=False)
        await service.evaluate(**args)
        await service.evaluate(**args)
        metrics = service.metrics()
    finally:
        await store.close()
    assert metrics["cache"] == {"hits": 1, "misses": 1, "entries": 1, "hit_rate": 0.5}
    assert {"applicability", "conflicts", "validation", "decision"} <= set(metrics["stage_seconds"])
    assert metrics["mean_ms_per_evaluation"] > 0


def test_timings_never_change_the_decision(pack):
    case = ALL_CASES[0]
    timings: dict[str, float] = {}
    assert _dump(_run(case, pack.published_items(), pack.published_exceptions())) == \
        _dump(evaluate_knowledge(items=pack.published_items(), exceptions=pack.published_exceptions(),
                                 context=BusinessContext.model_validate(case.context),
                                 facts={k: Fact.model_validate(v) for k, v in case.facts.items()},
                                 tenant=case.tenant, access=AccessContext.from_claims(case.principal),
                                 domain="packaging_labeling", release=REF, timings=timings))
    assert all(value >= 0 for value in timings.values()) and len(timings) == 7


# -- constraint rendering ----------------------------------------------------------------


@pytest.mark.parametrize("operator,value,text", [
    ("eq", True, "x.f equals true"), ("ne", "a", 'x.f does not equal "a"'), ("gt", 2, "x.f is greater than 2"),
    ("gte", 6, "x.f is at least 6"), ("lt", 1.5, "x.f is less than 1.5"), ("lte", 9, "x.f is at most 9"),
    ("in", ["a", "b"], 'x.f is one of ["a", "b"]'), ("contains", "sl", 'x.f contains "sl"'),
    ("is_empty", None, "x.f is empty"), ("not_empty", None, "x.f is not empty"),
])
def test_every_operator_has_a_golden_rendering(operator, value, text):
    assert render_predicate(Predicate(fact="x.f", operator=operator, value=value)) == text


def test_missing_value_semantics_and_modality_are_stated():
    predicate = Predicate(fact="x.f", operator="not_empty", when_missing="violation")
    assert render_predicate(predicate) == "x.f is not empty (a missing value counts as a violation)"
    constraint = Constraint(all_of=(Predicate(fact="a.b", operator="gte", value=8),
                                    Predicate(fact="c.d", operator="eq", value=True)))
    assert render_constraint(constraint, NormativeModality.PROHIBITED) == \
        "It is prohibited that a.b is at least 8 and c.d equals true."
    assert render_constraint(constraint, NormativeModality.MANDATORY).startswith("It is required that ")
    assert set(ConditionOperator) <= {ConditionOperator(op) for op in
                                      ("eq", "ne", "gt", "gte", "lt", "lte", "in", "contains", "is_empty", "not_empty")}


def test_review_queue_shows_the_executable_form_next_to_the_statement(pack):
    queue = review_queue(pack)
    assert queue and all("executable" in entry and "statement" in entry for entry in queue)
    rendered = {item.key: render_item(item) for item in pack.items}
    assert all(entry["executable"] == rendered[entry["key"]] for entry in queue)
    unchecked = next(i for i in pack.items if i.constraint is None)
    assert render_item(unchecked).startswith("No executable constraint")
