"""Contextual knowledge evaluation: contracts, applicability, validation,
conflicts, releases, caching and the consumer contract.

Deterministic and offline: the synthetic packaging pack is loaded from YAML,
releases use a temporary SQLite governance store, no network or model calls.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from graphrag.context_graph.applicability import ApplicabilityStatus, evaluate_applicability
from graphrag.context_graph.knowledge import (
    Applicability, BusinessContext, Constraint, DimensionScope, EvidenceRef, KnowledgeItem,
    NormativeModality, Predicate, PublicationStatus, SourceAuthority,
)
from graphrag.context_graph.knowledge_evaluation import (
    KnowledgeReleaseRef, detect_conflicts, evaluate_knowledge,
)
from graphrag.context_graph.knowledge_release import (
    KnowledgePublicationError, approve_item, case_mismatches, check_pack, load_cases, load_pack,
    publish_pack, review_queue, run_case,
)
from graphrag.context_graph.knowledge_service import KnowledgeEvaluationService, NoActiveReleaseError
from graphrag.context_graph.models import ConditionOperator
from graphrag.enterprise.models import AccessContext, ObligationDraft

ROOT = Path(__file__).resolve().parents[3]
PACK = ROOT / "config" / "knowledge_packs" / "packaging_labeling.yml"
CASES = ROOT / "evals" / "knowledge_eval" / "packaging_labeling_cases.json"
NOW = datetime(2026, 8, 1, tzinfo=timezone.utc)
REF = KnowledgeReleaseRef(release_id="test", domain="packaging_labeling", version=0)


@pytest.fixture(scope="module")
def pack():
    return load_pack(PACK)


def _item(**overrides) -> KnowledgeItem:
    base = dict(
        id="T-1", version=1, tenant="t", title="t", statement="s", subject_key="x.y",
        modality=NormativeModality.MANDATORY, authority=SourceAuthority.REGULATORY,
        status=PublicationStatus.PUBLISHED, recorded_at=NOW,
        applicability=Applicability(**{d: DimensionScope.unrestricted() for d in (
            "customer", "brand", "market", "product", "product_family", "workflow")}),
        constraint=Constraint(all_of=(Predicate(fact="a.b", operator=ConditionOperator.GTE, value=6),)),
        constraint_provenance={"authored_by": "user:c", "approval_ref": "A-1"},
        sources=(EvidenceRef(source_id="S-1"),), reviewed_by="user:c", review_ref="R-1",
    )
    base.update(overrides)
    return KnowledgeItem.model_validate(base)


# -- contracts --------------------------------------------------------------


class TestContracts:
    def test_extracted_text_never_becomes_executable_without_human_provenance(self):
        with pytest.raises(ValidationError, match="extracted text cannot carry an executable constraint"):
            _item(origin="extracted", constraint_provenance=None, status="draft", confidence=0.99)

    def test_published_constraint_requires_provenance_sources_and_reviewer(self):
        with pytest.raises(ValidationError, match="constraint_provenance"):
            _item(constraint_provenance=None)
        with pytest.raises(ValidationError, match="cite at least one source"):
            _item(sources=())
        with pytest.raises(ValidationError, match="reviewed_by and review_ref"):
            _item(reviewed_by="")

    def test_constraint_vocabulary_is_bounded_and_rejects_expressions(self):
        with pytest.raises(ValidationError):
            Predicate(fact="a.b", operator="matches_regex", value=".*")
        with pytest.raises(ValidationError, match="numeric"):
            Predicate(fact="a.b", operator=ConditionOperator.GT, value="__import__('os')")
        with pytest.raises(ValidationError):
            Predicate(fact="a.b; drop", operator=ConditionOperator.EQ, value=1)
        with pytest.raises(ValidationError, match="scalar"):
            Predicate(fact="a.b", operator=ConditionOperator.EQ, value={"$where": "1"})

    def test_obligation_draft_adapts_to_a_reviewable_draft_without_rules(self):
        draft = ObligationDraft(obligation="Supplier must notify within 5 days", evidence_chunk_id="c-1",
                                evidence_quote="must notify within 5 days", confidence=0.99)
        item = KnowledgeItem.from_obligation_draft(
            draft, item_id="OB-1", tenant="t", subject_key="contract.notice", title="Notice", recorded_at=NOW)
        assert item.status == PublicationStatus.DRAFT and item.constraint is None
        assert all(item.applicability.scope(d) is None for d in ("customer", "market", "workflow"))
        assert item.confidence == 0.99 and item.origin.value == "extracted"
        # approval needs a reviewer identity and still cannot invent a rule
        with pytest.raises(ValueError, match="reviewer identity"):
            approve_item(item, reviewed_by="", review_ref="R")
        published = approve_item(item, reviewed_by="user:curator", review_ref="REV-9")
        assert published.status == PublicationStatus.PUBLISHED and published.version == 2
        assert published.constraint is None  # still not machine-checkable


# -- applicability ----------------------------------------------------------


class TestApplicabilitySemantics:
    scoped = Applicability(market=DimensionScope(mode="in", values=["NORTHLAND"]),
                           **{d: DimensionScope.unrestricted() for d in (
                               "customer", "brand", "product", "product_family", "workflow")})

    def test_unrestricted_dimension_needs_no_context(self):
        all_any = Applicability(**{d: DimensionScope.unrestricted() for d in (
            "customer", "brand", "market", "product", "product_family", "workflow")})
        assert evaluate_applicability(all_any, BusinessContext()).status == ApplicabilityStatus.APPLIES

    def test_missing_context_is_unknown_not_broadened(self):
        result = evaluate_applicability(self.scoped, BusinessContext(customer=["X"]))
        assert result.status == ApplicabilityStatus.UNKNOWN
        assert {r.code for r in result.reasons} == {"context_missing"}

    def test_unknown_metadata_is_unknown_not_everywhere(self):
        result = evaluate_applicability(Applicability(), BusinessContext(market=["NORTHLAND"]))
        assert result.status == ApplicabilityStatus.UNKNOWN

    def test_a_definite_exclusion_wins_over_unknowns(self):
        partial = Applicability(market=DimensionScope(mode="in", values=["NORTHLAND"]))
        result = evaluate_applicability(partial, BusinessContext(market=["SOUTHLAND"]))
        assert result.status == ApplicabilityStatus.DOES_NOT_APPLY

    def test_multiple_values_report_partial_scope(self):
        result = evaluate_applicability(self.scoped, BusinessContext(market=["NORTHLAND", "SOUTHLAND"]))
        assert result.status == ApplicabilityStatus.APPLIES
        assert result.matched["market"] == ("NORTHLAND",)
        assert "partial_scope_match" in {r.code for r in result.reasons}

    def test_not_in_scope_and_empty_context_list(self):
        excluding = self.scoped.model_copy(update={"market": DimensionScope(mode="not_in", values=["SOUTHLAND"])})
        assert evaluate_applicability(excluding, BusinessContext(market=["SOUTHLAND"])).status == \
            ApplicabilityStatus.DOES_NOT_APPLY
        assert evaluate_applicability(excluding, BusinessContext(market=[])).status == ApplicabilityStatus.UNKNOWN

    def test_time_window_and_recorded_time(self):
        windowed = self.scoped.model_copy(update={
            "valid_from": datetime(2026, 7, 1, tzinfo=timezone.utc)})
        ctx = BusinessContext(market=["NORTHLAND"], as_of=datetime(2026, 6, 1, tzinfo=timezone.utc))
        assert evaluate_applicability(windowed, ctx).status == ApplicabilityStatus.DOES_NOT_APPLY
        assert evaluate_applicability(windowed, BusinessContext(market=["NORTHLAND"])).status == \
            ApplicabilityStatus.UNKNOWN  # window but no as_of
        late = evaluate_applicability(self.scoped, BusinessContext(market=["NORTHLAND"], as_of=NOW,
                                                                   known_as=datetime(2026, 1, 1, tzinfo=timezone.utc)),
                                      recorded_at=NOW)
        assert late.status == ApplicabilityStatus.DOES_NOT_APPLY
        assert late.reasons[0].code == "not_recorded_at_view"


# -- conflicts ---------------------------------------------------------------


class TestConflictRules:
    def test_bounds_in_the_same_direction_are_compatible(self):
        a = _item(id="A")
        b = _item(id="B", constraint=Constraint(all_of=(Predicate(fact="a.b", operator="gte", value=8),)))
        assert detect_conflicts([a, b]) == []

    def test_disjoint_bounds_and_different_required_values_conflict(self):
        a = _item(id="A")
        b = _item(id="B", constraint=Constraint(all_of=(Predicate(fact="a.b", operator="lt", value=5),)))
        assert [c.kind for c in detect_conflicts([a, b])] == ["incompatible_requirements"]
        c = _item(id="C", constraint=Constraint(all_of=(Predicate(fact="a.b", operator="eq", value=1),)))
        d = _item(id="D", constraint=Constraint(all_of=(Predicate(fact="a.b", operator="eq", value=2),)))
        assert [x.kind for x in detect_conflicts([c, d])] == ["incompatible_requirements"]

    def test_mandatory_and_recommendation_never_conflict_or_cancel(self):
        mandatory = _item(id="M")
        advisory = _item(id="R", modality="recommended",
                         constraint=Constraint(all_of=(Predicate(fact="a.b", operator="lt", value=5),)))
        assert detect_conflicts([mandatory, advisory]) == []


# -- the deterministic evaluation set ----------------------------------------


@pytest.mark.parametrize("case", load_cases(CASES), ids=lambda c: c.id)
def test_evaluation_case(pack, case):
    decision = run_case(case, items=pack.published_items(), exceptions=pack.published_exceptions(),
                        domain=pack.domain, release=REF)
    assert case_mismatches(case, decision) == []


def test_unauthorised_knowledge_never_leaks_into_any_part_of_the_decision(pack):
    case = next(c for c in load_cases(CASES) if c.id == "KE-15")
    decision = run_case(case, items=pack.published_items(), exceptions=pack.published_exceptions(),
                        domain=pack.domain, release=REF)
    dumped = json.dumps(decision.model_dump(mode="json"))
    assert "PKG-015" not in dumped and "DEMO-NL-REG-MEMO" not in dumped


def test_business_context_cannot_grant_access(pack):
    """Asking about the restricted rule's exact scope does not reveal it."""
    decision = evaluate_knowledge(
        items=pack.published_items(), context=BusinessContext(market=["NORTHLAND"], as_of=NOW),
        tenant="packaging_demo", access=AccessContext.from_claims({"sub": "u", "groups": ["marketing"]}),
        domain=pack.domain, release=REF)
    assert all(r.item_id != "PKG-015" for r in decision.results)


def test_every_applicable_requirement_is_evaluated_not_just_the_first(pack):
    case = next(c for c in load_cases(CASES) if c.id == "KE-02")
    decision = run_case(case, items=pack.published_items(), exceptions=pack.published_exceptions(),
                        domain=pack.domain, release=REF)
    performed = [r.item_id for r in decision.results if r.validation_performed]
    assert set(performed) >= {"PKG-001", "PKG-005", "PKG-007", "PKG-011", "PKG-012"}
    assert decision.escalation.policy_id == "knowledge-evaluation-escalation"


# -- publication gate and durable releases ----------------------------------


@pytest.fixture
async def store(tmp_path):
    from graphrag.domains.energy.governance_store import GovernanceStore

    governance = GovernanceStore(f"sqlite+aiosqlite:///{(tmp_path / 'gov.sqlite').as_posix()}",
                                 blob_root=tmp_path / "blobs")
    await governance.open()
    yield governance
    await governance.close()


class TestReleases:
    async def test_publish_serves_only_published_and_stamps_the_release(self, pack, store):
        cases = load_cases(CASES)
        release = await publish_pack(store, pack, published_by="user:publisher", cases=cases)
        assert release["version"] == 1 and release["checks"]["regression"] == "passed"
        assert "PKG-013@v1" not in release["item_keys"] and "PKG-014@v2" not in release["item_keys"]
        again = await publish_pack(store, pack, published_by="user:publisher", cases=cases)
        assert again["release_id"] == release["release_id"]  # identical content is not a new version
        service = KnowledgeEvaluationService(store)
        decision = await service.evaluate(
            tenant="packaging_demo", domain="packaging_labeling",
            access=AccessContext.from_claims({"sub": "u"}),
            context=BusinessContext(market=["NORTHLAND"], product_family=["otc_tablets"], as_of=NOW))
        assert decision.release.release_id == release["release_id"]
        assert decision.release.content_hash == release["content_hash"]

    async def test_failed_gate_keeps_the_previous_release_active(self, pack, store):
        first = await publish_pack(store, pack, published_by="user:publisher", cases=load_cases(CASES))
        broken = pack.model_copy(deep=True)
        index = next(i for i, item in enumerate(broken.items) if item.id == "PKG-011")
        data = broken.items[index].model_dump()
        data["constraint"] = {"all_of": [{"fact": "label.min_font_pt", "operator": "gte", "value": 9}]}
        broken.items[index] = KnowledgeItem.model_validate(data)
        with pytest.raises(KnowledgePublicationError) as exc:
            await publish_pack(store, broken, published_by="user:publisher", cases=load_cases(CASES))
        assert any("KE-01" in failure for failure in exc.value.failures)
        active = await store.active_knowledge_release("packaging_demo", "packaging_labeling")
        assert active["release_id"] == first["release_id"]

    async def test_referential_gate_and_rollback(self, pack, store):
        dangling = pack.model_copy(deep=True)
        data = dangling.items[0].model_dump()
        data["supersedes"] = ["PKG-404"]
        dangling.items[0] = KnowledgeItem.model_validate(data)
        assert any("unknown item PKG-404" in f for f in check_pack(dangling))
        cases = load_cases(CASES)
        first = await publish_pack(store, pack, published_by="user:p", cases=cases)
        revised = pack.model_copy(update={"disclaimer": pack.disclaimer + " (revised)"})
        second = await publish_pack(store, revised, published_by="user:p", cases=cases)
        assert second["version"] == 2
        restored = await store.rollback_knowledge_release(
            "packaging_demo", "packaging_labeling", first["release_id"], rolled_back_by="user:p")
        assert restored["version"] == 3 and restored["rolled_back_from"] == first["release_id"]
        active = await store.active_knowledge_release("packaging_demo", "packaging_labeling")
        assert active["content_hash"] == first["content_hash"]
        assert [r["version"] for r in await store.knowledge_release_history(
            "packaging_demo", "packaging_labeling")] == [1, 2, 3]

    async def test_no_release_is_a_typed_error(self, store):
        with pytest.raises(NoActiveReleaseError):
            await KnowledgeEvaluationService(store).catalog("packaging_demo", "packaging_labeling")

    def test_review_queue_lists_drafts_with_what_they_still_need(self, pack):
        queue = review_queue(pack)
        assert [q["key"] for q in queue] == ["PKG-013@v1"]
        assert any("applicability" in need for need in queue[0]["needs"])


# -- cache identity ----------------------------------------------------------


class TestDecisionCache:
    async def test_cache_never_crosses_tenant_access_context_or_release(self, pack, store):
        cases = load_cases(CASES)
        await publish_pack(store, pack, published_by="user:p", cases=cases)
        service = KnowledgeEvaluationService(store)
        ctx = BusinessContext(market=["NORTHLAND"], product_family=["otc_tablets"], as_of=NOW)
        plain = AccessContext.from_claims({"sub": "u"})
        regulatory = AccessContext.from_claims({"sub": "u", "groups": ["regulatory-affairs"]})
        args = dict(tenant="packaging_demo", domain="packaging_labeling", context=ctx)
        first = await service.evaluate(access=plain, **args)
        assert (await service.evaluate(access=plain, **args)) is first and service.cache_hits == 1
        privileged = await service.evaluate(access=regulatory, **args)
        assert privileged is not first
        assert any(r.item_id == "PKG-015" for r in privileged.results)
        assert all(r.item_id != "PKG-015" for r in first.results)
        other_view = await service.evaluate(access=plain, **{**args, "context": ctx.model_copy(
            update={"as_of": datetime(2026, 3, 1, tzinfo=timezone.utc)})})
        assert other_view.decision_id != first.decision_id
        revised = pack.model_copy(update={"disclaimer": pack.disclaimer + " (revised)"})
        await publish_pack(store, revised, published_by="user:p", cases=cases)
        after_release = await service.evaluate(access=plain, **args)
        assert after_release.release.release_id != first.release.release_id
        with pytest.raises(NoActiveReleaseError):
            await service.evaluate(access=plain, **{**args, "tenant": "other_tenant"})

    def test_query_cache_identity_includes_release_and_business_context(self):
        from graphrag.retrieval.query_cache import QueryCacheContext, build_cache_key

        base = QueryCacheContext(corpus_revision=1, requested_mode="local", effective_mode="local",
                                 model_route={}, prompt_version="p", retrieval_config={}, ontology_version="o")
        keys = {build_cache_key("q", "t", base),
                build_cache_key("q", "t", QueryCacheContext(**{**base.canonical_content(), "knowledge_release": "r2"})),
                build_cache_key("q", "t", QueryCacheContext(**{**base.canonical_content(),
                                                               "business_context_fingerprint": "ctx"}))}
        assert len(keys) == 3 and base.cache_schema_version == "4"
