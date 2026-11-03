"""Decision tables, business definitions, expert elicitation, use-case charters
and the answer-trust workflow (roadmap candidates, 2026-11-03)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from pydantic import ValidationError
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes import knowledge as knowledge_routes
from graphrag.context_graph.answer_trust import ROOT_CAUSES, trust_statistics
from graphrag.context_graph.charters import Charter, CharterError, load_charter, run_charter
from graphrag.context_graph.decision_tables import DecisionTable, table_errors, validate_table
from graphrag.context_graph.definitions import BusinessDefinition, ExecutableForm, visible_definitions
from graphrag.context_graph.elicitation import ElicitationError, interview_to_draft, record_disagreement
from graphrag.context_graph.knowledge import (
    BusinessContext, EvidenceRef, ExpertDisagreement, Fact, KnowledgeItem, Predicate, PublicationStatus,
)
from graphrag.context_graph.knowledge_decisions import decisions_using
from graphrag.context_graph.knowledge_release import check_pack_issues, load_cases, load_pack, publish_pack, review_queue
from graphrag.context_graph.knowledge_service import KnowledgeEvaluationService, set_knowledge_service
from graphrag.domains.energy.governance_store import GovernanceStore
from graphrag.enterprise.models import AccessContext

ROOT = Path(__file__).resolve().parents[3]
PACK = ROOT / "config" / "knowledge_packs" / "packaging_labeling.yml"
CASES = ROOT / "evals" / "knowledge_eval" / "packaging_labeling_cases.json"
TENANT, DOMAIN = "packaging_demo", "packaging_labeling"
NOW = datetime(2026, 8, 1, tzinfo=timezone.utc)
EV = [{"source_id": "ART"}]


# -- decision tables ----------------------------------------------------------------


def _table(rules, policy="unique", **extra):
    return DecisionTable.model_validate({
        "id": "TPL-CARTON", "version": 1, "tenant": TENANT, "title": "carton template", "hit_policy": policy,
        "inputs": [{"name": "market", "kind": "enum", "values": ["NORTHLAND", "SOUTHLAND"]},
                   {"name": "pack.size_ml", "kind": "number", "minimum": 0, "maximum": 500}],
        "outputs": ["template"], "rules": rules, **extra})


def _rule(rid, market=None, low=None, high=None, template="A", high_inclusive=True):
    when = {}
    if market:
        when["market"] = {"op": "eq", "value": market}
    if low is not None or high is not None:
        when["pack.size_ml"] = {"op": "range", "low": low, "high": high, "high_inclusive": high_inclusive}
    return {"id": rid, "when": when, "then": {"template": template}}


class TestDecisionTables:
    def test_a_complete_non_overlapping_table_is_valid(self):
        table = _table([_rule("R1", "NORTHLAND"), _rule("R2", "SOUTHLAND", high=100, high_inclusive=False),
                        _rule("R3", "SOUTHLAND", low=100)])
        assert validate_table(table) == []

    def test_overlap_and_gap_under_unique(self):
        overlap = _table([_rule("R1", "NORTHLAND"), _rule("R2", "NORTHLAND", high=50), _rule("R3", "SOUTHLAND")])
        assert {i.code for i in table_errors(overlap)} == {"overlap"}
        gap = _table([_rule("R1", "NORTHLAND"), _rule("R2", "SOUTHLAND", high=100, high_inclusive=False)])
        issues = table_errors(gap)
        assert [i.code for i in issues] == ["gap"] and issues[0].example["market"] == "SOUTHLAND"

    def test_hit_policies_decide_what_is_an_error(self):
        rules = [_rule("R1", "NORTHLAND"), _rule("R2", "NORTHLAND", high=50, template="B")]
        first = _table(rules, policy="first")
        assert [i.code for i in table_errors(first)] == ["gap"], "SOUTHLAND is still uncovered"
        assert any(i.code == "overlap" and i.severity == "info" for i in validate_table(first))
        assert {i.code for i in table_errors(_table(rules, policy="any"))} == {"conflicting_outputs", "gap"}
        assert table_errors(_table(rules, policy="collect")) == []

    @pytest.mark.parametrize("condition", [
        {"op": "feel", "value": "size > 10"},
        {"op": "eq", "value": "WESTLAND"},
        {"op": "range", "low": 10, "high": 5},
        {"op": "in", "values": []},
    ])
    def test_unsupported_content_is_refused_not_approximated(self, condition):
        field = "pack.size_ml" if condition["op"] == "range" else "market"
        with pytest.raises(ValidationError):
            _table([{"id": "R1", "when": {field: condition}, "then": {"template": "A"}}])

    def test_inputs_need_declared_domains(self):
        with pytest.raises(ValidationError, match="minimum <= maximum"):
            DecisionTable.model_validate({
                "id": "T", "version": 1, "tenant": "t", "title": "t", "hit_policy": "unique",
                "inputs": [{"name": "x", "kind": "number"}], "outputs": ["o"],
                "rules": [{"id": "R", "when": {}, "then": {"o": 1}}]})

    def test_the_gate_rejects_a_published_invalid_table(self):
        bad = _table([_rule("R1", "NORTHLAND")], status="published")
        pack = load_pack(PACK).model_copy(update={"decision_tables": [bad]})
        assert [i.code for i in check_pack_issues(pack)] == ["decision_table_invalid"]
        draft = _table([_rule("R1", "NORTHLAND")])
        assert check_pack_issues(load_pack(PACK).model_copy(update={"decision_tables": [draft]})) == []


# -- business definitions ----------------------------------------------------------------


def _definition(version=1, **extra):
    data = dict(id="DEF-MIN-FONT", version=version, tenant=TENANT, term="minimum font size", owner="user:owner",
                definition=f"Smallest printed character height, revision {version}", precision="exact",
                executable=ExecutableForm(kind="predicate",
                                          predicate=Predicate(fact="label.min_font_pt", operator="gte", value=6)),
                sources=(EvidenceRef(source_id="DEF-SRC"),), status="published", recorded_at=NOW)
    data.update(extra)
    return BusinessDefinition.model_validate(data)


def _pack_with_definition(version):
    pack = load_pack(PACK)
    items = [i.model_copy(update={"definition_refs": ("DEF-MIN-FONT",)}) if i.id == "PKG-007" else i
             for i in pack.items]
    return pack.model_copy(update={"items": items, "definitions": [_definition(version)]})


class TestBusinessDefinitions:
    def test_precision_rules(self):
        with pytest.raises(ValidationError, match="exact definition needs"):
            _definition(executable=None)
        with pytest.raises(ValidationError, match="false precision"):
            _definition(precision="loose")
        loose = _definition(precision="loose", executable=None)
        assert loose.executable is None
        with pytest.raises(ValidationError, match="precision_note"):
            _definition(precision="bounded")
        with pytest.raises(ValidationError, match="cites at least one source"):
            _definition(sources=())

    def test_tenants_define_the_same_term_independently(self):
        mine, theirs = _definition(), _definition(tenant="other", definition="something else")
        user = AccessContext.from_claims({"sub": "u"})
        assert [d.definition for d in visible_definitions([mine, theirs], tenant=TENANT, access=user)] == \
            [mine.definition]
        assert [d.tenant for d in visible_definitions([mine, theirs], tenant="other", access=user)] == ["other"]

    def test_the_gate_checks_references_and_duplicate_terms(self):
        pack = _pack_with_definition(1).model_copy(update={"definitions": []})
        assert [i.code for i in check_pack_issues(pack)] == ["definition_unknown"]
        duplicate = _pack_with_definition(1)
        duplicate = duplicate.model_copy(update={"definitions": [*duplicate.definitions,
                                                                 _definition(id="DEF-FONT-2")]})
        assert "duplicate_definition_term" in {i.code for i in check_pack_issues(duplicate)}

    async def test_a_definition_change_lists_the_decisions_that_used_the_old_version(self, tmp_path):
        store = GovernanceStore(f"sqlite+aiosqlite:///{(tmp_path / 'g.sqlite').as_posix()}", blob_root=tmp_path / "b")
        await store.open()
        try:
            cases = load_cases(CASES)
            case = next(c for c in cases if c.id == "KE-01")
            service = KnowledgeEvaluationService(store)
            inputs = dict(tenant=TENANT, domain=DOMAIN, access=AccessContext.from_claims(case.principal),
                          context=BusinessContext.model_validate(case.context),
                          facts={k: Fact.model_validate(v) for k, v in case.facts.items()})
            await publish_pack(store, _pack_with_definition(1), published_by="user:p", cases=cases)
            first = await service.evaluate(**inputs)
            await publish_pack(store, _pack_with_definition(2), published_by="user:p", cases=cases)
            second = await service.evaluate(**inputs)
            records = await service.decision_log(TENANT, DOMAIN)
        finally:
            await store.close()
        assert first.definitions_used == ("DEF-MIN-FONT@v1",) and second.definitions_used == ("DEF-MIN-FONT@v2",)
        old_users = decisions_using(records, "DEF-MIN-FONT@v1")
        assert len(old_users) == 1 and old_users == [r["record_id"] for r in records
                                                     if r["decision"]["decision_id"] == first.decision_id]


# -- expert elicitation ---------------------------------------------------------------------


ANSWERS = {"statement": "Cartons for children's products carry a choking-hazard warning.",
           "subject": "label.choking_warning", "modality": "mandatory",
           "scope": {"product_family": ["toys"], "market": "any"}, "confidence": "certain",
           "evidence": "INTERNAL-MEMO-12"}


class TestExpertElicitation:
    def test_an_interview_becomes_an_attributed_draft_with_unknown_scope_where_unstated(self):
        item = interview_to_draft(ANSWERS, item_id="PKG-100", tenant=TENANT, expert="user:anna",
                                  captured_at=NOW, role="regulatory lead")
        assert item.status == PublicationStatus.DRAFT and item.constraint is None
        assert item.expert_attribution[0].expert == "user:anna" and item.expert_attribution[0].method == "interview"
        assert item.applicability.scope("customer") is None, "an unstated dimension stays unknown"
        assert item.applicability.scope("market").mode == "any"
        assert {s.source_type for s in item.sources} == {"expert_interview", "cited_by_expert"}

    def test_uncertainty_means_needs_expert(self):
        unsure = interview_to_draft({**ANSWERS, "confidence": "unsure"}, item_id="PKG-101", tenant=TENANT,
                                    expert="user:anna", captured_at=NOW)
        scope_unsure = interview_to_draft({**ANSWERS, "scope": {"market": "not sure"}}, item_id="PKG-102",
                                          tenant=TENANT, expert="user:anna", captured_at=NOW)
        assert unsure.status == scope_unsure.status == PublicationStatus.NEEDS_EXPERT

    @pytest.mark.parametrize("answers,message", [
        ({k: v for k, v in ANSWERS.items() if k != "statement"}, "unanswered required"),
        ({**ANSWERS, "modality": "sort of"}, "is not one of"),
        ({**ANSWERS, "favourite_colour": "green"}, "unknown question"),
        ({**ANSWERS, "scope": {"planet": "mars"}}, "unknown scope dimension"),
    ])
    def test_invalid_interviews_are_refused(self, answers, message):
        with pytest.raises(ElicitationError, match=message):
            interview_to_draft(answers, item_id="PKG-103", tenant=TENANT, expert="user:anna", captured_at=NOW)

    def test_review_queue_shows_expert_needs_owner_and_disagreement(self):
        item = interview_to_draft({**ANSWERS, "confidence": "unsure"}, item_id="PKG-104", tenant=TENANT,
                                  expert="user:anna", captured_at=NOW)
        disagreement = record_disagreement(item, {"user:anna": "toys only", "user:ben": "all children's products"},
                                           disagreement_id="DIS-104", recorded_at=NOW)
        pack = load_pack(PACK).model_copy(update={"items": [*load_pack(PACK).items, item],
                                                  "disagreements": [disagreement]})
        entry = next(e for e in review_queue(pack) if e["key"] == "PKG-104@v1")
        assert entry["experts"] == ["user:anna"] and entry["review_owner"] == ""
        assert {"confirmation by a domain expert", "a named review owner",
                "resolution of expert disagreement DIS-104"} <= set(entry["needs"])

    def test_open_disagreement_blocks_publication_until_resolved_or_waived(self):
        pack = load_pack(PACK)
        target = pack.published_items()[0]
        base = dict(id="DIS-1", tenant=TENANT, item_id=target.id, experts=("user:a", "user:b"), recorded_at=NOW)
        blocked = pack.model_copy(update={"disagreements": [ExpertDisagreement(**base)]})
        assert [i.code for i in check_pack_issues(blocked)] == ["open_expert_disagreement"]
        resolved = pack.model_copy(update={"disagreements": [ExpertDisagreement(
            **base, status="resolved", resolution="b's wording adopted")]})
        assert check_pack_issues(resolved) == []
        with pytest.raises(ValidationError, match="approval reference"):
            ExpertDisagreement(**base, status="waived")
        with pytest.raises(ValidationError, match="records the resolution"):
            ExpertDisagreement(**base, status="resolved")

    def test_a_published_item_needs_a_named_reviewer(self):
        item = load_pack(PACK).published_items()[0]
        data = item.model_dump()
        data["reviewed_by"] = ""
        with pytest.raises(ValidationError):
            KnowledgeItem.model_validate(data)


# -- charters ------------------------------------------------------------------------------


class TestCharters:
    def test_bundled_charters_pass_with_denominators(self):
        for path in sorted((ROOT / "evals" / "charters").glob("*.yml")):
            result = run_charter(load_charter(path), root=ROOT)
            assert result["passed"], result
            assert all(m["denominator"] for m in result["metrics"].values())

    def _charter(self, **extra):
        data = dict(id="c", use_case="u", question_set="evals/knowledge_eval/packaging_labeling_cases.json",
                    split="development", label_owner="user:expert", threshold_source="expert panel 2026-10",
                    metrics={"overall_outcome_accuracy": {"min": 0.9}})
        data.update(extra)
        return Charter.model_validate(data)

    def test_a_missed_threshold_fails_the_use_case(self):
        result = run_charter(self._charter(metrics={"time_to_answer_p95_ms": {"max": 0.0}}), root=ROOT)
        assert result["passed"] is False

    def test_a_charter_without_label_owner_is_rejected(self, tmp_path):
        path = tmp_path / "c.yml"
        path.write_text(json.dumps({**self._charter().model_dump(), "label_owner": " "}), encoding="utf-8")
        with pytest.raises(CharterError, match="label owner"):
            load_charter(path)

    def test_unknown_metrics_and_wrong_splits_are_errors(self):
        with pytest.raises(CharterError, match="unknown metric"):
            run_charter(self._charter(metrics={"vibes": {"min": 1}}), root=ROOT)
        with pytest.raises(CharterError, match="split"):
            run_charter(self._charter(split="held_out"), root=ROOT)


# -- answer trust -------------------------------------------------------------------------------


SOUTH = {"customer": "CORNERSHOP", "brand": "OTHERBRAND", "market": "SOUTHLAND", "product": "PX-200",
         "product_family": "otc_tablets", "workflow": "label_review", "as_of": "2026-08-01T00:00:00Z"}
NORTH = {**SOUTH, "market": "NORTHLAND"}
GOOD = {"label.batch_code_present": True, "label.languages": ["sl"], "label.min_font_pt": 8,
        "label.storage_statement": "Store below 25 C", "label.allergen_statement": "x", "label.front_claims": []}


def _facts(values):
    return {k: Fact(value=v, evidence=tuple(EvidenceRef(**e) for e in EV)) for k, v in values.items()}


@pytest.fixture
async def trust(tmp_path):
    store = GovernanceStore(f"sqlite+aiosqlite:///{(tmp_path / 'g.sqlite').as_posix()}", blob_root=tmp_path / "b")
    await store.open()
    await publish_pack(store, load_pack(PACK), published_by="user:p", cases=load_cases(CASES))
    service = KnowledgeEvaluationService(store)

    async def decide(context, facts, claims=None):
        access = AccessContext.from_claims(claims or {"sub": "asker"})
        await service.evaluate(tenant=TENANT, domain=DOMAIN, access=access,
                               context=BusinessContext.model_validate(context), facts=_facts(facts))
        records = await service.decision_log(TENANT, DOMAIN)
        return records[-1]["record_id"] if records else None, access

    yield service, store, decide
    await store.close()


class TestAnswerTrust:
    async def test_each_root_cause_is_classified_from_its_bundle(self, trust):
        service, store, decide = trust
        cases = {}
        rid, who = await decide(NORTH, GOOD)
        cases["permission_filter"] = (rid, who, {})
        rid, who = await decide(SOUTH, GOOD)
        cases["expectation_mismatch"] = (rid, who, {})
        cases["model_error"] = (rid, who, {"answer_source": "llm_answer"})
        rid, who = await decide(SOUTH, {k: v for k, v in GOOD.items() if k != "label.storage_statement"})
        cases["extraction_error"] = (rid, who, {})
        rid, who = await decide({**SOUTH, "product_family": "shampoo", "market": "WESTLAND", "customer": "NOBODY"},
                                GOOD)
        cases["retrieval_miss"] = (rid, who, {})
        found = {}
        for expected in ("permission_filter", "expectation_mismatch", "model_error", "extraction_error",
                         "retrieval_miss"):
            record_id, reporter, extra = cases[expected]
            report = await service.report_distrust(TENANT, DOMAIN, record_id, reporter=reporter, reason="?", **extra)
            found[expected] = report["root_cause"]
        assert found == {k: k for k in found}
        stale_record, reporter = await decide(SOUTH, GOOD, {"sub": "late"})
        await publish_pack(store, load_pack(PACK).model_copy(update={"disclaimer": "revised"}), published_by="user:p",
                           cases=load_cases(CASES))
        stale = await service.report_distrust(TENANT, DOMAIN, stale_record, reporter=reporter, reason="old")
        assert stale["root_cause"] == "stale_source"
        assert set(found.values()) | {"stale_source"} <= set(ROOT_CAUSES)

    async def test_a_changed_definition_is_reported_before_staleness(self, tmp_path):
        store = GovernanceStore(f"sqlite+aiosqlite:///{(tmp_path / 'g.sqlite').as_posix()}", blob_root=tmp_path / "b")
        await store.open()
        try:
            cases = load_cases(CASES)
            await publish_pack(store, _pack_with_definition(1), published_by="user:p", cases=cases)
            service = KnowledgeEvaluationService(store)
            access = AccessContext.from_claims({"sub": "asker"})
            await service.evaluate(tenant=TENANT, domain=DOMAIN, access=access,
                                   context=BusinessContext.model_validate(SOUTH), facts=_facts(GOOD))
            record_id = (await service.decision_log(TENANT, DOMAIN))[0]["record_id"]
            await publish_pack(store, _pack_with_definition(2), published_by="user:p", cases=cases)
            report = await service.report_distrust(TENANT, DOMAIN, record_id, reporter=access, reason="wording")
        finally:
            await store.close()
        assert report["root_cause"] == "wrong_or_missing_definition"

    async def test_the_bundle_holds_only_what_the_reporter_may_read(self, trust):
        service, _, decide = trust
        record_id, _ = await decide(NORTH, GOOD, {"sub": "regulatory", "groups": ["regulatory-affairs"]})
        reporter = AccessContext.from_claims({"sub": "auditor"})
        report = await service.report_distrust(TENANT, DOMAIN, record_id, reporter=reporter, reason="check")
        bundle = report["bundle"]
        assert "PKG-015" not in json.dumps(bundle), "restricted knowledge never reaches the reporter"
        assert bundle["permission_filters"]["hidden_entries"] == 1
        assert bundle["question"]["facts"] == sorted(GOOD), "another user sees fact names, not values"
        assert bundle["replays_exactly"] is True and bundle["release"]["current"] is True

    def test_statistics_count_per_cause_and_time_to_fix(self):
        opened = NOW.isoformat()
        reports = [
            {"root_cause": "stale_source", "status": "fixed", "reported_at": opened,
             "fixed_at": (NOW + timedelta(hours=4)).isoformat()},
            {"root_cause": "stale_source", "status": "open", "reported_at": opened, "fixed_at": None},
            {"root_cause": "retrieval_miss", "status": "fixed", "reported_at": opened,
             "fixed_at": (NOW + timedelta(hours=1)).isoformat()},
        ]
        stats = trust_statistics(reports)
        assert stats["by_root_cause"]["stale_source"] == 2 and stats["open"] == 1
        assert stats["median_hours_to_fix"] == {**{c: None for c in ROOT_CAUSES},
                                                "stale_source": 4.0, "retrieval_miss": 1.0}


def _client(tmp_path: Path, claims: dict) -> TestClient:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = GovernanceStore(f"sqlite+aiosqlite:///{(tmp_path / 'g.sqlite').as_posix()}", blob_root=tmp_path / "b")
        await store.open()
        await publish_pack(store, load_pack(PACK), published_by="user:p", cases=load_cases(CASES))
        service = KnowledgeEvaluationService(store)
        await service.evaluate(tenant=TENANT, domain=DOMAIN, access=AccessContext.from_claims({"sub": "asker"}),
                               context=BusinessContext.model_validate(SOUTH), facts=_facts(GOOD))
        set_knowledge_service(service)
        try:
            yield
        finally:
            set_knowledge_service(None)
            await store.close()

    app = FastAPI(lifespan=lifespan)
    app.include_router(knowledge_routes.router)
    app.dependency_overrides[get_current_user] = lambda: claims
    return TestClient(app)


def test_trust_routes_enforce_who_may_report_and_track_fixes(tmp_path):
    claims = {"scope": "read write admin", "sub": "auditor", "tenant": TENANT}
    with _client(tmp_path, claims) as client:
        record_id = client.get(f"/knowledge/decisions/{DOMAIN}").json()["decisions"][0]["record_id"]
        claims.update(scope="read", sub="someone-else")
        denied = client.post(f"/knowledge/decisions/{DOMAIN}/{record_id}/distrust", json={"reason": "x"})
        assert denied.status_code == 403
        claims.update(sub="asker")
        reported = client.post(f"/knowledge/decisions/{DOMAIN}/{record_id}/distrust", json={"reason": "seems off"})
        assert reported.status_code == 200 and reported.json()["root_cause"] == "expectation_mismatch"
        report_id = reported.json()["report_id"]
        assert client.get(f"/knowledge/trust-reports/{DOMAIN}").status_code == 403
        claims.update(scope="admin", sub="auditor")
        listing = client.get(f"/knowledge/trust-reports/{DOMAIN}").json()
        assert listing["statistics"]["by_root_cause"]["expectation_mismatch"] == 1
        fixed = client.post(f"/knowledge/trust-reports/{DOMAIN}/{report_id}/fixed").json()
        assert fixed["status"] == "fixed" and fixed["fixed_by"] == "auditor"
        stats = client.get(f"/knowledge/trust-reports/{DOMAIN}").json()["statistics"]
        assert stats["open"] == 0 and stats["median_hours_to_fix"]["expectation_mismatch"] is not None
