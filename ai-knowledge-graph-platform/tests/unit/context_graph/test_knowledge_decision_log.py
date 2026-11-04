"""Durable decision log, exact replay, explanation fidelity, re-review impact
and remediation hints (roadmap candidates, 2026-11-03)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes import knowledge as knowledge_routes
from graphrag.context_graph.knowledge import BusinessContext, Fact, KnowledgeItem
from graphrag.context_graph.knowledge_decisions import (
    cited_keys, fidelity, influence, needs_review, replays_exactly, review_impact,
)
from graphrag.context_graph.knowledge_evaluation import KnowledgeReleaseRef
from graphrag.context_graph.knowledge_release import (
    KnowledgePublicationError, ReleaseCatalog, check_pack_issues, load_cases, load_pack, publish_pack,
)
from graphrag.context_graph.knowledge_service import KnowledgeEvaluationService, set_knowledge_service
from graphrag.domains.energy.governance_store import GovernanceStore
from graphrag.enterprise.models import AccessContext

ROOT = Path(__file__).resolve().parents[3]
PACK = ROOT / "config" / "knowledge_packs" / "packaging_labeling.yml"
CASES = ROOT / "evals" / "knowledge_eval" / "packaging_labeling_cases.json"
TENANT, DOMAIN = "packaging_demo", "packaging_labeling"


def _case_inputs(case):
    return dict(context=BusinessContext.model_validate(case.context),
                facts={k: Fact.model_validate(v) for k, v in case.facts.items()},
                access=AccessContext.from_claims(case.principal),
                expect_applicable_knowledge=case.expect_applicable_knowledge,
                semantic_candidates=case.semantic_candidates)


@pytest.fixture
async def service(tmp_path):
    store = GovernanceStore(f"sqlite+aiosqlite:///{(tmp_path / 'g.sqlite').as_posix()}", blob_root=tmp_path / "b")
    await store.open()
    await publish_pack(store, load_pack(PACK), published_by="user:p", cases=load_cases(CASES))
    yield KnowledgeEvaluationService(store)
    await store.close()


async def _record_all_cases(service):
    for case in load_cases(CASES):
        if case.tenant == TENANT:
            await service.evaluate(tenant=TENANT, domain=DOMAIN, **_case_inputs(case))


# -- the log -------------------------------------------------------------------


class TestDecisionLog:
    async def test_every_evaluation_is_recorded_once_and_replays_exactly(self, service):
        await _record_all_cases(service)
        records = await service.decision_log(TENANT, DOMAIN)
        tenant_cases = [c for c in load_cases(CASES) if c.tenant == TENANT]
        assert len(records) == len({r["record_id"] for r in records}) <= len(tenant_cases)
        assert records, "evaluations are recorded"
        await _record_all_cases(service)  # same inputs again: idempotent
        assert len(await service.decision_log(TENANT, DOMAIN)) == len(records)
        for record in records:
            stored, exact = await service.replay(TENANT, DOMAIN, record["record_id"])
            assert exact, record["record_id"]
            assert stored["inputs"]["facts"] == record["inputs"]["facts"], "full facts with evidence are kept"

    async def test_record_can_be_disabled(self, service):
        case = load_cases(CASES)[0]
        await service.evaluate(tenant=TENANT, domain=DOMAIN, record=False, **_case_inputs(case))
        assert await service.decision_log(TENANT, DOMAIN) == []

    async def test_records_are_tenant_scoped(self, service):
        await _record_all_cases(service)
        record = (await service.decision_log(TENANT, DOMAIN))[0]
        assert await service.decision_log("other_tenant", DOMAIN) == []
        assert await service._store.knowledge_decision("other_tenant", record["record_id"]) is None

    async def test_the_same_inputs_under_another_access_context_are_a_separate_record(self, service):
        case = load_cases(CASES)[0]
        inputs = _case_inputs(case)
        await service.evaluate(tenant=TENANT, domain=DOMAIN, **inputs)
        await service.evaluate(tenant=TENANT, domain=DOMAIN, **{
            **inputs, "access": AccessContext.from_claims({"sub": "someone-else"})})
        assert len(await service.decision_log(TENANT, DOMAIN)) == 2

    async def test_a_tampered_record_does_not_replay(self, service):
        await _record_all_cases(service)
        record = (await service.decision_log(TENANT, DOMAIN))[0]
        catalog = await service.release_catalog(TENANT, DOMAIN, record["release_id"])
        tampered = {**record, "decision": {**record["decision"], "overall": "compliant"
                                           if record["decision"]["overall"] != "compliant" else "non_compliant"}}
        assert replays_exactly(record, catalog) and not replays_exactly(tampered, catalog)


# -- explanation fidelity ----------------------------------------------------------


class TestExplanationFidelity:
    def test_no_decisive_entry_is_missing_from_the_explanation_on_the_evaluation_set(self):
        pack = load_pack(PACK)
        release = KnowledgeReleaseRef(release_id="fidelity", domain=DOMAIN, version=0)
        totals = {"cases": 0, "entries": 0, "decisive": 0}
        for case in load_cases(CASES):
            inputs = _case_inputs(case)
            _, rows = influence(items=pack.published_items(), exceptions=pack.published_exceptions(),
                                tenant=case.tenant, domain=DOMAIN, release=release, **inputs)
            report = fidelity(rows)
            assert report["faithful"], (case.id, report["uncited_decisive"])
            totals["cases"] += 1
            totals["entries"] += report["entries"]
            totals["decisive"] += report["decisive"]
        assert totals["decisive"] > 0, "the counterfactual check is not vacuous"

    def test_a_failing_requirement_is_decisive_and_a_non_applicable_one_is_not(self):
        pack = load_pack(PACK)
        case = next(c for c in load_cases(CASES) if c.expected.overall == "non_compliant")
        release = KnowledgeReleaseRef(release_id="fidelity", domain=DOMAIN, version=0)
        base, rows = influence(items=pack.published_items(), exceptions=pack.published_exceptions(),
                               tenant=case.tenant, domain=DOMAIN, release=release, **_case_inputs(case))
        by_key = {row.key: row for row in rows}
        failing = [r for r in base.results if r.outcome.value == "fail"]
        assert failing and all(by_key[f"{r.item_id}@v{r.item_version}"].decisive for r in failing)
        not_applicable = [r for r in base.results if r.applicability.value == "does_not_apply"]
        assert not_applicable and not any(
            by_key[f"{r.item_id}@v{r.item_version}"].decisive for r in not_applicable)
        assert set(base.applicable) <= cited_keys(base)


# -- re-review impact ---------------------------------------------------------------


class TestReviewImpact:
    async def test_an_expiring_waiver_flags_the_decisions_that_relied_on_it(self, service):
        await _record_all_cases(service)
        now = datetime(2026, 8, 1, tzinfo=timezone.utc)
        impacts = await service.review_impact(TENANT, DOMAIN, window_end=datetime(2027, 8, 1, tzinfo=timezone.utc),
                                              now=now)
        lapsing = [i for i in impacts if i.expiring_exceptions]
        assert lapsing, "EXC-001 ends 2027-07-01, inside the window"
        assert all("EXC-001@v1" in i.expiring_exceptions and "exception_expiring" in i.triggers for i in lapsing)
        assert any(i.changed for i in lapsing), "once the waiver lapses the waived rule is evaluated again"
        for impact in lapsing:
            assert impact.then["overall"] and impact.now["overall"], "then and now are both reported"

    async def test_an_unchanged_release_and_view_leaves_every_decision_unchanged(self, service):
        await _record_all_cases(service)
        impacts = await service.review_impact(TENANT, DOMAIN, now=datetime(2026, 8, 1, tzinfo=timezone.utc))
        assert impacts and not any(i.changed for i in impacts) and needs_review(impacts) == []

    async def test_a_changed_release_and_review_due_items_are_reported(self, service):
        await _record_all_cases(service)
        records = await service.decision_log(TENANT, DOMAIN)
        catalog = await service.catalog(TENANT, DOMAIN)
        stricter = []
        for item in catalog.items:
            data = item.model_dump()
            if item.id == "PKG-007":
                data["review_due"] = datetime(2026, 1, 1, tzinfo=timezone.utc)
            stricter.append(KnowledgeItem.model_validate(data))
        revised = ReleaseCatalog(ref=KnowledgeReleaseRef(release_id="revised", domain=DOMAIN, version=99),
                                 items=tuple(i for i in stricter if i.id != "PKG-001"),
                                 exceptions=catalog.exceptions)
        impacts = review_impact(records, revised, now=datetime(2026, 8, 1, tzinfo=timezone.utc))
        assert all("release_changed" in i.triggers for i in impacts)
        assert any(i.changed and "PKG-001" in i.changed_items for i in impacts), "a removed rule changes outcomes"
        assert any("PKG-007" in i.review_due_items and "review_due" in i.triggers for i in impacts)

    def test_review_due_is_human_set_and_normalised_to_utc(self):
        item = load_pack(PACK).published_items()[0]
        data = item.model_dump()
        data["review_due"] = datetime(2027, 1, 1, 12, tzinfo=timezone(timedelta(hours=2)))
        assert KnowledgeItem.model_validate(data).review_due.utcoffset() == timedelta(0)
        assert item.review_due is None


# -- remediation hints ----------------------------------------------------------------


class TestRemediationHints:
    def test_every_gate_issue_has_code_path_and_hint(self):
        pack = load_pack(PACK).model_copy(deep=True)
        data = pack.items[0].model_dump()
        data["supersedes"] = ["PKG-404"]
        pack.items[0] = KnowledgeItem.model_validate(data)
        issues = check_pack_issues(pack)
        assert issues and all(i.code and i.path and i.hint and i.message for i in issues)
        dangling = next(i for i in issues if i.code == "supersedes_unknown_item")
        assert "PKG-404" in dangling.message and dangling.path.endswith("/supersedes")

    async def test_publication_errors_carry_structured_issues(self, tmp_path):
        store = GovernanceStore(f"sqlite+aiosqlite:///{(tmp_path / 'g.sqlite').as_posix()}", blob_root=tmp_path / "b")
        await store.open()
        try:
            with pytest.raises(KnowledgePublicationError) as caught:
                await publish_pack(store, load_pack(PACK), published_by="user:p")
        finally:
            await store.close()
        assert [i.code for i in caught.value.issues] == ["regression_cases_missing"]
        assert caught.value.failures == [caught.value.issues[0].message], "the text form is unchanged"

    async def test_evaluation_input_errors_name_path_and_fix(self, service):
        from mcp_server.capabilities.knowledge_evaluate import evaluate_knowledge_capability

        set_knowledge_service(service)
        try:
            result = await evaluate_knowledge_capability(
                TENANT, DOMAIN, {"market": "NORTHLAND", "planet": "mars"},
                facts={"label.min_font_pt": {"value": float("nan"), "evidence": [{"source_id": "S"}]}})
        finally:
            set_knowledge_service(None)
        issues = result["error"]["detail"]
        assert result["error"]["code"] == "invalid_request"
        assert all({"loc", "msg", "code", "path", "hint"} <= set(issue) for issue in issues)
        assert any(issue["path"] == "planet" and "Remove" in issue["hint"] for issue in issues)


# -- routes ------------------------------------------------------------------------------


def _client(tmp_path: Path, claims: dict) -> TestClient:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = GovernanceStore(f"sqlite+aiosqlite:///{(tmp_path / 'g.sqlite').as_posix()}", blob_root=tmp_path / "b")
        await store.open()
        await publish_pack(store, load_pack(PACK), published_by="user:p", cases=load_cases(CASES))
        service = KnowledgeEvaluationService(store)
        await _record_all_cases(service)
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


class TestDecisionRoutes:
    def test_log_replay_and_impact_are_admin_only_and_tenant_scoped(self, tmp_path):
        claims = {"scope": "read write admin", "sub": "auditor", "tenant": TENANT}
        with _client(tmp_path, claims) as client:
            listing = client.get(f"/knowledge/decisions/{DOMAIN}").json()["decisions"]
            assert listing and "inputs" not in listing[0] and "decision" not in listing[0]
            replayed = client.get(f"/knowledge/decisions/{DOMAIN}/{listing[0]['record_id']}/replay")
            assert replayed.status_code == 200 and replayed.json()["replays_exactly"] is True
            missing = client.get(f"/knowledge/decisions/{DOMAIN}/KR-{'0' * 24}/replay")
            assert missing.status_code == 404
            impact = client.post(f"/knowledge/decisions/{DOMAIN}/review-impact",
                                 json={"window_end": "2027-08-01T00:00:00Z"}).json()
            assert impact["evaluated"] == len(listing) and impact["affected"] >= 1
            claims["scope"] = "read write"
            assert client.get(f"/knowledge/decisions/{DOMAIN}").status_code == 403
            claims.update(scope="admin", tenant="other_tenant")
            assert client.get(f"/knowledge/decisions/{DOMAIN}").json() == {"decisions": []}
