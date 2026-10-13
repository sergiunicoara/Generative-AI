"""IATF follow-ups: mapping review, metrics, dashboard tab, CronJob example, opt-in retrieval/prompt flags."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes import iatf_audit as routes
from graphrag.dashboard.tabs import iatf_audit as tab
from graphrag.domains.automotive import audit, metrics
from graphrag.domains.automotive import signoff as so
from graphrag.retrieval.answer_policy import ANSWER_LANGUAGE_RULE, answer_prompt

ROOT = Path(__file__).resolve().parents[2]


class _FakeNeo4j:
    def __init__(self):
        self.head, self.entries, self.calls = {"hash": so.GENESIS, "seq": 0}, [], []

    async def run(self, cypher, **p):
        self.calls.append((cypher, p))
        if cypher.startswith("CREATE CONSTRAINT"):
            return []
        if cypher is so.HEAD:
            return [dict(self.head)]
        if cypher is so.APPEND:
            self.head = {"hash": p["hash"], "seq": p["seq"]}
            self.entries.append(p)
            return [{"id": p["id"]}]
        if cypher is so.LIST:
            return [{"entry": e} for e in self.entries if p["clause"] in (None, e["clause"])]
        raise AssertionError(cypher)


@pytest.fixture(autouse=True)
def _schema_flag():
    so.SignoffStore._schema_ready = False
    yield
    so.SignoffStore._schema_ready = False


def _client(sub="qe-1"):
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_current_user] = lambda: {"scope": "read write", "sub": sub, "tenant": "automotive"}
    return TestClient(app)


# ── mapping review ──────────────────────────────────────────────────────────────────────────────────────────────
def test_mapping_digest_tracks_prefixes_and_keywords_only():
    c = audit.Clause("8.7", "NC", ("IL-NC",), ("neconform",))
    assert audit.mapping_digest(c) == audit.mapping_digest(audit.Clause("8.7", "renamed", ("IL-NC",), ("neconform",)))
    assert audit.mapping_digest(c) != audit.mapping_digest(audit.Clause("8.7", "NC", ("IL-NC", "REG-NC"),
                                                                        ("neconform",)))


def test_mapping_state_transitions():
    c = audit.Clause("8.7", "NC", ("IL-NC",), ("neconform",))
    digest = audit.mapping_digest(c)
    assert audit.mapping_state(c, None, "proposed")["status"] == "proposed"
    ok = {"decision": "mapping_confirmed", "evidence_digest": digest, "reviewer": "qe", "signed_at": "t"}
    assert audit.mapping_state(c, ok, "proposed")["status"] == "confirmed"
    assert audit.mapping_state(c, {**ok, "decision": "mapping_rejected"}, "proposed")["status"] == "rejected"
    assert audit.mapping_state(c, {**ok, "evidence_digest": "old"}, "proposed")["status"] == "changed_since_review"


async def test_latest_by_clause_separates_evidence_signoffs_from_mapping_reviews():
    store = so.SignoffStore(_FakeNeo4j())
    await store.append(tenant="t", clause="8.7", decision="accepted", reviewer="a", evidence_digest="e")
    await store.append(tenant="t", clause="8.7", decision="mapping_confirmed", reviewer="qe", evidence_digest="m")
    assert (await store.latest_by_clause("t"))["8.7"]["decision"] == "accepted"
    assert (await store.latest_by_clause("t", kind="mapping"))["8.7"]["decision"] == "mapping_confirmed"
    assert (await store.verify("t"))["valid"] is True


def test_mapping_review_route_records_the_token_subject_and_the_view_shows_confirmed():
    neo = _FakeNeo4j()
    with patch.object(routes, "get_neo4j", return_value=neo):
        client = _client()
        entry = client.post("/audit/iatf/clauses/8.7/mapping-review",
                            json={"decision": "mapping_confirmed", "reviewer": "forged"}).json()
        view = client.get("/audit/iatf/clauses").json()
        assert client.post("/audit/iatf/clauses/99.9/mapping-review",
                           json={"decision": "mapping_confirmed"}).status_code == 404
        assert client.post("/audit/iatf/clauses/8.7/signoff",
                           json={"decision": "mapping_confirmed"}).status_code == 422
    assert entry["reviewer"] == "qe-1" and entry["decision"] == "mapping_confirmed"
    mapping = {c["clause"]: c["mapping"]["status"] for c in view["clauses"]}
    assert mapping["8.7"] == "confirmed" and mapping["8.4"] == "proposed"


def test_export_rows_include_mapping_status():
    view = audit.audit_view(now=None)
    rows = audit.export_rows(view)
    assert "mapping_status" in audit.CSV_FIELDS and {r["mapping_status"] for r in rows} == {"proposed"}


# ── metrics ─────────────────────────────────────────────────────────────────────────────────────────────────────
def test_metrics_record_counts_mapping_conflicts_and_chain():
    prometheus = pytest.importorskip("prometheus_client")
    view = audit.audit_view()
    metrics.record_view("metrics-test", view)
    metrics.record_chain("metrics-test", {"valid": False})
    reg = prometheus.REGISTRY
    assert reg.get_sample_value("graphrag_iatf_clauses", {"tenant": "metrics-test", "status": "missing"}) == 3
    assert reg.get_sample_value("graphrag_iatf_clause_mappings",
                                {"tenant": "metrics-test", "status": "proposed"}) == len(view["clauses"])
    assert reg.get_sample_value("graphrag_iatf_clauses_with_conflicts", {"tenant": "metrics-test"}) >= 1
    assert reg.get_sample_value("graphrag_iatf_signoff_chain_valid", {"tenant": "metrics-test"}) == 0


def test_metrics_never_raise_on_a_malformed_view():
    metrics.record_view("metrics-test", {"counts": None, "clauses": None})


# ── dashboard tab ───────────────────────────────────────────────────────────────────────────────────────────────
def test_tab_renders_rows_chain_state_and_stale_signoff(monkeypatch):
    view = audit.audit_view()
    c84 = next(c for c in view["clauses"] if c["clause"] == "8.4")
    entries = [{"seq": 1, "clause": "8.4", "decision": "accepted", "reviewer": "aud",
                "evidence_digest": "old-digest"}]

    def fake_get(path, params=None):
        return view if path.endswith("/clauses") else {"entries": entries, "chain": {"valid": True}}

    monkeypatch.setattr(tab, "_get", fake_get)
    rendered = str(tab.render("automotive"))
    assert "verified" in rendered and "IATF 16949 Audit" in rendered
    rows = {r["clause"]: r for r in tab.clause_rows(view, tab._latest_evidence_signoffs(entries))}
    assert rows["8.4"]["signoff"] == "accepted by aud (evidence changed since)"
    assert rows["8.5.4"]["status"] == "✗ missing" and c84["evidence"]


def test_tab_shows_error_and_signoff_posts_the_decision(monkeypatch):
    monkeypatch.setattr(tab, "_get", lambda path, params=None: {"_http_error": "HTTP 503"})
    assert "unavailable" in str(tab.render("automotive"))
    posted = {}
    monkeypatch.setattr(tab, "_post", lambda path, json=None: posted.update(path=path, json=json) or {"seq": 4})
    assert tab.sign_off("iatf-more-btn", [], []) == "Select a clause first."
    msg = tab.sign_off("iatf-more-btn", [{"clause": "10.2"}], [])
    assert posted == {"path": "/audit/iatf/clauses/10.2/signoff", "json": {"decision": "needs_more_evidence"}}
    assert "#4" in msg and tab.sign_off("other", [{"clause": "x"}], []) == ""


# ── deployment example ──────────────────────────────────────────────────────────────────────────────────────────
def test_cronjob_example_parses_is_hardened_and_not_in_kustomization():
    path = ROOT / "deploy" / "kubernetes" / "iatf-freshness-cronjob.example.yaml"
    docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
    assert [d["kind"] for d in docs] == ["CronJob", "CronJob"]
    for d in docs:
        pod = d["spec"]["jobTemplate"]["spec"]["template"]["spec"]
        assert pod["securityContext"]["runAsNonRoot"] is True
        assert all(c["securityContext"]["readOnlyRootFilesystem"] for c in pod["containers"])
    key_env = docs[1]["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]["env"][0]
    assert key_env["valueFrom"]["secretKeyRef"]["name"] == "graphrag-iatf-checkpoint"
    kustomization = (ROOT / "deploy" / "kubernetes" / "kustomization.yaml").read_text(encoding="utf-8")
    assert "iatf-freshness-cronjob" not in kustomization


# ── opt-in flags (off by default, not golden-validated) ─────────────────────────────────────────────────────────
def test_answer_language_rule_is_opt_in():
    assert ANSWER_LANGUAGE_RULE not in answer_prompt({})
    with_rule = answer_prompt({"answer_language_rule_enabled": True})
    assert ANSWER_LANGUAGE_RULE in with_rule and "{context}" in with_rule and "{question}" in with_rule


def test_new_retrieval_flags_default_off():
    from graphrag.core.config import get_settings

    retrieval = get_settings().retrieval
    assert retrieval.get("entity_description_supersession_filter_enabled") is False
    assert retrieval.get("answer_language_rule_enabled") is False


async def test_entity_neighbor_query_carries_the_description_filter_parameter():
    from graphrag.graph.neo4j_client import Neo4jClient

    captured = {}

    class Probe(Neo4jClient):
        def __init__(self):
            pass

        async def run(self, cypher, **params):
            captured.update(cypher=cypher, params=params)
            return []

    await Probe().get_entity_neighbors(["c1"], tenant="t", include_superseded=False,
                                       hide_superseded_descriptions=True)
    assert captured["params"]["hide_superseded_descriptions"] is True
    assert "description_superseded" in captured["cypher"] and "e.source_doc_id" in captured["cypher"]
    await Probe().get_entity_neighbors(["c1"], tenant="t")
    assert captured["params"]["hide_superseded_descriptions"] is False


async def test_gremlin_refuses_supersession_filters_instead_of_ignoring_them():
    from graphrag.graph.gremlin_client import GremlinBackend

    backend = GremlinBackend.__new__(GremlinBackend)
    with pytest.raises(NotImplementedError, match="superseded"):
        await backend.get_entity_neighbors(["c1"], include_superseded=False)
    with pytest.raises(NotImplementedError, match="superseded"):
        await backend.get_entity_neighbors(["c1"], hide_superseded_descriptions=True)
