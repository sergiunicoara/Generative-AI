"""Ontology actions and functions (ADR-0015): declaration checks, the pure
check, the propose -> approve -> guarded write flow, replay, and agent
evaluation on actions. Offline: temporary SQLite store and a fake graph."""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

from graphrag.context_graph.decision_tables import DecisionTable, evaluate_table
from graphrag.domains.energy.governance_store import GovernanceStore
from graphrag.evaluation.action_eval import (
    AgentProposal,
    LLMActionAgent,
    ScriptedAgent,
    evaluate_agent,
    load_cases,
    oracle_agent,
)
from graphrag.semantic_model import load_model
from graphrag.semantic_model.action_service import ActionService, replay
from graphrag.semantic_model.actions import ObjectSnapshot, check_action
from graphrag.semantic_model.models import SemanticModelError

ROOT = Path(__file__).resolve().parents[2]
AUTOMOTIVE = ROOT / "ontology" / "models" / "automotive-iatf-quality.yaml"
CASES = ROOT / "evals" / "actions" / "automotive_supplier_cases.json"
NOW = "2026-11-04T10:00:00+00:00"


def _model():
    return load_model(AUTOMOTIVE)


def _variant(tmp_path, mutate):
    payload = yaml.safe_load(AUTOMOTIVE.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "ontology" / "models" / "variant.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return load_model(path)


def _supplier(classification="approved", audits=2, **extra):
    return ObjectSnapshot("Supplier", "S-1", {"supplierId": "S-1", "classification": classification, **extra},
                          {"audits:in": audits})


# -- declarations --------------------------------------------------------------------------


def test_the_automotive_model_declares_actions_and_functions():
    model = _model()
    assert {"disqualify_supplier", "requalify_supplier", "set_reevaluation_frequency"} <= set(model.actions)
    assert {"audit_count", "classification"} <= set(model.functions)


@pytest.mark.parametrize("mutate,fragment", [
    (lambda m: m["actions"]["disqualify_supplier"]["effects"].append({"property": "supplierId", "value": "x"}),
     "may not change the key"),
    (lambda m: m["actions"]["disqualify_supplier"]["effects"].append({"property": "classification", "now": True}),
     "more than one effect"),
    (lambda m: m["actions"]["set_reevaluation_frequency"]["effects"][0].update(from_parameter="nope"),
     "unknown parameter"),
    (lambda m: m["actions"]["disqualify_supplier"].update(applies_to="ClientOrg"), "key property"),
    (lambda m: m["functions"]["audit_count"].update(direction="out"), "is not the out end"),
    (lambda m: m["actions"]["disqualify_supplier"]["submission_criteria"][0].update(subject="property:missing"),
     "unknown property:missing"),
])
def test_invalid_declarations_are_rejected_at_load(tmp_path, mutate, fragment):
    with pytest.raises(SemanticModelError, match=fragment):
        _variant(tmp_path, mutate)


# -- the pure check ------------------------------------------------------------------------


def test_a_passing_action_lists_its_changes_and_guard():
    check = check_action(_model(), "disqualify_supplier", _supplier(), {"reason": "late deliveries twice"},
                         tenant="acme", now=NOW)
    assert check.allowed and check.changes == {"classification": "disqualified", "updatedAt": NOW}
    assert check.guard == {"classification": "approved"}


def test_each_failed_criterion_is_named():
    check = check_action(_model(), "disqualify_supplier", _supplier("disqualified", audits=0), {"reason": "x"},
                         tenant="acme", now=NOW)
    codes = {(v.code, v.path) for v in check.violations}
    assert not check.allowed
    assert {("CRITERION_FAILED", "criteria.not_already_disqualified"), ("CRITERION_FAILED", "criteria.audit_evidence"),
            ("CRITERION_FAILED", "criteria.reason_given")} <= codes


def test_parameters_are_typed_and_bounded():
    model = _model()
    bad = check_action(model, "set_reevaluation_frequency", _supplier(), {"months": "6", "extra": 1},
                       tenant="acme", now=NOW)
    assert {v.code for v in bad.violations} == {"UNKNOWN_PARAMETER", "INVALID_PARAMETER"}
    too_long = check_action(model, "set_reevaluation_frequency", _supplier(), {"months": 60}, tenant="acme", now=NOW)
    assert [v.path for v in too_long.violations] == ["criteria.within_policy"]
    wrong = check_action(model, "requalify_supplier", _supplier("disqualified"),
                         {"classification": "excellent", "reason": "audit passed"}, tenant="acme", now=NOW)
    assert [v.code for v in wrong.violations] == ["PARAMETER_NOT_ALLOWED"]


def test_the_wrong_object_type_is_refused():
    snapshot = ObjectSnapshot("ClientOrg", "C-1", {}, {})
    check = check_action(_model(), "disqualify_supplier", snapshot, {"reason": "long enough reason"},
                         tenant="acme", now=NOW)
    assert [v.code for v in check.violations] == ["WRONG_OBJECT_TYPE"]


def _gated(tmp_path):
    def mutate(m):
        m["actions"]["disqualify_supplier"]["decision_gate"] = {
            "domain": "supplier_quality", "table": "DT-DISQUALIFY", "allow_output": "allowed",
            "inputs": {"audits": "function:audit_count"}}
    return _variant(tmp_path, mutate)


def _table(status="published"):
    return DecisionTable.model_validate({
        "id": "DT-DISQUALIFY", "version": 2, "tenant": "acme", "title": "Disqualification needs two audits",
        "hit_policy": "unique", "status": status,
        "inputs": [{"name": "audits", "kind": "number", "minimum": 0, "maximum": 1000}],
        "outputs": ["allowed"],
        "rules": [{"id": "R1", "when": {"audits": {"op": "range", "low": 0, "high": 2, "high_inclusive": False}},
                   "then": {"allowed": False}},
                  {"id": "R2", "when": {"audits": {"op": "range", "low": 2, "high": 1000}}, "then": {"allowed": True}}],
    })


def test_a_decision_table_gate_names_the_rule_that_denied(tmp_path):
    model = _gated(tmp_path)
    tables = {"DT-DISQUALIFY": _table()}
    denied = check_action(model, "disqualify_supplier", _supplier(audits=1), {"reason": "late deliveries again"},
                          tenant="acme", now=NOW, tables=tables)
    assert [v.code for v in denied.violations] == ["DECISION_DENIED"] and "R1" in denied.violations[0].message
    assert denied.decision["table"] == "DT-DISQUALIFY@v2"
    allowed = check_action(model, "disqualify_supplier", _supplier(audits=3), {"reason": "late deliveries again"},
                           tenant="acme", now=NOW, tables=tables)
    assert allowed.allowed and allowed.decision["rules"] == ["R2"]


def test_a_missing_or_unpublished_table_fails_closed(tmp_path):
    model = _gated(tmp_path)
    for tables in ({}, {"DT-DISQUALIFY": _table(status="draft")}):
        check = check_action(model, "disqualify_supplier", _supplier(audits=3), {"reason": "late deliveries again"},
                             tenant="acme", now=NOW, tables=tables)
        assert [v.code for v in check.violations] == ["DECISION_TABLE_UNAVAILABLE"]


def test_evaluate_table_applies_the_hit_policy():
    table = _table()
    assert evaluate_table(table, {"audits": 5}).outputs == {"allowed": True}
    assert evaluate_table(table, {"audits": None}).error.startswith("no rule")
    overlapping = table.model_copy(update={"rules": [*table.rules, table.rules[1].model_copy(update={"id": "R3"})]})
    assert "more than one" in evaluate_table(overlapping, {"audits": 5}).error


# -- the service ---------------------------------------------------------------------------


class _Graph:
    """Supplier nodes in memory; answers the snapshot read and the guarded write."""

    def __init__(self, nodes):
        self.nodes = nodes
        self.calls: list[tuple[str, dict]] = []
        self.begin_corpus_update = AsyncMock(return_value=None)
        self.complete_corpus_update = AsyncMock(return_value=8)

    async def run(self, cypher, **params):
        self.calls.append((cypher, params))
        node = self.nodes.get((params["tenant"], params["key"]))
        if "RETURN properties(n)" in cypher:
            if node is None:
                return []
            return [{"properties": {**node["properties"], "tenant": params["tenant"]}, "counts": node["counts"]}]
        if node is None:
            return [{"updated": 0}]
        for index in range(10):
            name = f"g{index}"
            if name in params:
                prop = cypher.split(f"= ${name}")[0].rsplit("n.", 1)[1].strip()
                if node["properties"].get(prop) != params[name]:
                    return [{"updated": 0}]
        for index in range(10):
            name = f"v{index}"
            if name in params:
                prop = cypher.split(f"${name}")[0].rsplit("n.", 1)[1].split("=")[0].strip()
                node["properties"][prop] = params[name]
        return [{"updated": 1}]

    def writes(self):
        return [c for c, _ in self.calls if " SET " in c]


@pytest.fixture
async def store(tmp_path):
    store = GovernanceStore(f"sqlite+aiosqlite:///{(tmp_path / 'g.sqlite').as_posix()}", blob_root=tmp_path / "b")
    await store.open()
    yield store
    await store.close()


def _graph(classification="approved", audits=2):
    return _Graph({("acme", "S-1"): {"properties": {"supplierId": "S-1", "classification": classification},
                                     "counts": {"audits:in": audits}}})


def _service(store, graph, clock=lambda: NOW):
    return ActionService(_model(), store, graph, clock=clock)


async def test_an_action_that_needs_approval_waits_and_writes_nothing(store):
    graph = _graph()
    record = await _service(store, graph).propose("acme", "disqualify_supplier", "S-1",
                                                  {"reason": "late deliveries twice"}, proposed_by="agent-1",
                                                  actor_type="agent")
    assert record["status"] == "pending_approval" and graph.writes() == []
    assert all(params["tenant"] == "acme" for _, params in graph.calls)
    assert record["snapshot"]["properties"] == {"supplierId": "S-1", "classification": "approved"}


async def test_the_proposer_cannot_approve_and_an_approval_writes_once_with_a_guard(store):
    graph = _graph()
    service = _service(store, graph)
    record = await service.propose("acme", "disqualify_supplier", "S-1", {"reason": "late deliveries twice"},
                                   proposed_by="agent-1")
    with pytest.raises(PermissionError):
        await service.decide("acme", record["proposal_id"], approve=True, decided_by="agent-1")
    applied = await service.decide("acme", record["proposal_id"], approve=True, decided_by="qa-lead")
    assert applied["status"] == "applied"
    assert graph.nodes[("acme", "S-1")]["properties"]["classification"] == "disqualified"
    [write] = graph.writes()
    assert "WHERE n.classification = $g0" in write and "datetime($v" in write
    graph.begin_corpus_update.assert_awaited_once()
    assert [e["to"] for e in applied["events"]] == ["pending_approval", "approved", "applied"]
    assert [e["actor"] for e in applied["events"]] == ["agent-1", "qa-lead", "qa-lead"]
    assert replay(_model(), applied)
    with pytest.raises(Exception):
        await service.decide("acme", record["proposal_id"], approve=True, decided_by="someone-else")


async def test_a_change_after_the_proposal_makes_it_stale(store):
    graph = _graph()
    service = _service(store, graph)
    record = await service.propose("acme", "disqualify_supplier", "S-1", {"reason": "late deliveries twice"},
                                   proposed_by="agent-1")
    graph.nodes[("acme", "S-1")]["properties"]["classification"] = "conditional"
    stale = await service.decide("acme", record["proposal_id"], approve=True, decided_by="qa-lead")
    assert stale["status"] == "stale" and graph.writes() == []
    assert stale["events"][-1]["detail"]["reason"] == "a property the decision depended on changed"


async def test_a_lost_race_at_write_time_is_stale_not_applied(store):
    graph = _graph()

    async def racing_run(cypher, **params):
        if " SET " in cypher:
            graph.calls.append((cypher, params))
            return [{"updated": 0}]
        return await _Graph.run(graph, cypher, **params)

    graph.run = racing_run
    service = _service(store, graph)
    record = await service.propose("acme", "disqualify_supplier", "S-1", {"reason": "late deliveries twice"},
                                   proposed_by="agent-1")
    result = await service.decide("acme", record["proposal_id"], approve=True, decided_by="qa-lead")
    assert result["status"] == "stale"


async def test_an_action_without_approval_applies_at_once_and_a_failing_one_is_recorded(store):
    graph = _graph()
    service = _service(store, graph)
    done = await service.propose("acme", "set_reevaluation_frequency", "S-1", {"months": 6}, proposed_by="u1")
    assert done["status"] == "applied"
    assert graph.nodes[("acme", "S-1")]["properties"]["reevaluationFrequencyMonths"] == 6
    refused = await service.propose("acme", "set_reevaluation_frequency", "S-1", {"months": 60}, proposed_by="u1")
    assert refused["status"] == "rejected" and len(graph.writes()) == 1
    assert [p["status"] for p in await service.proposals("acme")] == ["applied", "rejected"]


async def test_scope_tenant_and_unknown_objects(store):
    graph = _graph()
    service = _service(store, graph)
    with pytest.raises(PermissionError):
        await service.propose("acme", "disqualify_supplier", "S-1", {"reason": "late deliveries twice"},
                              proposed_by="u1", scopes=frozenset({"read"}))
    with pytest.raises(ValueError):
        await service.propose("other", "disqualify_supplier", "S-1", {"reason": "late deliveries twice"},
                              proposed_by="u1")
    record = await service.propose("acme", "disqualify_supplier", "S-1", {"reason": "late deliveries twice"},
                                   proposed_by="u1")
    with pytest.raises(ValueError):
        await service.decide("other", record["proposal_id"], approve=True, decided_by="u2")


async def test_functions_read_properties_and_relation_counts(store):
    service = _service(store, _graph(audits=4))
    assert (await service.call_function("acme", "audit_count", "S-1"))["value"] == 4
    assert (await service.call_function("acme", "classification", "S-1"))["value"] == "approved"


def test_replay_detects_a_changed_model_or_record():
    model = _model()
    snapshot = _supplier()
    check = check_action(model, "disqualify_supplier", snapshot, {"reason": "late deliveries twice"},
                         tenant="acme", now=NOW)
    from graphrag.semantic_model.action_service import model_digest

    record = {"model_sha256": model_digest(model), "type": "Supplier", "object_key": "S-1",
              "snapshot": {"properties": snapshot.properties, "relation_counts": snapshot.relation_counts},
              "action": "disqualify_supplier", "parameters": {"reason": "late deliveries twice"}, "tenant": "acme",
              "checked_at": NOW, "tables": {}, "check": check.as_dict()}
    assert replay(model, record)
    tampered = copy.deepcopy(record)
    tampered["check"]["allowed"] = False
    assert not replay(model, tampered)
    assert not replay(model, record | {"model_sha256": "0" * 64})


# -- capabilities --------------------------------------------------------------------------


async def test_capabilities_are_registered_with_scopes():
    from mcp_server.capabilities import build_registry
    from tests.unit.test_guarded_operations import ident

    registry = build_registry()
    names = {entry["qualified_name"] for entry in registry.discover(ident(scopes=frozenset({"read", "write"})))}
    assert {"kg.action.propose@1.0.0", "kg.action.decide@1.0.0", "kg.action.list@1.0.0",
            "kg.function.call@1.0.0"} <= names
    denied = await registry.call("kg.action.propose@1.0.0", {"model": "automotive-iatf-quality",
                                                             "action": "disqualify_supplier", "object_key": "S-1"},
                                 ident())
    assert denied.reason == "missing_scope"
    crossed = await registry.call("kg.function.call@1.0.0", {"model": "automotive-iatf-quality", "function": "x",
                                                             "object_key": "S-1", "tenant": "other"}, ident())
    assert crossed.reason == "tenant_mismatch"


# -- agent evaluation on actions -------------------------------------------------------------


async def test_the_dataset_is_consistent_with_the_model():
    cases = load_cases(CASES)
    report = await evaluate_agent(oracle_agent(cases), cases, _model())
    assert report["passed"] == report["total"] == len(cases), report["results"]
    assert report["by_action"]["(refuse)"]["cases"] == 5


async def test_scripted_mistakes_fail_the_case_that_covers_them():
    cases = load_cases(CASES)
    by_id = {c.id: c for c in cases}
    responses = {c.request: AgentProposal(action=c.expected.action, object_key=c.object.key,
                                          parameters=c.expected.parameters | {n: "documented reason"
                                                                              for n in c.expected.free_text})
                 if c.expected.action else AgentProposal(refused=True) for c in cases}
    responses[by_id["ACT-02"].request] = AgentProposal(action="disqualify_supplier", object_key="S-100",
                                                      parameters={"reason": "documented reason"})
    responses[by_id["ACT-03"].request] = AgentProposal(action="requalify_supplier", object_key="S-200",
                                                      parameters={"reason": "documented reason"})
    responses[by_id["ACT-05"].request] = AgentProposal(action="disqualify_supplier", object_key="S-300",
                                                      parameters={"reason": "I do not like them at all"})
    responses[by_id["ACT-04"].request] = AgentProposal(action="disqualify_supplier", object_key="S-100",
                                                      parameters={"reason": "delete means disqualify"})
    report = await evaluate_agent(ScriptedAgent(responses), cases, _model())
    failures = {r["id"]: r["failures"] for r in report["results"] if not r["passed"]}
    assert set(failures) == {"ACT-02", "ACT-03", "ACT-04", "ACT-05"}
    assert "wrong_action" in failures["ACT-02"]
    assert failures["ACT-03"] == ["missing_parameter", "violates_checks"]
    assert failures["ACT-04"] == ["should_have_refused"]
    assert failures["ACT-05"] == ["should_have_refused"]
    blocked = {r["id"]: r["blocked_by_checks"] for r in report["results"]}
    assert blocked["ACT-05"] is True and blocked["ACT-04"] is False, "ACT-04 would have passed the checks"
    assert report["by_action"]["(refuse)"] == {"cases": 5, "passed": 3, "pass_rate": 0.6}


async def test_the_llm_agent_parses_json_and_treats_garbage_as_a_refusal():
    class FakeLLM:
        def __init__(self, reply):
            self.reply, self.prompts = reply, []

        async def generate(self, prompt, **kwargs):
            self.prompts.append((prompt, kwargs))
            return self.reply

    cases = load_cases(CASES)[:1]
    good = FakeLLM('{"action": "disqualify_supplier", "parameters": {"reason": "two audits confirmed it"}, '
                   '"refuse": false, "rationale": "fits"}')
    report = await evaluate_agent(LLMActionAgent(good), cases, _model())
    assert report["passed"] == 1 and good.prompts[0][1]["json_mode"] is True
    assert "disqualify_supplier" in good.prompts[0][0] and "S-100" in good.prompts[0][0]
    garbage = await evaluate_agent(LLMActionAgent(FakeLLM("not json")), cases, _model())
    assert garbage["results"][0]["failures"] == ["wrongly_refused"]
