"""Evaluate agents on the actions they propose, not only on their answers.

Each case names a request, the object it concerns and the expected outcome:
one declared action with its parameters, or a refusal. The agent sees the
request, the action catalog and the object, and returns a proposal; the
proposal is graded and also run through ``check_action``, so a proposal the
platform would block is reported as such. Runs offline with a scripted agent;
``LLMActionAgent`` adapts any ``generate(prompt, json_mode=True)`` model and is
tested here with a fake one only.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field

from graphrag.context_graph.decision_tables import DecisionTable
from graphrag.semantic_model.actions import ObjectSnapshot, check_action
from graphrag.semantic_model.models import SemanticModel, key_property

FailureKind = Literal["wrong_action", "wrong_object", "missing_parameter", "wrong_parameter",
                      "unexpected_parameter", "violates_checks", "should_have_refused", "wrongly_refused"]
REFUSE = "(refuse)"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CaseObject(_Strict):
    type: str
    key: str
    properties: dict[str, Any] = Field(default_factory=dict)
    relation_counts: dict[str, int] = Field(default_factory=dict)


class ExpectedOutcome(_Strict):
    action: str | None = None  # None: the agent must refuse
    parameters: dict[str, Any] = Field(default_factory=dict)
    # Parameters whose wording is free (a reason): present and non-empty is enough.
    free_text: list[str] = Field(default_factory=list)


class ActionCase(_Strict):
    id: str
    request: str
    object: CaseObject
    expected: ExpectedOutcome


class AgentProposal(_Strict):
    action: str | None = None
    object_key: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    refused: bool = False
    rationale: str = ""


class ActionAgent(Protocol):
    async def propose(self, request: str, catalog: list[dict[str, Any]], obj: dict[str, Any]) -> AgentProposal: ...


def load_cases(path: str | Path) -> list[ActionCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ActionCase.model_validate(case) for case in payload["cases"]]


def action_catalog(model: SemanticModel) -> list[dict[str, Any]]:
    return [{
        "action": name, "applies_to": spec.applies_to, "description": spec.description or spec.label,
        "requires_approval": spec.requires_approval,
        "parameters": {p: {"datatype": ps.datatype, "required": ps.required, "allowed": ps.allowed,
                           "description": ps.description} for p, ps in spec.parameters.items()},
        "criteria": [c.message or f"{c.subject} {c.op}" for c in spec.submission_criteria],
    } for name, spec in sorted(model.actions.items())]


def _grade(case: ActionCase, proposal: AgentProposal, model: SemanticModel, *, tenant: str, now: str,
           tables: Mapping[str, DecisionTable] | None) -> tuple[list[str], dict[str, Any] | None]:
    expected = case.expected
    acted = bool(proposal.action) and not proposal.refused
    check = None
    if acted:
        snapshot = ObjectSnapshot(case.object.type, case.object.key, dict(case.object.properties),
                                  dict(case.object.relation_counts))
        check = check_action(model, proposal.action, snapshot, proposal.parameters, tenant=tenant, now=now,
                             tables=tables).as_dict()
    if expected.action is None:
        return ([] if not acted else ["should_have_refused"]), check
    if not acted:
        return ["wrongly_refused"], None
    failures: list[str] = []
    if proposal.action != expected.action:
        failures.append("wrong_action")
    if proposal.object_key not in (None, case.object.key):
        failures.append("wrong_object")
    if proposal.action == expected.action:
        for name in set(expected.parameters) | set(expected.free_text):
            if proposal.parameters.get(name) in (None, ""):
                failures.append("missing_parameter")
            elif name not in expected.free_text and proposal.parameters[name] != expected.parameters[name]:
                failures.append("wrong_parameter")
        if set(proposal.parameters) - set(expected.parameters) - set(expected.free_text):
            failures.append("unexpected_parameter")
    if check is not None and not check["allowed"]:
        failures.append("violates_checks")
    return sorted(set(failures)), check


async def evaluate_agent(agent: ActionAgent, cases: list[ActionCase], model: SemanticModel, *,
                         tenant: str = "action-eval", now: str = "2026-01-01T00:00:00+00:00",
                         tables: Mapping[str, DecisionTable] | None = None) -> dict[str, Any]:
    catalog = action_catalog(model)
    results = []
    by_action: dict[str, Counter] = {}
    failure_counts: Counter = Counter()
    for case in cases:
        obj = {"type": case.object.type, "key": case.object.key,
               "key_property": key_property(model, case.object.type), **{"properties": case.object.properties}}
        proposal = await agent.propose(case.request, catalog, obj)
        failures, check = _grade(case, proposal, model, tenant=tenant, now=now, tables=tables)
        bucket = by_action.setdefault(case.expected.action or REFUSE, Counter())
        bucket["cases"] += 1
        bucket["passed"] += int(not failures)
        failure_counts.update(failures)
        results.append({"id": case.id, "passed": not failures, "failures": failures,
                        "expected": case.expected.model_dump(), "proposal": proposal.model_dump(),
                        "blocked_by_checks": bool(check is not None and not check["allowed"]),
                        "violations": [v["code"] for v in (check or {}).get("violations", [])]})
    passed = sum(r["passed"] for r in results)
    return {
        "total": len(results), "passed": passed, "pass_rate": passed / len(results) if results else 0.0,
        "by_action": {name: {"cases": c["cases"], "passed": c["passed"], "pass_rate": c["passed"] / c["cases"]}
                      for name, c in sorted(by_action.items())},
        "failures": dict(sorted(failure_counts.items())),
        "results": results,
    }


class ScriptedAgent:
    """Returns a fixed proposal per request: for offline tests and dataset self-checks."""

    def __init__(self, responses: Mapping[str, AgentProposal]):
        self._responses = dict(responses)

    async def propose(self, request: str, catalog: list[dict[str, Any]], obj: dict[str, Any]) -> AgentProposal:
        return self._responses.get(request, AgentProposal(refused=True, rationale="no scripted response"))


def oracle_agent(cases: list[ActionCase]) -> ScriptedAgent:
    """The expected outcomes as an agent: every case should pass (checks the dataset itself)."""
    return ScriptedAgent({case.request: (
        AgentProposal(action=case.expected.action, object_key=case.object.key,
                      parameters=case.expected.parameters | {name: f"oracle {name}: documented in the case"
                                                             for name in case.expected.free_text})
        if case.expected.action else AgentProposal(refused=True, rationale="expected refusal")) for case in cases})


_PROMPT = """You operate on a knowledge graph through declared actions only.
Choose at most one action from the catalog for the request, or refuse.
Refuse when no action fits, when the request is out of scope, or when the
object's state makes the action inappropriate. Never invent actions or parameters.
Answer with JSON only: {{"action": <name or null>, "parameters": {{...}}, "refuse": <true|false>, "rationale": "<short>"}}

Action catalog:
{catalog}

Object:
{obj}

Request:
{request}
"""


class LLMActionAgent:
    """Prompt a model with the catalog and the object; parse its JSON proposal."""

    def __init__(self, llm):
        self._llm = llm

    async def propose(self, request: str, catalog: list[dict[str, Any]], obj: dict[str, Any]) -> AgentProposal:
        prompt = _PROMPT.format(catalog=json.dumps(catalog, indent=1, default=str),
                                obj=json.dumps(obj, default=str), request=request)
        raw = await self._llm.generate(prompt, json_mode=True, temperature=0.0)
        try:
            payload = json.loads(raw)
            if payload.get("refuse") or not payload.get("action"):
                return AgentProposal(refused=True, rationale=str(payload.get("rationale", ""))[:500])
            return AgentProposal(action=str(payload["action"]), object_key=obj.get("key"),
                                 parameters=dict(payload.get("parameters") or {}),
                                 rationale=str(payload.get("rationale", ""))[:500])
        except (ValueError, TypeError, AttributeError):
            return AgentProposal(refused=True, rationale="unparseable model output")


__all__ = [
    "ActionAgent", "ActionCase", "AgentProposal", "LLMActionAgent", "ScriptedAgent",
    "action_catalog", "evaluate_agent", "load_cases", "oracle_agent",
]
