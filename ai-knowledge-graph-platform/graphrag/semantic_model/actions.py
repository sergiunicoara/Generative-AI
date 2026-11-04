"""Pure checks for ontology actions and functions (ADR-0015).

Given an object snapshot (its properties and relation counts, read from the
graph by the caller) and the proposed parameters, ``check_action`` decides
whether a declared action may proceed and what it would change. It is
deterministic in its inputs, including the action time, so a recorded
proposal can be replayed to the same verdict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from graphrag.context_graph.decision_tables import DecisionTable, evaluate_table
from graphrag.semantic_model.models import ActionSpec, Criterion, SemanticModel, key_property
from graphrag.semantic_model.runtime import SemanticMutationValidator, _datatype_ok


@dataclass(frozen=True)
class ObjectSnapshot:
    type_name: str
    key: str
    properties: dict[str, Any]
    # "<relation>:<out|in>" -> count, for the relations functions refer to
    relation_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Violation:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class ActionCheck:
    allowed: bool
    violations: tuple[Violation, ...]
    changes: dict[str, Any]
    guard: dict[str, Any]
    decision: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "violations": [asdict(v) for v in self.violations],
                "changes": self.changes, "guard": self.guard, "decision": self.decision}


class UnknownFunctionError(KeyError):
    pass


def counted_relations(model: SemanticModel, type_name: str) -> list[tuple[str, str]]:
    """(relation, direction) pairs that functions on ``type_name`` count."""
    return sorted({(f.relation, f.direction) for f in model.functions.values()
                   if f.kind == "relation_count" and f.relation and model.is_subtype(type_name, f.applies_to)})


def function_value(model: SemanticModel, name: str, snapshot: ObjectSnapshot) -> Any:
    function = model.functions.get(name)
    if function is None or not model.is_subtype(snapshot.type_name, function.applies_to):
        raise UnknownFunctionError(name)
    if function.kind == "property":
        return snapshot.properties.get(function.property or "")
    return int(snapshot.relation_counts.get(f"{function.relation}:{function.direction}", 0))


def _subject(model: SemanticModel, subject: str, snapshot: ObjectSnapshot, parameters: Mapping[str, Any]) -> Any:
    kind, name = subject.split(":", 1)
    if kind == "property":
        return snapshot.properties.get(name)
    if kind == "parameter":
        return parameters.get(name)
    return function_value(model, name, snapshot)


def _criterion_holds(criterion: Criterion, value: Any) -> bool:
    op = criterion.op
    if op == "present":
        return value is not None and value != ""
    if op == "absent":
        return value is None or value == ""
    if op == "eq":
        return value == criterion.value
    if op == "ne":
        return value != criterion.value
    if op == "in":
        return value in criterion.values
    if op == "not_in":
        return value not in criterion.values
    if value is None:
        return False
    if op in {"min_length", "max_length"}:
        length = len(str(value))
        return length >= criterion.value if op == "min_length" else length <= criterion.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value >= criterion.value if op == "min" else value <= criterion.value


def _check_parameters(action: ActionSpec, parameters: Mapping[str, Any]) -> list[Violation]:
    violations = [Violation("UNKNOWN_PARAMETER", f"parameters.{name}", f"{name!r} is not a parameter of this action")
                  for name in sorted(set(parameters) - set(action.parameters))]
    for name, spec in action.parameters.items():
        value = parameters.get(name)
        if value is None:
            if spec.required:
                violations.append(Violation("REQUIRED_PARAMETER", f"parameters.{name}", f"{name!r} is required"))
            continue
        if not _datatype_ok(spec.datatype, value):
            violations.append(Violation("INVALID_PARAMETER", f"parameters.{name}", f"{name!r} must be {spec.datatype}"))
            continue
        if spec.allowed and value not in spec.allowed:
            violations.append(Violation("PARAMETER_NOT_ALLOWED", f"parameters.{name}",
                                        f"{name!r} must be one of {spec.allowed}"))
        if spec.max_length is not None and isinstance(value, str) and len(value) > spec.max_length:
            violations.append(Violation("PARAMETER_TOO_LONG", f"parameters.{name}",
                                        f"{name!r} exceeds {spec.max_length} characters"))
    return violations


def check_action(
    model: SemanticModel,
    action_name: str,
    snapshot: ObjectSnapshot,
    parameters: Mapping[str, Any],
    *,
    tenant: str,
    now: str,
    tables: Mapping[str, DecisionTable] | None = None,
) -> ActionCheck:
    """Decide whether ``action_name`` may run on ``snapshot`` with ``parameters``."""
    action = model.actions.get(action_name)
    if action is None:
        return ActionCheck(False, (Violation("UNKNOWN_ACTION", "action", f"no action {action_name!r}"),), {}, {})
    if not model.is_subtype(snapshot.type_name, action.applies_to):
        return ActionCheck(False, (Violation("WRONG_OBJECT_TYPE", "object",
                                             f"{action_name} applies to {action.applies_to}, not {snapshot.type_name}"),),
                           {}, {})
    violations = _check_parameters(action, parameters)
    parameters_ok = not violations
    if parameters_ok:
        for criterion in action.submission_criteria:
            if not _criterion_holds(criterion, _subject(model, criterion.subject, snapshot, parameters)):
                violations.append(Violation("CRITERION_FAILED", f"criteria.{criterion.id}",
                                            criterion.message or f"{criterion.subject} fails {criterion.op}"))

    decision = None
    gate = action.decision_gate
    if gate is not None and parameters_ok:
        table = (tables or {}).get(gate.table)
        if table is None or table.status != "published":
            violations.append(Violation("DECISION_TABLE_UNAVAILABLE", "decision_gate",
                                        f"published decision table {gate.table} is not available"))
        else:
            inputs = {column: _subject(model, subject, snapshot, parameters) for column, subject in gate.inputs.items()}
            result = evaluate_table(table, inputs)
            decision = {"table": table.key, "inputs": inputs, "rules": list(result.rule_ids),
                        "outputs": result.outputs, "error": result.error}
            if result.error:
                violations.append(Violation("DECISION_TABLE_ERROR", "decision_gate", result.error))
            elif result.outputs.get(gate.allow_output) is not True:
                violations.append(Violation("DECISION_DENIED", "decision_gate",
                                            f"{table.key} rule {', '.join(result.rule_ids)} does not allow this action"))

    changes: dict[str, Any] = {}
    for effect in action.effects:
        if effect.now:
            changes[effect.property] = now
        elif effect.from_parameter is not None:
            changes[effect.property] = parameters.get(effect.from_parameter)
        else:
            changes[effect.property] = effect.value
    declared = model.effective_properties(action.applies_to)
    if "updatedAt" in declared and "updatedAt" not in changes and declared["updatedAt"].datatype == "dateTime":
        changes["updatedAt"] = now
    if parameters_ok:
        post_state = {k: v for k, v in snapshot.properties.items() if k in declared} | changes
        for violation in SemanticMutationValidator(model).validate_node(snapshot.type_name, post_state, tenant=tenant):
            violations.append(Violation(f"POST_STATE_{violation.code}", violation.path, violation.message))

    guarded = {effect.property for effect in action.effects}
    guarded |= {c.subject.split(":", 1)[1] for c in action.submission_criteria if c.subject.startswith("property:")}
    if gate is not None:
        guarded |= {s.split(":", 1)[1] for s in gate.inputs.values() if s.startswith("property:")}
    key = key_property(model, action.applies_to)
    guarded.discard(key or "")
    guard = {name: snapshot.properties.get(name) for name in sorted(guarded)}
    return ActionCheck(not violations, tuple(violations), changes, guard, decision)


__all__ = [
    "ActionCheck",
    "ObjectSnapshot",
    "UnknownFunctionError",
    "Violation",
    "check_action",
    "counted_relations",
    "function_value",
]
