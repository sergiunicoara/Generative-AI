"""DMN-style decision tables: a bounded representation, validated statically.

Some knowledge is not a requirement over facts but a mapping from inputs to
outputs ("which carton template applies for this market and pack size").  A
conjunction of predicates cannot hold it without approximation.  This module
represents such knowledge as a decision table with an explicit hit policy and
validates it statically; it deliberately has **no execution path** into the
evaluator (see ``docs/adr/0014-knowledge-formalism.md``). The one execution
path is ``evaluate_table``, used only by ontology action gates (ADR-0015).

Bounded on purpose:

* conditions use a fixed vocabulary (``any``, ``eq``, ``in``, numeric
  ``range`` with inclusive/exclusive bounds); there are no expressions, no
  FEEL, no functions.  Anything else is refused at load time;
* every input declares its domain (allowed values, or a numeric interval), so
  gaps and overlaps can be checked exhaustively over a finite partition;
* hit policies are ``unique`` (exactly one rule may match: overlaps and gaps
  are errors), ``first`` (order decides: gaps are errors, overlaps are
  reported as information), ``any`` (overlapping rules must give equal
  outputs) and ``collect`` (overlaps allowed; gaps reported).
"""

from __future__ import annotations

from itertools import product
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_MAX_CELLS = 20_000


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InputColumn(_Strict):
    name: str = Field(pattern=r"^[a-z][a-z0-9_.]{0,127}$")
    kind: Literal["enum", "number"]
    values: tuple[str, ...] = ()          # enum domain
    minimum: float | None = None          # number domain (inclusive)
    maximum: float | None = None

    @model_validator(mode="after")
    def _domain(self):
        if self.kind == "enum" and (not self.values or len(set(self.values)) != len(self.values)):
            raise ValueError(f"input {self.name}: an enum input needs distinct allowed values")
        if self.kind == "number" and (self.minimum is None or self.maximum is None or self.minimum > self.maximum):
            raise ValueError(f"input {self.name}: a numeric input needs minimum <= maximum")
        return self


class Condition(_Strict):
    op: Literal["any", "eq", "in", "range"] = "any"
    value: str | float | None = None
    values: tuple[str, ...] = ()
    low: float | None = None
    high: float | None = None
    low_inclusive: bool = True
    high_inclusive: bool = True


class TableRule(_Strict):
    id: str = Field(min_length=1, max_length=64)
    when: dict[str, Condition]
    then: dict[str, str | float | bool]


class DecisionTable(_Strict):
    id: str = Field(pattern=r"^[A-Z][A-Z0-9_\-]{0,63}$")
    version: int = Field(ge=1)
    tenant: str = Field(min_length=1)
    title: str = Field(min_length=1)
    hit_policy: Literal["unique", "first", "any", "collect"]
    inputs: tuple[InputColumn, ...] = Field(min_length=1, max_length=8)
    outputs: tuple[str, ...] = Field(min_length=1, max_length=8)
    rules: tuple[TableRule, ...] = Field(min_length=1, max_length=200)
    status: Literal["draft", "in_review", "published", "withdrawn"] = "draft"

    @model_validator(mode="after")
    def _shape(self):
        names = {column.name: column for column in self.inputs}
        if len(names) != len(self.inputs):
            raise ValueError("input names must be unique")
        ids = [rule.id for rule in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("rule ids must be unique")
        for rule in self.rules:
            unknown = set(rule.when) - set(names)
            if unknown:
                raise ValueError(f"rule {rule.id}: unknown inputs {sorted(unknown)}")
            if set(rule.then) != set(self.outputs):
                raise ValueError(f"rule {rule.id}: must set exactly the outputs {list(self.outputs)}")
            for name, condition in rule.when.items():
                _check_condition(rule.id, names[name], condition)
        return self

    @property
    def key(self) -> str:
        return f"{self.id}@v{self.version}"


def _check_condition(rule_id: str, column: InputColumn, condition: Condition) -> None:
    where = f"rule {rule_id}, input {column.name}"
    if condition.op == "any":
        return
    if column.kind == "enum":
        if condition.op == "eq":
            if not isinstance(condition.value, str) or condition.value not in column.values:
                raise ValueError(f"{where}: eq needs one of {list(column.values)}")
        elif condition.op == "in":
            if not condition.values or not set(condition.values) <= set(column.values):
                raise ValueError(f"{where}: in needs a non-empty subset of {list(column.values)}")
        else:
            raise ValueError(f"{where}: {condition.op} is not defined for an enum input")
        return
    if condition.op == "eq":
        if isinstance(condition.value, bool) or not isinstance(condition.value, (int, float)):
            raise ValueError(f"{where}: eq on a numeric input needs a number")
        return
    if condition.op != "range" or (condition.low is None and condition.high is None):
        raise ValueError(f"{where}: a numeric input takes eq or a range with at least one bound")
    if condition.low is not None and condition.high is not None and condition.low > condition.high:
        raise ValueError(f"{where}: range low exceeds high")


# ---------------------------------------------------------------------------
# static validation over a finite partition of the input domains


def _cells(column: InputColumn, table: DecisionTable) -> list:
    """Representative points: every enum value, or every boundary and the
    midpoints between boundaries of a numeric domain."""
    if column.kind == "enum":
        return list(column.values)
    points = {column.minimum, column.maximum}
    for rule in table.rules:
        condition = rule.when.get(column.name)
        if condition is None:
            continue
        for bound in (condition.low, condition.high, condition.value if condition.op == "eq" else None):
            if isinstance(bound, (int, float)) and column.minimum <= bound <= column.maximum:
                points.add(float(bound))
    ordered = sorted(points)
    mids = [(a + b) / 2 for a, b in zip(ordered, ordered[1:])]
    return sorted(set(ordered) | set(mids))


def _matches(condition: Condition | None, value) -> bool:
    if condition is None or condition.op == "any":
        return True
    if condition.op == "eq":
        return value == condition.value
    if condition.op == "in":
        return value in condition.values
    if condition.low is not None and (value < condition.low or (value == condition.low and not condition.low_inclusive)):
        return False
    if condition.high is not None and (value > condition.high or (value == condition.high and not condition.high_inclusive)):
        return False
    return True


class TableResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule_ids: tuple[str, ...] = ()
    outputs: dict[str, str | float | bool] = Field(default_factory=dict)
    error: str = ""


def evaluate_table(table: DecisionTable, inputs: dict) -> TableResult:
    """Evaluate one input row under the table's hit policy.

    Used only by ontology action gates (ADR-0015); knowledge evaluation still
    has no execution path into decision tables (ADR-0014). ``collect`` is not
    a single verdict and is reported as an error; a missing input matches only
    ``any`` conditions.
    """
    def matches(rule) -> bool:
        for column in table.inputs:
            condition = rule.when.get(column.name)
            value = inputs.get(column.name)
            if value is None:
                if condition is not None and condition.op != "any":
                    return False
                continue
            try:
                if not _matches(condition, value):
                    return False
            except TypeError:
                return False
        return True

    hits = [rule for rule in table.rules if matches(rule)]
    ids = tuple(rule.id for rule in hits)
    if not hits:
        return TableResult(error=f"no rule of {table.key} matches the inputs")
    if table.hit_policy == "collect":
        return TableResult(rule_ids=ids, error=f"{table.key} uses hit policy collect, not a single verdict")
    if table.hit_policy == "unique" and len(hits) > 1:
        return TableResult(rule_ids=ids, error=f"more than one rule of {table.key} matches under hit policy unique")
    if table.hit_policy == "any" and any(rule.then != hits[0].then for rule in hits[1:]):
        return TableResult(rule_ids=ids, error=f"matching rules of {table.key} disagree under hit policy any")
    chosen = hits[0]
    return TableResult(rule_ids=ids if table.hit_policy != "first" else (chosen.id,), outputs=dict(chosen.then))


class TableIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: Literal["overlap", "gap", "conflicting_outputs", "too_large"]
    severity: Literal["error", "info"]
    rules: tuple[str, ...] = ()
    example: dict[str, str | float] = Field(default_factory=dict)
    message: str


def validate_table(table: DecisionTable) -> list[TableIssue]:
    """Overlaps and gaps under the table's hit policy (empty = valid)."""
    axes = [_cells(column, table) for column in table.inputs]
    total = 1
    for axis in axes:
        total *= len(axis)
    if total > _MAX_CELLS:
        return [TableIssue(code="too_large", severity="error",
                           message=f"{total} input combinations exceed the {_MAX_CELLS} that can be checked")]
    issues: list[TableIssue] = []
    seen_overlaps: set[tuple[str, ...]] = set()
    gap_reported = False
    for point in product(*axes):
        values = {column.name: value for column, value in zip(table.inputs, point)}
        hits = [rule for rule in table.rules
                if all(_matches(rule.when.get(name), value) for name, value in values.items())]
        if not hits and not gap_reported:
            gap_reported = True
            issues.append(TableIssue(code="gap", severity="info" if table.hit_policy == "collect" else "error",
                                     example=values, message=f"no rule matches {values}"))
        if len(hits) > 1:
            ids = tuple(rule.id for rule in hits)
            if ids in seen_overlaps:
                continue
            seen_overlaps.add(ids)
            if table.hit_policy == "unique":
                issues.append(TableIssue(code="overlap", severity="error", rules=ids, example=values,
                                         message=f"rules {list(ids)} all match {values} under hit policy unique"))
            elif table.hit_policy == "any" and len({tuple(sorted(r.then.items())) for r in hits}) > 1:
                issues.append(TableIssue(code="conflicting_outputs", severity="error", rules=ids, example=values,
                                         message=f"rules {list(ids)} match {values} with different outputs"))
            elif table.hit_policy == "first":
                issues.append(TableIssue(code="overlap", severity="info", rules=ids, example=values,
                                         message=f"rules {list(ids)} overlap; the first one decides"))
    return issues


def table_errors(table: DecisionTable) -> list[TableIssue]:
    return [issue for issue in validate_table(table) if issue.severity == "error"]


__all__ = ["Condition", "DecisionTable", "InputColumn", "TableIssue", "TableRule", "table_errors", "validate_table"]
