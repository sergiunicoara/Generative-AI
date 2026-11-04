"""Validated metamodel for storage-independent domain semantics."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SemanticModelError(ValueError):
    """The canonical model is internally inconsistent."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Cardinality(StrictModel):
    min: int | None = Field(default=None, ge=0)
    max: int | None = Field(default=None, ge=0)
    exact: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def consistent(self) -> "Cardinality":
        if self.exact is not None and (self.min is not None or self.max is not None):
            raise ValueError("exact cannot be combined with min or max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min cannot exceed max")
        return self

    @property
    def minimum(self) -> int | None:
        return self.exact if self.exact is not None else self.min

    @property
    def maximum(self) -> int | None:
        return self.exact if self.exact is not None else self.max


class PropertySpec(StrictModel):
    datatype: Literal["string", "integer", "decimal", "boolean", "date", "dateTime"]
    required: bool = False
    key: bool = False
    cardinality: Cardinality = Field(default_factory=Cardinality)
    label: str = ""
    description: str = ""
    vocabulary: str | None = None

    @model_validator(mode="after")
    def required_cardinality(self) -> "PropertySpec":
        if self.required and self.cardinality.minimum is None:
            self.cardinality.min = 1
        return self


class MixinSpec(StrictModel):
    label: str = ""
    description: str = ""
    properties: dict[str, PropertySpec] = Field(default_factory=dict)


class TypeSpec(StrictModel):
    label: str = ""
    description: str = ""
    lpg_label: str = ""
    abstract: bool = False
    extends: str | None = None
    mixins: list[str] = Field(default_factory=list)
    properties: dict[str, PropertySpec] = Field(default_factory=dict)
    # Each entry is one composite key: the *combination* of these (required)
    # properties is unique, unlike `key: true` on a single property.
    composite_keys: list[list[str]] = Field(default_factory=list)
    deprecated: bool = False
    replaced_by: str | None = None


class QualifiedCardinality(StrictModel):
    """Cardinality counted only over relation targets of ``on_class`` (a
    subtype of the relation's declared target), e.g. "at least one of the
    supplied components is a SafetyComponent"."""

    on_class: str
    cardinality: Cardinality


class RelationSpec(StrictModel):
    source: str
    target: str
    label: str = ""
    description: str = ""
    lpg_type: str = ""
    cardinality: Cardinality = Field(default_factory=Cardinality)
    qualified: list[QualifiedCardinality] = Field(default_factory=list)
    # Relationship attributes: native on Neo4j relationships; carried on an
    # rdf:Statement reification class in OWL/SHACL (see the compiler).
    properties: dict[str, PropertySpec] = Field(default_factory=dict)
    deprecated: bool = False
    replaced_by: str | None = None


class ArtifactPaths(StrictModel):
    owl: str
    shacl: str
    neo4j: str
    diagnostics: str
    # Optional: a human-readable Markdown rendering of the same diagnostics
    # (roadmap "P0 — target capability matrix and loss diagnostics": a
    # machine-readable report already existed via `diagnostics`; this is the
    # human-readable counterpart). Optional so existing/test models that
    # don't declare it keep loading unchanged; `compile_to_disk` only writes
    # it when a model configures a path.
    diagnostics_report: str | None = None
    # Optional: a Mermaid classDiagram ERD/frame view (roadmap "P1 — visual
    # and tool-friendly semantic modelling"). Same optionality rationale as
    # `diagnostics_report` above.
    erd: str | None = None
    # Optional: the full rule x target capability matrix (every rule, every
    # target, typed outcome) -- the positive counterpart to `diagnostics`.
    capability_matrix: str | None = None
    # Optional: an ISO GQL graph-type rendering (interoperability surface).
    gql: str | None = None


Scalar = str | int | float | bool

_IDENTIFIER = r"^[A-Za-z_][A-Za-z0-9_]{0,63}$"
_SUBJECT = r"^(property|parameter|function):[A-Za-z_][A-Za-z0-9_]{0,63}$"


class ParameterSpec(StrictModel):
    """A typed action parameter (ontology actions, ADR-0015)."""

    datatype: Literal["string", "integer", "decimal", "boolean", "date", "dateTime"]
    required: bool = False
    description: str = ""
    allowed: list[Scalar] = Field(default_factory=list)
    max_length: int | None = Field(default=None, ge=1, le=4000)


class Criterion(StrictModel):
    """A submission criterion over the target object, a parameter or a function.

    ``subject`` is ``property:<name>``, ``parameter:<name>`` or ``function:<name>``.
    """

    id: str = Field(pattern=_IDENTIFIER)
    subject: str = Field(pattern=_SUBJECT)
    op: Literal["eq", "ne", "in", "not_in", "present", "absent", "min", "max", "min_length", "max_length"]
    value: Scalar | None = None
    values: list[Scalar] = Field(default_factory=list)
    message: str = ""

    @model_validator(mode="after")
    def operand_matches_op(self) -> "Criterion":
        if self.op in {"in", "not_in"} and not self.values:
            raise ValueError(f"criterion {self.id}: {self.op} needs values")
        if self.op in {"eq", "ne", "min", "max", "min_length", "max_length"} and self.value is None:
            raise ValueError(f"criterion {self.id}: {self.op} needs value")
        if self.op in {"min", "max", "min_length", "max_length"} and (
                isinstance(self.value, bool) or not isinstance(self.value, (int, float))):
            raise ValueError(f"criterion {self.id}: {self.op} needs a number")
        return self


class ActionEffect(StrictModel):
    """Set one property of the target object: to a literal, a parameter or the action time."""

    property: str = Field(pattern=_IDENTIFIER)
    value: Scalar | None = None
    from_parameter: str | None = Field(default=None, pattern=_IDENTIFIER)
    now: bool = False

    @model_validator(mode="after")
    def exactly_one_source(self) -> "ActionEffect":
        if sum((self.value is not None, self.from_parameter is not None, self.now)) != 1:
            raise ValueError(f"effect on {self.property}: set exactly one of value, from_parameter, now")
        return self


class DecisionGate(StrictModel):
    """A published decision table whose ``allow_output`` must be true for the action to proceed."""

    domain: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    table: str = Field(pattern=r"^[A-Z][A-Z0-9_\-]{0,63}$")
    inputs: dict[str, str] = Field(default_factory=dict)  # column -> subject
    allow_output: str = Field(min_length=1, max_length=128)


class ActionSpec(StrictModel):
    """A typed operation on one object of ``applies_to``, proposed, checked,
    approved when required, written and logged (ADR-0015)."""

    label: str = ""
    description: str = ""
    applies_to: str
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    submission_criteria: list[Criterion] = Field(default_factory=list)
    effects: list[ActionEffect] = Field(min_length=1)
    decision_gate: DecisionGate | None = None
    requires_approval: bool = True
    required_scope: str = Field(default="write", pattern=r"^[a-z][a-z0-9:_.-]{0,63}$")


class FunctionSpec(StrictModel):
    """Read-only logic bound to an object type: a property or a relation count."""

    label: str = ""
    description: str = ""
    applies_to: str
    kind: Literal["property", "relation_count"]
    property: str | None = None
    relation: str | None = None
    direction: Literal["out", "in"] = "out"

    @model_validator(mode="after")
    def operand_matches_kind(self) -> "FunctionSpec":
        if self.kind == "property" and not self.property:
            raise ValueError("a property function names its property")
        if self.kind == "relation_count" and not self.relation:
            raise ValueError("a relation_count function names its relation")
        return self


class SemanticModel(StrictModel):
    id: str
    version: str
    namespace: str
    label: str
    prefixes: dict[str, str]
    unknown_property_policy: Literal["allow", "reject"] = "reject"
    mixins: dict[str, MixinSpec] = Field(default_factory=dict)
    types: dict[str, TypeSpec]
    relations: dict[str, RelationSpec] = Field(default_factory=dict)
    actions: dict[str, ActionSpec] = Field(default_factory=dict)
    functions: dict[str, FunctionSpec] = Field(default_factory=dict)
    artifacts: ArtifactPaths
    source_path: Path | None = Field(default=None, exclude=True)
    source_lines: dict[str, int] = Field(default_factory=dict, exclude=True)

    def type_name(self, value: str) -> str | None:
        if value in self.types:
            return value
        upper = value.upper()
        return next((name for name, spec in self.types.items() if self.lpg_label(name, spec) == upper), None)

    @staticmethod
    def lpg_label(name: str, spec: TypeSpec) -> str:
        return spec.lpg_label or re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()

    def lpg_relation(self, name: str, spec: RelationSpec) -> str:
        return spec.lpg_type or re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()

    def ancestors(self, name: str) -> list[str]:
        result: list[str] = []
        current = self.types[name].extends
        while current and current in self.types:
            result.append(current)
            current = self.types[current].extends
        return result

    def is_subtype(self, candidate: str, expected: str) -> bool:
        candidate_name = self.type_name(candidate)
        expected_name = self.type_name(expected)
        return bool(candidate_name and expected_name and (
            candidate_name == expected_name or expected_name in self.ancestors(candidate_name)
        ))

    def effective_properties(self, name: str) -> dict[str, PropertySpec]:
        result: dict[str, PropertySpec] = {}
        lineage = list(reversed(self.ancestors(name))) + [name]
        for type_name in lineage:
            spec = self.types[type_name]
            for mixin in spec.mixins:
                result.update(self.mixins[mixin].properties)
            result.update(spec.properties)
        return result

    def composite_keys_for(self, name: str) -> list[list[str]]:
        """Composite keys declared on a type and its ancestors, in lineage order."""
        keys: list[list[str]] = []
        for type_name in [*reversed(self.ancestors(name)), name]:
            for members in self.types[type_name].composite_keys:
                if members not in keys:
                    keys.append(members)
        return keys

    def relations_for(self, name: str) -> dict[str, RelationSpec]:
        return {
            relation_name: relation
            for relation_name, relation in self.relations.items()
            if self.is_subtype(name, relation.source)
        }


def _source_lines(text: str) -> dict[str, int]:
    lines: dict[str, int] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        match = re.match(r"^(\s*)([A-Za-z_][A-Za-z0-9_-]*):(?:\s|$)", raw)
        if match:
            lines.setdefault(match.group(2), number)
    return lines


def _validate_semantics(model: SemanticModel) -> None:
    errors: list[str] = []
    if not re.fullmatch(r"\d+\.\d+\.\d+", model.version):
        errors.append("version must be semantic version major.minor.patch")
    if not model.namespace.endswith(("#", "/")):
        errors.append("namespace must end in '#' or '/'")
    for name, spec in model.types.items():
        if spec.extends and ":" not in spec.extends and spec.extends not in model.types:
            errors.append(f"type {name}: unknown parent {spec.extends}")
        for mixin in spec.mixins:
            if mixin not in model.mixins:
                errors.append(f"type {name}: unknown mixin {mixin}")
        if spec.replaced_by and spec.replaced_by not in model.types:
            errors.append(f"type {name}: unknown replacement {spec.replaced_by}")
    for name, relation in model.relations.items():
        if relation.source not in model.types:
            errors.append(f"relation {name}: unknown source type {relation.source}")
        if relation.target not in model.types:
            errors.append(f"relation {name}: unknown target type {relation.target}")
        if relation.replaced_by and relation.replaced_by not in model.relations:
            errors.append(f"relation {name}: unknown replacement {relation.replaced_by}")
    for start in model.types:
        seen: set[str] = set()
        current: str | None = start
        while current and current in model.types:
            if current in seen:
                errors.append(f"inheritance cycle involving {current}")
                break
            seen.add(current)
            current = model.types[current].extends
    # lineage walks below assume the hierarchy is acyclic and fully resolved
    if not errors:
        errors.extend(_validate_extensions(model))
    if not errors:
        errors.extend(_validate_actions(model))
    if errors:
        raise SemanticModelError("; ".join(dict.fromkeys(errors)))


_LITERAL_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "decimal": (int, float), "boolean": (bool,),
    "date": (str,), "dateTime": (str,),
}


def _literal_ok(datatype: str, value: Any) -> bool:
    if datatype in {"integer", "decimal"} and isinstance(value, bool):
        return False
    return isinstance(value, _LITERAL_TYPES[datatype])


def key_property(model: SemanticModel, type_name: str) -> str | None:
    keys = [name for name, prop in model.effective_properties(type_name).items() if prop.key]
    return keys[0] if len(keys) == 1 else None


def subject_datatype(model: SemanticModel, action: ActionSpec | None, applies_to: str, subject: str) -> str | None:
    """Datatype of a criterion/gate subject, or None if it does not resolve."""
    kind, name = subject.split(":", 1)
    if kind == "property":
        prop = model.effective_properties(applies_to).get(name)
        return prop.datatype if prop else None
    if kind == "parameter":
        param = action.parameters.get(name) if action else None
        return param.datatype if param else None
    function = model.functions.get(name)
    if function is None or not model.is_subtype(applies_to, function.applies_to):
        return None
    if function.kind == "relation_count":
        return "integer"
    prop = model.effective_properties(function.applies_to).get(function.property or "")
    return prop.datatype if prop else None


def _validate_actions(model: SemanticModel) -> list[str]:
    errors: list[str] = []
    for name, function in model.functions.items():
        where = f"function {name}"
        if not re.fullmatch(_IDENTIFIER, name):
            errors.append(f"{where}: name must be an identifier")
        if function.applies_to not in model.types:
            errors.append(f"{where}: unknown type {function.applies_to}")
            continue
        if function.kind == "property" and function.property not in model.effective_properties(function.applies_to):
            errors.append(f"{where}: {function.applies_to} has no property {function.property}")
        if function.kind == "relation_count":
            relation = model.relations.get(function.relation or "")
            if relation is None:
                errors.append(f"{where}: unknown relation {function.relation}")
            else:
                endpoint = relation.source if function.direction == "out" else relation.target
                if not model.is_subtype(function.applies_to, endpoint):
                    errors.append(f"{where}: {function.applies_to} is not the {function.direction} end of {function.relation}")
    for name, action in model.actions.items():
        where = f"action {name}"
        if not re.fullmatch(_IDENTIFIER, name):
            errors.append(f"{where}: name must be an identifier")
        spec = model.types.get(action.applies_to)
        if spec is None:
            errors.append(f"{where}: unknown type {action.applies_to}")
            continue
        if spec.abstract:
            errors.append(f"{where}: {action.applies_to} is abstract")
        key = key_property(model, action.applies_to)
        if key is None:
            errors.append(f"{where}: {action.applies_to} needs exactly one key property to address an object")
        properties = model.effective_properties(action.applies_to)
        for param_name, param in action.parameters.items():
            if not re.fullmatch(_IDENTIFIER, param_name) or param_name == "tenant":
                errors.append(f"{where}: invalid parameter name {param_name!r}")
            for allowed in param.allowed:
                if not _literal_ok(param.datatype, allowed):
                    errors.append(f"{where}: parameter {param_name} allows a value that is not {param.datatype}")
        subjects = [(f"criterion {c.id}", c.subject) for c in action.submission_criteria]
        if action.decision_gate:
            subjects += [(f"decision input {column}", subject) for column, subject in action.decision_gate.inputs.items()]
            for subject in action.decision_gate.inputs.values():
                if not re.fullmatch(_SUBJECT, subject):
                    errors.append(f"{where}: invalid decision input subject {subject!r}")
        for label, subject in subjects:
            if re.fullmatch(_SUBJECT, subject) and subject_datatype(model, action, action.applies_to, subject) is None:
                errors.append(f"{where}: {label} refers to unknown {subject}")
        if len({c.id for c in action.submission_criteria}) != len(action.submission_criteria):
            errors.append(f"{where}: duplicate criterion id")
        targets = [effect.property for effect in action.effects]
        if len(set(targets)) != len(targets):
            errors.append(f"{where}: a property is set by more than one effect")
        for effect in action.effects:
            prop = properties.get(effect.property)
            if prop is None:
                errors.append(f"{where}: {action.applies_to} has no property {effect.property}")
                continue
            if effect.property in {key, "tenant"}:
                errors.append(f"{where}: an effect may not change the key or tenant")
            if effect.now and prop.datatype != "dateTime":
                errors.append(f"{where}: now can only set a dateTime property")
            if effect.value is not None and not _literal_ok(prop.datatype, effect.value):
                errors.append(f"{where}: value for {effect.property} is not {prop.datatype}")
            if effect.from_parameter is not None:
                param = action.parameters.get(effect.from_parameter)
                if param is None:
                    errors.append(f"{where}: effect reads unknown parameter {effect.from_parameter}")
                elif param.datatype != prop.datatype:
                    errors.append(f"{where}: parameter {effect.from_parameter} is {param.datatype}, "
                                  f"{effect.property} is {prop.datatype}")
    return errors


def statement_class(relation_name: str) -> str:
    """The reification class carrying a relation's attributes in RDF."""
    return relation_name[:1].upper() + relation_name[1:] + "Statement"


def _validate_extensions(model: SemanticModel) -> list[str]:
    """Composite keys, qualified cardinalities and relation attributes."""
    errors: list[str] = []
    node_properties = {name for spec in model.types.values() for name in spec.properties}
    node_properties |= {name for mixin in model.mixins.values() for name in mixin.properties}
    for name, spec in model.types.items():
        effective = model.effective_properties(name)
        for members in spec.composite_keys:
            label = f"type {name}: composite key {members}"
            if len(members) < 2 or len(set(members)) != len(members):
                errors.append(f"{label} needs at least two distinct properties")
                continue
            for member in members:
                if member not in effective:
                    errors.append(f"{label}: unknown property {member}")
                elif not effective[member].cardinality.minimum:
                    errors.append(f"{label}: member {member} must be required")
    for name, relation in model.relations.items():
        for qualified in relation.qualified:
            if qualified.on_class not in model.types:
                errors.append(f"relation {name}: unknown qualified class {qualified.on_class}")
            elif not model.is_subtype(qualified.on_class, relation.target):
                errors.append(f"relation {name}: qualified class {qualified.on_class} is not a "
                              f"subtype of target {relation.target}")
            card = qualified.cardinality
            if card.minimum is None and card.maximum is None:
                errors.append(f"relation {name}: qualified cardinality on {qualified.on_class} sets no bound")
        if relation.properties and statement_class(name) in model.types:
            errors.append(f"relation {name}: reification class {statement_class(name)} clashes with a type")
        for attribute, prop in relation.properties.items():
            if prop.key:
                errors.append(f"relation {name}: attribute {attribute} cannot be a key")
            if attribute in node_properties or attribute in model.relations:
                errors.append(f"relation {name}: attribute {attribute} reuses a property or relation name")
    return errors


def load_model(path: str | Path) -> SemanticModel:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    payload: Any = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise SemanticModelError("semantic model must be a YAML mapping")
    try:
        model = SemanticModel.model_validate(payload)
    except ValueError as exc:
        raise SemanticModelError(str(exc)) from exc
    model.source_path = source.resolve()
    model.source_lines = _source_lines(text)
    _validate_semantics(model)
    return model
