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
    deprecated: bool = False
    replaced_by: str | None = None


class RelationSpec(StrictModel):
    source: str
    target: str
    label: str = ""
    description: str = ""
    lpg_type: str = ""
    cardinality: Cardinality = Field(default_factory=Cardinality)
    deprecated: bool = False
    replaced_by: str | None = None


class ArtifactPaths(StrictModel):
    owl: str
    shacl: str
    neo4j: str
    diagnostics: str


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
    if errors:
        raise SemanticModelError("; ".join(dict.fromkeys(errors)))


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
