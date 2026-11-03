"""Runtime controls for semantic rules that database DDL cannot express."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from graphrag.semantic_model.models import SemanticModel


@dataclass(frozen=True)
class MutationViolation:
    code: str
    path: str
    message: str


class MutationValidationError(ValueError):
    def __init__(self, violations: list[MutationViolation]) -> None:
        self.violations = violations
        super().__init__("; ".join(item.message for item in violations))

    def as_dict(self) -> dict[str, Any]:
        return {"valid": False, "violations": [asdict(item) for item in self.violations]}


_PYTHON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "decimal": (int, float, Decimal),
    "boolean": (bool,), "date": (date, str), "dateTime": (datetime, str),
}


def _is_iso_temporal(value: Any, datatype: str) -> bool:
    if not isinstance(value, str):
        if datatype == "dateTime":
            return isinstance(value, datetime)
        return isinstance(value, date) and not isinstance(value, datetime)
    try:
        if datatype == "dateTime":
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _datatype_ok(datatype: str, value: Any) -> bool:
    if datatype in {"date", "dateTime"}:
        return _is_iso_temporal(value, datatype)
    if datatype in {"integer", "decimal"} and isinstance(value, bool):
        return False
    if datatype == "boolean":
        return type(value) is bool
    return isinstance(value, _PYTHON_TYPES[datatype])


class SemanticMutationValidator:
    """Validate LPG mutations before they reach the shared graph write path.

    Relationship-count checks accept an ``existing_count`` supplied by the
    caller. Such checks must run in the same database transaction as the write
    in production; a separate check-then-write can race and is not represented
    here as atomic enforcement.
    """

    def __init__(self, model: SemanticModel) -> None:
        self.model = model

    def validate_node(self, type_name: str, properties: dict[str, Any], *, tenant: str) -> list[MutationViolation]:
        violations: list[MutationViolation] = []
        resolved = self.model.type_name(type_name)
        if not resolved:
            return [MutationViolation("UNKNOWN_TYPE", "type", f"Unknown semantic type {type_name!r}")]
        if not tenant.strip():
            violations.append(MutationViolation("TENANT_REQUIRED", "tenant", "A trusted tenant is required"))
        spec = self.model.types[resolved]
        if spec.abstract:
            violations.append(MutationViolation("ABSTRACT_TYPE", "type", f"Abstract type {resolved} cannot be instantiated"))
        allowed = self.model.effective_properties(resolved)
        if self.model.unknown_property_policy == "reject":
            for name in sorted(set(properties) - set(allowed)):
                violations.append(MutationViolation("UNKNOWN_PROPERTY", name, f"Property {name!r} is not declared for {resolved}"))
        for name, prop in allowed.items():
            value = properties.get(name)
            if prop.cardinality.minimum and value is None:
                violations.append(MutationViolation("REQUIRED_PROPERTY", name, f"Property {name!r} is required for {resolved}"))
                continue
            if value is not None and not _datatype_ok(prop.datatype, value):
                violations.append(MutationViolation("INVALID_DATATYPE", name, f"Property {name!r} must be {prop.datatype}"))
        return violations

    def validate_relation_properties(
        self, relation_type: str, properties: dict[str, Any], *, tenant: str,
    ) -> list[MutationViolation]:
        """Check relationship attributes (presence, datatype, undeclared names)."""
        match = next((name for name, spec in self.model.relations.items()
                      if relation_type in {name, self.model.lpg_relation(name, spec)}), None)
        if not match:
            return [MutationViolation("UNKNOWN_RELATION", "relation", f"Unknown relation {relation_type!r}")]
        violations: list[MutationViolation] = []
        if not tenant.strip():
            violations.append(MutationViolation("TENANT_REQUIRED", "tenant", "A trusted tenant is required"))
        declared = self.model.relations[match].properties
        if self.model.unknown_property_policy == "reject":
            for name in sorted(set(properties) - set(declared)):
                violations.append(MutationViolation("UNKNOWN_RELATION_PROPERTY", name,
                    f"Attribute {name!r} is not declared for relation {match}"))
        for name, prop in declared.items():
            value = properties.get(name)
            if prop.cardinality.minimum and value is None:
                violations.append(MutationViolation("REQUIRED_RELATION_PROPERTY", name,
                    f"Attribute {name!r} is required for relation {match}"))
            elif value is not None and not _datatype_ok(prop.datatype, value):
                violations.append(MutationViolation("INVALID_DATATYPE", name,
                    f"Attribute {name!r} must be {prop.datatype}"))
        return violations

    def validate_relation(
        self, relation_type: str, source_type: str, target_type: str, *, tenant: str,
        existing_count: int | None = None, existing_by_target_type: dict[str, int] | None = None,
    ) -> list[MutationViolation]:
        violations: list[MutationViolation] = []
        match = next((name for name, spec in self.model.relations.items()
                      if relation_type in {name, self.model.lpg_relation(name, spec)}), None)
        if not match:
            return [MutationViolation("UNKNOWN_RELATION", "relation", f"Unknown relation {relation_type!r}")]
        if not tenant.strip():
            violations.append(MutationViolation("TENANT_REQUIRED", "tenant", "A trusted tenant is required"))
        relation = self.model.relations[match]
        if not self.model.is_subtype(source_type, relation.source):
            violations.append(MutationViolation("INVALID_SOURCE_TYPE", "source.type",
                f"{match} requires source type {relation.source}, got {source_type}"))
        if not self.model.is_subtype(target_type, relation.target):
            violations.append(MutationViolation("INVALID_TARGET_TYPE", "target.type",
                f"{match} requires target type {relation.target}, got {target_type}"))
        maximum = relation.cardinality.maximum
        if maximum is not None and existing_count is not None and existing_count >= maximum:
            violations.append(MutationViolation("MAX_CARDINALITY", "relation",
                f"{match} permits at most {maximum} relation(s) from one source"))
        # Qualified maximums count only existing relations whose target is of
        # the qualifying class; the new relation counts when its own target is.
        if existing_by_target_type is not None:
            for qualified in relation.qualified:
                limit = qualified.cardinality.maximum
                if limit is None or not self.model.is_subtype(target_type, qualified.on_class):
                    continue
                current = sum(count for kind, count in existing_by_target_type.items()
                              if self.model.is_subtype(kind, qualified.on_class))
                if current >= limit:
                    violations.append(MutationViolation("QUALIFIED_MAX_CARDINALITY", "relation",
                        f"{match} permits at most {limit} relation(s) to {qualified.on_class} from one source"))
        return violations

    def require_node(self, type_name: str, properties: dict[str, Any], *, tenant: str) -> None:
        if violations := self.validate_node(type_name, properties, tenant=tenant):
            raise MutationValidationError(violations)

    def require_relation(self, relation_type: str, source_type: str, target_type: str, *, tenant: str) -> None:
        if violations := self.validate_relation(relation_type, source_type, target_type, tenant=tenant):
            raise MutationValidationError(violations)
