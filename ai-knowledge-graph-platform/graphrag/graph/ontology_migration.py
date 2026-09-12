"""Dry-run and compatibility checks for ontology migrations."""

from __future__ import annotations

from dataclasses import dataclass, field


class OntologyMigrationBlockedError(RuntimeError):
    """Raised when a migration's competency-query gate fails and wasn't forced.

    See ``graphrag/graph/competency_gate.py`` and
    ``OntologyRegistry.apply_ontology_migration``'s ``competency_report``
    parameter -- this is distinct from a plain incompatible-diff rejection
    (``ValueError``, raised earlier in the same method): a competency
    failure means the diff itself is structurally fine, but the ontology
    can no longer answer a question it was previously able to answer.
    """


@dataclass(frozen=True)
class MigrationReport:
    compatible: bool
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    renamed: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def plan_migration(current: dict, target: dict) -> MigrationReport:
    """Compare class/property keys without mutating the live ontology."""
    current_classes = set((current.get("classes") or {}).keys())
    target_classes = set((target.get("classes") or {}).keys())
    current_props = set((current.get("properties") or {}).keys())
    target_props = set((target.get("properties") or {}).keys())
    migration_map = target.get("migration_map") or {}
    removed = sorted((current_classes - target_classes) | (current_props - target_props))
    renamed = [(old, new) for old, new in migration_map.items() if old in removed]
    unresolved = [item for item in removed if item not in migration_map]
    warnings = [f"unmapped removal: {item}" for item in unresolved]
    return MigrationReport(
        compatible=not unresolved,
        added=sorted((target_classes - current_classes) | (target_props - current_props)),
        removed=removed,
        renamed=renamed,
        warnings=warnings,
    )
