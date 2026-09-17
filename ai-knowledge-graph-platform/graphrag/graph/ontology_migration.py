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


def find_affected_competency_questions(report: MigrationReport, *, domain_id: str) -> list[str]:
    """Roadmap "P0 -- competency questions and model-impact testing", bullet 3
    ("model-change impact analysis showing affected... queries... and golden
    cases"): given a migration's added/removed/renamed names and the domain
    they apply to, report which of that domain's competency questions
    (``graphrag.graph.competency_gate.competency_questions_by_domain()``)
    reference one of the removed-or-renamed-from names in their own query
    text, so a reviewer can see *which specific competency question* a
    breaking ontology change would need re-running or rewriting -- not just
    that the class/property diff is structurally unmapped.

    ``domain_id`` is the domain's ``ontology.id`` from its
    ``config/ontologies/*.yml`` (or ``ontology/models/*.yaml`` for Energy),
    e.g. ``"aerospace-regulatory"``. An unregistered ``domain_id`` (no
    competency questions defined for it yet) returns an empty list rather
    than raising -- most domains in this repo have no competency questions
    at all today (see ``competency_questions_by_domain()``'s docstring).

    Honest about what this does not cover: R2RML/RML mappings, SHACL
    shapes, generated Neo4j constraints, and API contracts are not
    inspected -- only the one surface a competency question's query text
    actually touches. Renamed-*to* names are not checked either (a rename
    is only "affected" via its old name still appearing somewhere it
    shouldn't -- the new name appearing is expected, not a signal of
    impact). A question is flagged on a plain substring match against its
    query text, which can both under- and over-match (e.g. a name that is
    also a substring of an unrelated identifier); treat the result as a
    reviewer's starting point, not a certified impact list.
    """
    from graphrag.graph.competency_gate import competency_questions_by_domain

    factory = competency_questions_by_domain().get(domain_id)
    if factory is None:
        return []
    changed_names = set(report.removed) | {old for old, _new in report.renamed}
    if not changed_names:
        return []
    return [
        question.id
        for question in factory()
        if any(name in question.query for name in changed_names)
    ]
