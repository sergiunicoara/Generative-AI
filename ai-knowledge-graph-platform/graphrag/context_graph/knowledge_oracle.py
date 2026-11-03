"""An independent, deliberately naive oracle for generated evaluations.

Written separately from ``applicability.py`` and ``knowledge_evaluation.py``
(it imports neither) so that agreement between the two is evidence rather
than a tautology.  It covers what can be stated simply: per-item
applicability for items without supersession or exceptions, and the
validation outcome of a single-predicate numeric constraint when the fact is
present with evidence and the item is not in a conflict.  Anything outside
that is reported as ``None`` (not judged) and excluded from the denominators.
"""

from __future__ import annotations

from datetime import datetime

from graphrag.context_graph.knowledge import DIMENSIONS, BusinessContext, Fact, KnowledgeItem


def oracle_applicability(item: KnowledgeItem, context: BusinessContext) -> str | None:
    if item.supersedes:
        return None
    known_as: datetime | None = context.known_as or context.as_of
    excluded = unknown = False
    if known_as is not None and item.recorded_at > known_as:
        excluded = True
    window = item.applicability
    if window.valid_from or window.valid_to:
        if context.as_of is None:
            unknown = True
        else:
            if window.valid_from and window.valid_from > context.as_of:
                excluded = True
            if window.valid_to and window.valid_to <= context.as_of:
                excluded = True
    for dim in DIMENSIONS:
        scope = getattr(window, dim)
        if scope is None:
            unknown = True
            continue
        if scope.mode == "any":
            continue
        requested = getattr(context, dim)
        if requested is None:
            unknown = True
            continue
        allowed = set(scope.values)
        remaining = set(requested) & allowed if scope.mode == "in" else set(requested) - allowed
        if not remaining:
            excluded = True
    if excluded:
        return "does_not_apply"
    return "unknown" if unknown else "applies"


def oracle_outcome(item: KnowledgeItem, facts: dict[str, Fact]) -> str | None:
    """Pass/fail for one numeric predicate with an evidenced fact, else None."""
    if item.constraint is None or len(item.constraint.all_of) != 1 or item.modality.value == "permitted":
        return None
    predicate = item.constraint.all_of[0]
    fact = facts.get(predicate.fact)
    if fact is None or fact.value is None or not fact.evidence:
        return None
    if isinstance(fact.value, bool) or not isinstance(fact.value, (int, float)):
        return None
    holds = {"gte": fact.value >= predicate.value, "lte": fact.value <= predicate.value,
             "gt": fact.value > predicate.value, "lt": fact.value < predicate.value}.get(predicate.operator.value)
    if holds is None:
        return None
    if item.modality.value == "prohibited":
        return "fail" if holds else "pass"
    return "pass" if holds else "fail"


__all__ = ["oracle_applicability", "oracle_outcome"]
