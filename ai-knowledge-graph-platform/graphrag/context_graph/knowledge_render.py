"""Controlled-natural-language rendering of executable constraints.

A review surface, not a rule language: reviewers read what the *executable*
constraint checks next to the human statement, so they approve the rule the
evaluator will run, not only the prose.  The rendering is deterministic, uses
the fixed operator vocabulary of ``ConditionOperator`` and the item's
modality, and states the missing-fact semantics explicitly.  It never feeds
back into evaluation.  The wording style is inspired by business-rule
vocabularies such as SBVR, but no SBVR model is implied.
"""

from __future__ import annotations

import json

from graphrag.context_graph.knowledge import Constraint, KnowledgeItem, NormativeModality, Predicate
from graphrag.context_graph.models import ConditionOperator

_PHRASES = {
    ConditionOperator.EQ: "equals {value}",
    ConditionOperator.NE: "does not equal {value}",
    ConditionOperator.GT: "is greater than {value}",
    ConditionOperator.GTE: "is at least {value}",
    ConditionOperator.LT: "is less than {value}",
    ConditionOperator.LTE: "is at most {value}",
    ConditionOperator.IN: "is one of {value}",
    ConditionOperator.CONTAINS: "contains {value}",
    ConditionOperator.IS_EMPTY: "is empty",
    ConditionOperator.NOT_EMPTY: "is not empty",
}

_LEAD = {
    NormativeModality.MANDATORY: "It is required that",
    NormativeModality.RECOMMENDED: "It is recommended that",
    NormativeModality.PROHIBITED: "It is prohibited that",
    NormativeModality.PERMITTED: "It is permitted that",
}


def _literal(value) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_literal(v) for v in value) + "]"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(value)


def render_predicate(predicate: Predicate) -> str:
    text = f"{predicate.fact} " + _PHRASES[predicate.operator].format(value=_literal(predicate.value))
    if predicate.when_missing == "violation":
        text += " (a missing value counts as a violation)"
    return text


def render_constraint(constraint: Constraint, modality: NormativeModality) -> str:
    """One sentence per constraint, e.g. ``It is required that
    label.min_font_pt is at least 8 and label.batch_code_present equals true.``"""
    clauses = " and ".join(render_predicate(p) for p in constraint.all_of)
    return f"{_LEAD[modality]} {clauses}."


def render_item(item: KnowledgeItem) -> str:
    """The executable form of an item, or why there is none."""
    if item.constraint is None:
        return "No executable constraint: this item is not machine-checked and goes to human review."
    return render_constraint(item.constraint, item.modality)


__all__ = ["render_constraint", "render_item", "render_predicate"]
