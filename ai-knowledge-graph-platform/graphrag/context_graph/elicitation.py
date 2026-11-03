"""Expert elicitation: interview answers to reviewable drafts, with attribution.

Much domain knowledge lives in people, not documents.  This module turns an
interview (``config/knowledge_packs/templates/expert_interview.yml``) into a
draft knowledge item that records who said it and how, and records
disagreement between experts as a first-class object.

What it never does: produce an executable constraint, declare applicability
the expert was unsure about, or publish.  An "unsure" answer, or a scope
answered "not sure", yields ``NEEDS_EXPERT`` instead of ``DRAFT``; unstated
dimensions stay unknown (never "applies everywhere").  Publication still needs
a curator, a named reviewer and the release gate, and an open disagreement
blocks it (``check_pack_issues``).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from graphrag.context_graph.knowledge import (
    DIMENSIONS, Applicability, DimensionScope, EvidenceRef, ExpertAttribution, ExpertDisagreement, KnowledgeItem,
    KnowledgeOrigin, NormativeModality, PublicationStatus, SourceAuthority,
)

TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "config" / "knowledge_packs" / "templates" / "expert_interview.yml"


class ElicitationError(ValueError):
    """The interview answers cannot become a draft as given."""


def load_template(path: Path = TEMPLATE_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _scope(answer: Any) -> tuple[Applicability, bool]:
    """Applicability from a scope answer, and whether any part was 'not sure'."""
    unsure = False
    scopes: dict[str, DimensionScope | None] = {d: None for d in DIMENSIONS}
    for dimension, value in (answer or {}).items():
        if dimension not in DIMENSIONS:
            raise ElicitationError(f"unknown scope dimension {dimension!r}")
        if value in ("not sure", "not_sure", None, ""):
            unsure = True
        elif value == "any":
            scopes[dimension] = DimensionScope.unrestricted()
        else:
            values = [value] if isinstance(value, str) else list(value)
            scopes[dimension] = DimensionScope(mode="in", values=values)
    return Applicability(**scopes), unsure


def interview_to_draft(
    answers: Mapping[str, Any], *, item_id: str, tenant: str, expert: str, captured_at: datetime,
    role: str = "", review_owner: str = "", template: Mapping[str, Any] | None = None,
) -> KnowledgeItem:
    template = template or load_template()
    questions = {q["id"]: q for q in template["questions"]}
    missing = [q for q, spec in questions.items() if spec.get("required") and not answers.get(q)]
    if missing:
        raise ElicitationError(f"unanswered required questions: {missing}")
    for question, value in answers.items():
        spec = questions.get(question)
        if spec is None:
            raise ElicitationError(f"answer to unknown question {question!r}")
        if spec.get("choices") and value not in spec["choices"]:
            raise ElicitationError(f"{question}: {value!r} is not one of {spec['choices']}")
    applicability, scope_unsure = _scope(answers.get("scope"))
    needs_expert = answers["confidence"] == "unsure" or scope_unsure
    evidence = answers.get("evidence")
    sources = [EvidenceRef(source_id=f"interview:{expert}:{captured_at.date().isoformat()}",
                           source_type="expert_interview", quote=str(answers["statement"])[:2000],
                           recorded_at=captured_at)]
    if evidence:
        sources.append(EvidenceRef(source_id=str(evidence)[:256], source_type="cited_by_expert"))
    note = "; ".join(f"{k}: {answers[k]}" for k in ("exceptions", "others_disagree") if answers.get(k))
    return KnowledgeItem(
        id=item_id, version=1, tenant=tenant, title=str(answers["statement"])[:200],
        statement=str(answers["statement"]), subject_key=str(answers["subject"]),
        modality=NormativeModality(answers["modality"]), authority=SourceAuthority.INTERNAL,
        status=PublicationStatus.NEEDS_EXPERT if needs_expert else PublicationStatus.DRAFT,
        origin=KnowledgeOrigin.CURATED, applicability=applicability, sources=tuple(sources),
        recorded_at=captured_at, review_owner=review_owner,
        expert_attribution=(ExpertAttribution(expert=expert, role=role, method="interview",
                                              captured_at=captured_at, note=note[:1000]),),
    )


def record_disagreement(
    item: KnowledgeItem, positions: Mapping[str, str], *, disagreement_id: str, recorded_at: datetime,
) -> ExpertDisagreement:
    """Open a disagreement between at least two experts about one item."""
    return ExpertDisagreement(id=disagreement_id, tenant=item.tenant, item_id=item.id,
                              experts=tuple(sorted(positions)), positions=dict(positions),
                              recorded_at=recorded_at)


__all__ = ["ElicitationError", "TEMPLATE_PATH", "interview_to_draft", "load_template", "record_disagreement"]
