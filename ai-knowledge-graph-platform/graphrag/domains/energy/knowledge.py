"""Energy maintenance guidance as shared contextual knowledge.

The published Energy graph (SHACL-gated, versioned in the durable governance
store) already holds the maintenance guidance revisions: threshold, component
type, validity window, recorded time and supersession.  This module projects
those revisions into :class:`~graphrag.context_graph.knowledge.KnowledgeItem`
so the shared evaluator decides applicability, conflicts and per-asset
outcomes for the ``maintenance_review`` answer.

The projection is derived, not authored: its constraint comes from the
structured ``energy:temperatureReviewThreshold`` of a published revision, and
its approval is the publication gate that admitted that revision.  It is
recorded that way in ``constraint_provenance`` rather than attributed to a
human reviewer.

The Energy answer's own selection (``answers.select_bulletin``) remains the
authority for the answer text and its abstentions; the shared evaluation is
added alongside it and a regression test checks the two agree.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from graphrag.context_graph.knowledge import (
    Applicability, BusinessContext, Constraint, ConstraintProvenance, DimensionScope, EvidenceRef, Fact,
    KnowledgeItem, NormativeModality, Predicate, PublicationStatus, SourceAuthority,
)
from graphrag.context_graph.knowledge_evaluation import (
    EvaluationDecision, KnowledgeReleaseRef, evaluate_knowledge,
)
from graphrag.context_graph.models import ConditionOperator
from graphrag.enterprise.models import AccessContext, DocumentAccessPolicy

ENERGY_KNOWLEDGE_DOMAIN = "energy_maintenance"
WORKFLOW = "maintenance_review"
TEMPERATURE_FACT = "observation.temperature_c"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "unknown"


def _local(iri: str) -> str:
    return iri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def bulletin_items(
    rows: list[dict[str, Any]], *, tenant: str, publication_version: str, source_type,
) -> list[KnowledgeItem]:
    """Project ``bulletin_candidates.rq`` rows (already recorded-time filtered)."""
    by_iri: dict[str, dict[str, Any]] = {}
    supersedes: dict[str, set[str]] = {}
    for row in rows:
        by_iri.setdefault(row["bulletin"], row)
        if row.get("supersedes"):
            supersedes.setdefault(row["bulletin"], set()).add(row["supersedes"])
    id_of = {iri: row["bulletinId"] for iri, row in by_iri.items()}
    items = []
    for iri, row in sorted(by_iri.items(), key=lambda kv: kv[1]["bulletinId"]):
        component = str(row["componentType"])
        provenance = ConstraintProvenance(
            authored_by="system:energy-publication-projection",
            approval_ref=f"energy-publication:{publication_version or 'unversioned'}",
        )
        items.append(KnowledgeItem(
            id=str(row["bulletinId"]), version=1, tenant=tenant,
            title=f"{row['bulletinId']} {component} temperature review guidance",
            statement=(f"Review {component} assets whose temperature exceeds "
                       f"{row['threshold']} (from published guidance {row['bulletinId']})."),
            subject_key=f"maintenance.temperature_review.{_slug(component)}",
            series=f"maintenance.temperature_review.{_slug(component)}",
            # The answer path treats an exceedance as "advisory review is
            # required", never as a hard non-compliance: hence recommended.
            modality=NormativeModality.RECOMMENDED,
            authority=SourceAuthority.INTERNAL,
            status=PublicationStatus.PUBLISHED,
            applicability=Applicability(
                customer=DimensionScope.unrestricted(), brand=DimensionScope.unrestricted(),
                market=DimensionScope.unrestricted(),
                product=DimensionScope(mode="in", values=(component,)),
                product_family=DimensionScope.unrestricted(),
                workflow=DimensionScope(mode="in", values=(WORKFLOW,)),
                valid_from=datetime.fromisoformat(str(row["validFrom"]).replace("Z", "+00:00")),
                valid_to=(datetime.fromisoformat(str(row["validTo"]).replace("Z", "+00:00"))
                          if row.get("validTo") else None),
            ),
            constraint=Constraint(all_of=(Predicate(
                fact=TEMPERATURE_FACT, operator=ConditionOperator.LTE, value=float(row["threshold"]),
            ),)),
            constraint_provenance=provenance,
            sources=(EvidenceRef(
                source_id=str(row["bulletinId"]), source_type=source_type(iri),
                locator="temperatureReviewThreshold",
                recorded_at=datetime.fromisoformat(str(row["recordedAt"]).replace("Z", "+00:00")),
            ),),
            access=DocumentAccessPolicy(mode="tenant", state="known"),
            recorded_at=datetime.fromisoformat(str(row["recordedAt"]).replace("Z", "+00:00")),
            supersedes=tuple(sorted(id_of.get(t, _local(t)) for t in supersedes.get(iri, ()))),
            reviewed_by="system:energy-publication-gate",
            review_ref=f"energy-publication:{publication_version or 'unversioned'}",
        ))
    return items


def _compact(decision: EvaluationDecision) -> dict[str, Any]:
    return {
        "decision_id": decision.decision_id,
        "overall": decision.overall.value,
        "applicable": list(decision.applicable),
        "unknown_applicability": list(decision.unknown_applicability),
        "conflicts": [c.model_dump(mode="json") for c in decision.conflicts],
        "results": [
            {"item": f"{r.item_id}@v{r.item_version}", "modality": r.modality.value,
             "applicability": r.applicability.value, "outcome": r.outcome.value,
             "outcome_reason": r.outcome_reason,
             "predicates": [p.model_dump(mode="json") for p in r.predicates],
             "citations": [c.source_id for c in r.citations]}
            for r in decision.results
        ],
        "human_review_required": decision.human_review_required,
        "human_review_reasons": list(decision.human_review_reasons),
        "escalation_reason": decision.escalation.reason_code,
    }


def evaluate_maintenance(
    *, items: list[KnowledgeItem], findings: list[dict[str, Any]], component_types: list[str],
    tenant: str, as_of: datetime, known_as: datetime, release: KnowledgeReleaseRef,
) -> dict[str, Any]:
    """Shared evaluation for a maintenance review.

    ``guidance`` evaluates which guidance applies to the reviewed component
    types (no facts).  ``assets`` evaluates each flagged asset's observation
    against every applicable guidance revision, with the observation as cited
    evidence.  An absent component type is reported as unknown context.
    """
    access = AccessContext(subject_id="energy-answer", principals=[f"tenant:{tenant}"], groups_resolved=True)
    guidance_context = BusinessContext(
        product=component_types or None, workflow=[WORKFLOW], as_of=as_of, known_as=known_as,
    )
    guidance = evaluate_knowledge(
        items=items, context=guidance_context, tenant=tenant, access=access,
        domain=ENERGY_KNOWLEDGE_DOMAIN, release=release, expect_applicable_knowledge=True,
    )
    assets = []
    for finding in findings:
        context = BusinessContext(
            product=[finding["component_type"]], workflow=[WORKFLOW], as_of=as_of, known_as=known_as,
        )
        facts = {TEMPERATURE_FACT: Fact(value=float(finding["temperature"]), evidence=(EvidenceRef(
            source_id=finding["observation_source_id"], source_type=finding.get("observation_source_type", "observation"),
            locator=finding.get("locator", "temperature"),
        ),))}
        decision = evaluate_knowledge(
            items=items, context=context, facts=facts, tenant=tenant, access=access,
            domain=ENERGY_KNOWLEDGE_DOMAIN, release=release, expect_applicable_knowledge=True,
        )
        assets.append({"asset": finding["asset"], **_compact(decision)})
    # The guidance block answers "which guidance applies here?" only; it has
    # no observation facts, so no compliance outcome is reported for it.
    guidance_view = {
        "decision_id": guidance.decision_id,
        "applicable": list(guidance.applicable),
        "unknown_applicability": list(guidance.unknown_applicability),
        "not_applicable": [
            {"item": f"{r.item_id}@v{r.item_version}", "reasons": [x.code for x in r.applicability_reasons]}
            for r in guidance.results if r.applicability.value == "does_not_apply"
        ],
        "conflicts": [c.model_dump(mode="json") for c in guidance.conflicts],
        "human_review_required": guidance.human_review_required,
        "human_review_reasons": list(guidance.human_review_reasons),
    }
    return {
        "domain": ENERGY_KNOWLEDGE_DOMAIN,
        "release": release.model_dump(mode="json"),
        "guidance": guidance_view,
        "assets": assets,
        "note": ("Shared contextual-knowledge evaluation of the published guidance. The answer text and "
                 "its abstention above remain decided by the Energy temporal selection."),
    }


__all__ = ["ENERGY_KNOWLEDGE_DOMAIN", "bulletin_items", "evaluate_maintenance"]
