"""Deterministic, three-valued applicability: does a knowledge item bind in
this business context at this temporal view?

Semantics (documented in ``docs/contextual-knowledge-evaluation.md``):

* An unrestricted dimension (``DimensionScope(mode="any")``) never needs a
  context value.
* A restricted dimension needs the caller to state that dimension.  Missing
  request context gives ``unknown`` -- it never broadens applicability.
* Unknown applicability metadata on the item (no scope declared for a
  dimension) gives ``unknown`` -- it is never read as "applies everywhere".
* Several context values on one dimension (e.g. a carton sold in two markets)
  mean the decision covers all of them: the item applies if it binds any of
  them, and the matched values are reported so the partial scope is visible.
* A definite mismatch on any dimension gives ``does_not_apply`` even if another
  dimension is unknown -- one proven exclusion is enough.
* Time: an item not yet recorded at ``known_as`` is not visible; one outside
  its valid window at ``as_of`` does not apply; a window that cannot be checked
  because ``as_of`` is missing is ``unknown``.  Supersession is resolved by the
  evaluator across items (see ``knowledge_evaluation.py``).

Authorization is not decided here.  Items reach this module only after the
caller's access has been checked (``knowledge_evaluation.visible_items``).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from graphrag.context_graph.knowledge import DIMENSIONS, Applicability, BusinessContext


class ApplicabilityStatus(StrEnum):
    APPLIES = "applies"
    DOES_NOT_APPLY = "does_not_apply"
    UNKNOWN = "unknown"


class ApplicabilityReason(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    dimension: str = ""
    detail: str = ""


class ApplicabilityResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: ApplicabilityStatus
    reasons: tuple[ApplicabilityReason, ...] = ()
    matched: dict[str, tuple[str, ...]] = Field(default_factory=dict)


def _iso(value: datetime | None) -> str:
    return value.isoformat() if value else ""


def evaluate_applicability(
    applicability: Applicability,
    context: BusinessContext,
    *,
    recorded_at: datetime | None = None,
) -> ApplicabilityResult:
    """Evaluate one item's (or exception's) scope against a context."""
    exclusions: list[ApplicabilityReason] = []
    unknowns: list[ApplicabilityReason] = []
    satisfied: list[ApplicabilityReason] = []
    matched: dict[str, tuple[str, ...]] = {}

    known_as = context.known_as or context.as_of
    as_of = context.as_of

    # -- transaction time: what was knowable at the view -------------------
    if recorded_at is not None and known_as is not None and recorded_at > known_as:
        exclusions.append(ApplicabilityReason(
            code="not_recorded_at_view", dimension="time",
            detail=f"recorded {_iso(recorded_at)}, after known_as {_iso(known_as)}",
        ))

    # -- valid time ----------------------------------------------------------
    if applicability.valid_from or applicability.valid_to:
        if as_of is None:
            unknowns.append(ApplicabilityReason(
                code="as_of_missing", dimension="time",
                detail="the item has a validity window but the request gives no as_of",
            ))
        else:
            if applicability.valid_from and applicability.valid_from > as_of:
                exclusions.append(ApplicabilityReason(
                    code="not_yet_valid", dimension="time",
                    detail=f"valid from {_iso(applicability.valid_from)}, after as_of {_iso(as_of)}",
                ))
            if applicability.valid_to and applicability.valid_to <= as_of:
                exclusions.append(ApplicabilityReason(
                    code="expired", dimension="time",
                    detail=f"validity ended {_iso(applicability.valid_to)}, at or before as_of {_iso(as_of)}",
                ))

    # -- business dimensions -------------------------------------------------
    for dimension in DIMENSIONS:
        scope = applicability.scope(dimension)
        if scope is None:
            unknowns.append(ApplicabilityReason(
                code="applicability_metadata_unknown", dimension=dimension,
                detail="the item declares no scope for this dimension",
            ))
            continue
        if scope.mode == "any":
            continue
        requested = context.values(dimension)
        if requested is None:
            unknowns.append(ApplicabilityReason(
                code="context_missing", dimension=dimension,
                detail=f"the item is restricted on {dimension} but the request does not state it",
            ))
            continue
        allowed = set(scope.values)
        if scope.mode == "in":
            hit = tuple(sorted(set(requested) & allowed))
        else:
            hit = tuple(sorted(set(requested) - allowed))
        if not hit:
            exclusions.append(ApplicabilityReason(
                code="outside_scope" if scope.mode == "in" else "excluded_by_scope",
                dimension=dimension,
                detail=f"{dimension} {list(requested)} vs {scope.mode} {list(scope.values)}",
            ))
            continue
        matched[dimension] = hit
        if len(hit) < len(requested):
            satisfied.append(ApplicabilityReason(
                code="partial_scope_match", dimension=dimension,
                detail=f"binds {list(hit)} of the requested {list(requested)}",
            ))

    if exclusions:
        return ApplicabilityResult(
            status=ApplicabilityStatus.DOES_NOT_APPLY,
            reasons=tuple(exclusions + unknowns), matched=matched,
        )
    if unknowns:
        return ApplicabilityResult(
            status=ApplicabilityStatus.UNKNOWN, reasons=tuple(unknowns), matched=matched,
        )
    return ApplicabilityResult(
        status=ApplicabilityStatus.APPLIES, reasons=tuple(satisfied), matched=matched,
    )


__all__ = ["ApplicabilityReason", "ApplicabilityResult", "ApplicabilityStatus", "evaluate_applicability"]
