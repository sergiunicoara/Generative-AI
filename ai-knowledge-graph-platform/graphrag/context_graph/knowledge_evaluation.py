"""Contextual knowledge evaluation: applicability -> validation -> conflicts ->
decision.

Pipeline, in order, for one request:

1. **Authorization first.**  Items and exceptions are filtered to the trusted
   tenant, to published status, and to what the caller's ``AccessContext`` may
   read, after selecting the single latest version per id.  Nothing filtered
   out here appears anywhere in the result -- not as a candidate,
   explanation, conflict or citation.  Business context never grants access.
2. **Applicability** of every remaining item (three-valued, see
   ``applicability.py``), then transitive, order-independent supersession.
3. **Authorized exceptions** that apply in this context and cover the
   target's whole bound context (waivers, alone or jointly; precedence only
   with a winner that is itself in force).  Partial coverage leaves the
   target in force for the uncovered values.
4. **Validation** of every applicable requirement -- all of them, not the
   first matching one -- against bounded predicates over evidenced facts.
5. **Conflicts** among applicable requirements, resolved only by an applicable
   published exception.
6. **Decision**: the overall result goes through the existing Context Graph
   policy engine (an escalation ``PolicyVersion``), so escalation stays a
   declarative, versioned policy rather than ad-hoc code.

Principles enforced here:

* Mandatory and prohibited requirements stay in force alongside
  recommendations; recommendations never block and never cancel them.
* A permission does not override a prohibition: the pair is a conflict.
* Greater specificity, or a declared ``overrides`` list, authorises nothing.
* An exception needs explicit scope, provenance and validity (see contracts).
* Contradictory applicable requirements are a conflict unless an applicable
  published exception resolves them.
* A missing or unevidenced fact is unknown unless the predicate declares that
  absence is a violation.
* "Does not apply" and "applies but fails" are always reported differently.

No chain-of-thought is stored: the decision holds structured results only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from enum import StrEnum
from itertools import combinations
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from graphrag.context_graph.applicability import (
    ApplicabilityReason, ApplicabilityResult, ApplicabilityStatus, evaluate_applicability,
)
from graphrag.context_graph.knowledge import (
    BLOCKING_MODALITIES, DIMENSIONS, Applicability, BusinessContext, Constraint, EvidenceRef, Fact,
    KnowledgeException,
    KnowledgeItem, NormativeModality, Predicate, PublicationStatus, content_digest,
)
from graphrag.context_graph.models import (
    ConditionOperator, PolicyCondition, PolicyResult, PolicyRule, PolicyVersion, RuleMatch,
)
from graphrag.context_graph.policy_engine import _matches, evaluate_policy
from graphrag.core.tenancy import require_tenant
from graphrag.enterprise.models import ACLState, AccessContext, DocumentAccessPolicy

ESCALATION_POLICY_ID = "knowledge-evaluation-escalation"
ESCALATION_POLICY_VERSION = "1.1.0"


class ValidationOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICT = "conflict"
    NOT_EVALUATED = "not_evaluated"


class OverallOutcome(StrEnum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"
    NO_APPLICABLE_KNOWLEDGE = "no_applicable_knowledge"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class PredicateResult(_Frozen):
    fact: str
    operator: str
    expected: Any = None
    actual: Any = None
    result: str  # "true" | "false" | "unknown" | "missing_violation" | "invalid"
    reason: str = ""


class ConstraintResult(_Frozen):
    item_id: str
    item_version: int
    title: str
    subject_key: str
    modality: NormativeModality
    authority: str
    blocking: bool
    applicability: ApplicabilityStatus
    applicability_reasons: tuple[ApplicabilityReason, ...] = ()
    matched_context: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    outcome: ValidationOutcome
    outcome_reason: str
    validation_performed: bool
    predicates: tuple[PredicateResult, ...] = ()
    missing_facts: tuple[str, ...] = ()
    citations: tuple[EvidenceRef, ...] = ()
    exception_refs: tuple[str, ...] = ()
    # exceptions that apply here but cover only part of the item's bound
    # context: the item stays in force and is evaluated for every value
    partial_exception_refs: tuple[str, ...] = ()
    # where a partially superseded or partially waived item still binds, as
    # disjoint context boxes (a dimension left out is unbounded); empty when
    # the item binds its whole matched context
    in_force_context: tuple[dict[str, tuple[str, ...]], ...] = ()
    conflict_ids: tuple[str, ...] = ()


class KnowledgeConflict(_Frozen):
    conflict_id: str
    kind: str
    subject_key: str
    item_keys: tuple[str, ...]
    detail: str
    resolved: bool = False
    resolution_ref: str = ""


class AppliedException(_Frozen):
    exception_key: str
    kind: str
    target_item_ids: tuple[str, ...]
    winner_item_id: str = ""
    granted_by: str
    approval_ref: str
    citations: tuple[EvidenceRef, ...] = ()


class EscalationResult(_Frozen):
    policy_id: str
    policy_version: str
    result: PolicyResult
    matched_rule: str
    reason_code: str


class KnowledgeReleaseRef(_Frozen):
    release_id: str
    domain: str
    version: int | str
    content_hash: str = ""


class EvaluationDecision(_Frozen):
    decision_id: str
    tenant: str
    domain: str
    release: KnowledgeReleaseRef
    context: BusinessContext
    facts_used: dict[str, Any] = Field(default_factory=dict)
    knowledge: tuple[str, ...] = ()           # every visible item key considered
    applicable: tuple[str, ...] = ()
    unknown_applicability: tuple[str, ...] = ()
    results: tuple[ConstraintResult, ...] = ()
    conflicts: tuple[KnowledgeConflict, ...] = ()
    exceptions_applied: tuple[AppliedException, ...] = ()
    missing_evidence: tuple[dict[str, str], ...] = ()
    citations: tuple[EvidenceRef, ...] = ()
    overall: OverallOutcome
    human_review_required: bool
    human_review_reasons: tuple[str, ...] = ()
    escalation: EscalationResult
    schema_version: str = "knowledge-decision/v1"


# ---------------------------------------------------------------------------
# authorization


def can_read(policy: DocumentAccessPolicy, access: AccessContext) -> bool:
    """Python twin of ``enterprise.access.document_access_predicate``, always
    enforced (fail closed) for contextual knowledge."""
    principals = set(access.principals)
    if principals & set(policy.deny_principals):
        return False
    if policy.mode == "tenant":
        return policy.state != ACLState.UNKNOWN
    if policy.state != ACLState.KNOWN:
        return False
    if not principals & set(policy.allow_principals):
        return False
    return (not policy.requires_group_resolution) or access.groups_resolved


def _effective_versions(entries: Iterable, tenant: str) -> list:
    """Single effective version per id: the highest version wins.

    Selected *before* status and access checks, so a newer withdrawn,
    unpublished or inaccessible version is never silently replaced by an
    obsolete published one.  Deterministic regardless of input order.
    """
    latest: dict[str, Any] = {}
    for entry in entries:
        if entry.tenant != tenant:
            continue
        current = latest.get(entry.id)
        if current is None or entry.version > current.version:
            latest[entry.id] = entry
    return [latest[key] for key in sorted(latest)]


def visible_items(
    items: Iterable[KnowledgeItem], *, tenant: str, access: AccessContext,
) -> list[KnowledgeItem]:
    tenant = require_tenant(tenant)
    return [
        item for item in _effective_versions(items, tenant)
        if item.status == PublicationStatus.PUBLISHED and can_read(item.access, access)
    ]


def visible_exceptions(
    exceptions: Iterable[KnowledgeException], *, tenant: str, access: AccessContext,
) -> list[KnowledgeException]:
    tenant = require_tenant(tenant)
    return [
        exc for exc in _effective_versions(exceptions, tenant)
        if exc.status == PublicationStatus.PUBLISHED and can_read(exc.access, access)
    ]


# ---------------------------------------------------------------------------
# validation


def _bounded_repr(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 200:
        return value[:200] + "..."
    if isinstance(value, list) and len(value) > 20:
        return value[:20] + ["..."]
    return value


_NUMERIC_OPS = (ConditionOperator.GT, ConditionOperator.GTE, ConditionOperator.LT, ConditionOperator.LTE)


def _category(value: Any) -> str:
    """Type category used for strict comparisons (bool is not a number)."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    return "other"


def _same(a: Any, b: Any) -> bool:
    return _category(a) == _category(b) and a == b


def _typed_match(actual: Any, operator: ConditionOperator, expected: Any) -> bool | None:
    """Evaluate one operator with strict, explicit type rules.

    Returns ``None`` when the fact's type is not comparable with the operator
    and value -- never coerces ("7" is not 7, True is not 1) and never raises.
    Emptiness checks accept any type.
    """
    if operator in (ConditionOperator.IS_EMPTY, ConditionOperator.NOT_EMPTY):
        return _matches(actual, operator, expected)
    kind = _category(actual)
    if operator in _NUMERIC_OPS:
        if kind != "number":
            return None
        return _matches(actual, operator, expected)
    if operator in (ConditionOperator.EQ, ConditionOperator.NE):
        if kind != _category(expected) or kind in ("list", "other"):
            return None
        equal = actual == expected
        return equal if operator == ConditionOperator.EQ else not equal
    if operator == ConditionOperator.IN:
        if kind in ("list", "other") or kind not in {_category(v) for v in expected}:
            return None
        return any(_same(actual, v) for v in expected)
    if operator == ConditionOperator.CONTAINS:
        if kind == "list":
            return any(_same(element, expected) for element in actual)
        if kind == "string" and _category(expected) == "string":
            return expected in actual
        return None
    return None


def evaluate_predicate(predicate: Predicate, facts: Mapping[str, Fact]) -> tuple[PredicateResult, list[EvidenceRef]]:
    """Result is one of ``true``, ``false``, ``unknown`` (missing or
    unevidenced), ``missing_violation`` (absent, and the predicate declares
    absence a violation) or ``invalid`` (fact type not comparable)."""
    fact = facts.get(predicate.fact)
    missing = fact is None or fact.value is None
    common = dict(fact=predicate.fact, operator=predicate.operator.value, expected=predicate.value)
    if not missing and not fact.evidence:
        return PredicateResult(**common, actual=_bounded_repr(fact.value), result="unknown",
                               reason="fact_without_evidence"), []
    if missing:
        if predicate.when_missing == "violation":
            return PredicateResult(**common, actual=None, result="missing_violation",
                                   reason="absence_is_violation"), []
        return PredicateResult(**common, actual=None, result="unknown", reason="fact_missing"), []
    holds = _typed_match(fact.value, predicate.operator, predicate.value)
    if holds is None:
        return PredicateResult(**common, actual=_bounded_repr(fact.value), result="invalid",
                               reason="fact_type_mismatch"), []
    return PredicateResult(**common, actual=_bounded_repr(fact.value),
                           result="true" if holds else "false"), list(fact.evidence)


def evaluate_constraint(
    constraint: Constraint, modality: NormativeModality, facts: Mapping[str, Fact],
) -> tuple[ValidationOutcome, str, list[PredicateResult], list[str], list[EvidenceRef]]:
    """Return outcome, reason, predicate results, missing facts, fact evidence.

    For mandatory/recommended the constraint describes the required state; for
    prohibited it describes the forbidden state (holds -> fail).  A declared
    missing-fact violation fails under every modality: it is never read as
    "the forbidden state is absent".  An invalid (type-mismatched) fact is
    insufficient evidence, never a pass or fail.
    """
    results: list[PredicateResult] = []
    evidence: list[EvidenceRef] = []
    for predicate in constraint.all_of:
        result, cited = evaluate_predicate(predicate, facts)
        results.append(result)
        evidence.extend(cited)
    missing = [r.fact for r in results if r.result in ("unknown", "invalid")]
    values = {r.result for r in results}
    insufficient_reason = "fact_type_mismatch" if "invalid" in values else "facts_missing_or_unevidenced"
    if "missing_violation" in values:
        return ValidationOutcome.FAIL, "absence_is_violation", results, missing, evidence
    if modality == NormativeModality.PROHIBITED:
        if "false" in values:
            return ValidationOutcome.PASS, "prohibited_state_absent", results, missing, evidence
        if values & {"unknown", "invalid"}:
            return ValidationOutcome.INSUFFICIENT_EVIDENCE, insufficient_reason, results, missing, evidence
        return ValidationOutcome.FAIL, "prohibited_state_present", results, missing, evidence
    if "false" in values:
        return ValidationOutcome.FAIL, "requirement_not_met", results, missing, evidence
    if values & {"unknown", "invalid"}:
        return ValidationOutcome.INSUFFICIENT_EVIDENCE, insufficient_reason, results, missing, evidence
    return ValidationOutcome.PASS, "requirement_met", results, missing, evidence


# ---------------------------------------------------------------------------
# conflicts


_LOWER = (ConditionOperator.GT, ConditionOperator.GTE)
_UPPER = (ConditionOperator.LT, ConditionOperator.LTE)


def _incompatible(a: KnowledgeItem, b: KnowledgeItem) -> str:
    """Return why two requirements cannot both be satisfied, or "".

    Only provable unsatisfiability counts: two different required values
    (``eq``), disjoint allowed sets (``in``), or numeric bounds with an empty
    intersection.  Two bounds in the same direction (font >= 6 and >= 8) are
    compatible -- the stricter simply decides -- and are not a conflict.
    """
    if a.constraint is None or b.constraint is None:
        return ""
    shared = {p.fact for p in a.constraint.all_of} & {p.fact for p in b.constraint.all_of}
    by_fact: dict[str, list[Predicate]] = {}
    for p in (*a.constraint.all_of, *b.constraint.all_of):
        if p.fact in shared:
            by_fact.setdefault(p.fact, []).append(p)
    for fact, group in sorted(by_fact.items()):
        eqs = {repr(p.value) for p in group if p.operator == ConditionOperator.EQ}
        if len(eqs) > 1:
            return f"{fact} must equal {' and '.join(sorted(eqs))}"
        sets = [set(p.value) for p in group if p.operator == ConditionOperator.IN]
        if len(sets) > 1 and not set.intersection(*sets):
            return f"{fact} allowed sets are disjoint"
        lowers = [(p.value, p.operator == ConditionOperator.GT) for p in group if p.operator in _LOWER]
        uppers = [(p.value, p.operator == ConditionOperator.LT) for p in group if p.operator in _UPPER]
        if lowers and uppers:
            low, low_strict = max(lowers, key=lambda x: (x[0], x[1]))
            high, high_strict = min(uppers, key=lambda x: (x[0], not x[1]))
            if low > high or (low == high and (low_strict or high_strict)):
                return f"{fact} must be {'>' if low_strict else '>='} {low} and {'<' if high_strict else '<='} {high}"
    return ""


def detect_conflicts(
    items: list[KnowledgeItem], ordered: frozenset[frozenset[str]] = frozenset(),
) -> list[KnowledgeConflict]:
    """Pairwise conflicts; ``ordered`` pairs (item ids related by supersession)
    are revisions of each other, not contenders, and are skipped."""
    conflicts: list[KnowledgeConflict] = []
    for a, b in combinations(sorted(items, key=lambda i: i.id), 2):
        if frozenset({a.id, b.id}) in ordered:
            continue
        kinds: list[tuple[str, str]] = []
        if a.id in b.overrides or b.id in a.overrides:
            claimant = b if a.id in b.overrides else a
            kinds.append(("unauthorized_override",
                          f"{claimant.id} declares it overrides "
                          f"{(a if claimant is b else b).id} without an authorised exception"))
        if a.series and a.series == b.series:
            kinds.append(("ambiguous_revision",
                          f"two revisions of series {a.series!r} are in force and no supersession orders them"))
        elif a.subject_key == b.subject_key:
            pair = {a.modality, b.modality}
            if pair == {NormativeModality.MANDATORY, NormativeModality.PROHIBITED}:
                kinds.append(("mandatory_vs_prohibited", "one requires what the other prohibits"))
            elif pair == {NormativeModality.PERMITTED, NormativeModality.PROHIBITED}:
                kinds.append(("permission_vs_prohibition",
                              "a permission does not override a prohibition"))
            elif a.modality == b.modality and a.modality in (
                NormativeModality.MANDATORY, NormativeModality.RECOMMENDED,
            ):
                detail = _incompatible(a, b)
                if detail:
                    kinds.append(("incompatible_requirements", detail))
        for kind, detail in kinds:
            keys = (a.key, b.key)
            conflicts.append(KnowledgeConflict(
                conflict_id="CF-" + content_digest([kind, keys])[:12], kind=kind,
                subject_key=a.subject_key if a.subject_key == b.subject_key else f"{a.subject_key}|{b.subject_key}",
                item_keys=keys, detail=detail,
            ))
    return conflicts


# ---------------------------------------------------------------------------
# escalation policy (declarative, evaluated by the existing policy engine)


def escalation_policy(tenant: str) -> PolicyVersion:
    def rule(rule_id: str, field: str, result: PolicyResult, reason: str, priority: int) -> PolicyRule:
        return PolicyRule(
            id=rule_id, priority=priority, match=RuleMatch.ALL, result=result, reason_code=reason,
            conditions=[PolicyCondition(field=field, operator=ConditionOperator.GT, value=0)],
        )

    return PolicyVersion(
        id=f"{ESCALATION_POLICY_ID}@{ESCALATION_POLICY_VERSION}", tenant=tenant,
        policy_id=ESCALATION_POLICY_ID, version=ESCALATION_POLICY_VERSION,
        title="Escalate contextual-knowledge decisions that cannot be made deterministically",
        default_result=PolicyResult.ALLOW, enforcement_mode="enforced",
        rules=[
            rule("unresolved-conflict", "summary.unresolved_conflicts", PolicyResult.ESCALATE,
                 "unresolved_conflict", 10),
            rule("unresolved-exception", "summary.unresolved_exceptions", PolicyResult.ESCALATE,
                 "unresolved_exception", 15),
            rule("blocking-failed", "summary.blocking_failed", PolicyResult.DENY,
                 "blocking_requirement_failed", 20),
            rule("blocking-applicability-unknown", "summary.blocking_applicability_unknown",
                 PolicyResult.ESCALATE, "applicability_unknown", 30),
            rule("blocking-insufficient", "summary.blocking_insufficient", PolicyResult.ESCALATE,
                 "insufficient_evidence", 40),
            rule("blocking-unchecked", "summary.blocking_unchecked", PolicyResult.ESCALATE,
                 "requirement_not_machine_checkable", 50),
            rule("expected-knowledge-missing", "summary.expected_knowledge_missing",
                 PolicyResult.ESCALATE, "no_applicable_knowledge", 60),
        ],
    )


# ---------------------------------------------------------------------------
# coverage: does one applying scope include everything another binds?


def _bound(app: Applicability, result: ApplicabilityResult, context: BusinessContext,
           dimension: str) -> set[str] | None:
    """Context values an applying item binds on one dimension.

    ``None`` means unbounded: the scope is unrestricted and the request does
    not state the dimension.
    """
    scope = app.scope(dimension)
    if scope is not None and scope.mode == "any":
        values = context.values(dimension)
        return None if values is None else set(values)
    return set(result.matched.get(dimension, ()))


def _uncovered(coverer: Applicability, coverer_result: ApplicabilityResult,
               target: Applicability, target_result: ApplicabilityResult,
               context: BusinessContext) -> list[str]:
    """Dimensions on which ``target`` binds context values ``coverer`` does not.

    Both must apply.  An empty list means the coverer applies everywhere the
    target does, so it may waive, override or supersede the target as a whole;
    otherwise the target stays in force for the uncovered values.
    """
    gaps: list[str] = []
    for dimension in DIMENSIONS:
        covering = _bound(coverer, coverer_result, context, dimension)
        if covering is None:
            continue
        bound = _bound(target, target_result, context, dimension)
        if bound is None or not bound <= covering:
            gaps.append(dimension)
    return gaps


# A bound context as a box: per dimension, the bound values, or ``None`` when
# unbounded.  Several partial coverers jointly cover a target only if the union
# of their boxes includes every combination of the target's values -- a waiver
# for (market A, product P1) and one for (market B, product P2) leave
# (A, P2) and (B, P1) uncovered, so per-dimension unions are not enough.
Box = dict[str, frozenset[str] | None]

#: Fragment cap for the uncovered remainder.  Past it, nothing counts as
#: covered (fail closed): the target stays in force for its whole context.
_REMAINDER_LIMIT = 256


def _box(app: Applicability, result: ApplicabilityResult, context: BusinessContext) -> Box:
    box: Box = {}
    for dimension in DIMENSIONS:
        bound = _bound(app, result, context, dimension)
        box[dimension] = None if bound is None else frozenset(bound)
    return box


def _subtract(box: Box, coverer: Box) -> list[Box] | None:
    """``box`` minus ``coverer`` as disjoint boxes, or ``None`` when the
    coverer removes nothing.

    A coverer bounded on a dimension the box leaves unbounded removes nothing:
    finitely many values never cover an unstated, open-ended dimension.
    """
    for dimension in DIMENSIONS:
        covering = coverer[dimension]
        if covering is None:
            continue
        bound = box[dimension]
        if bound is None or not bound & covering:
            return None
    pieces: list[Box] = []
    current = dict(box)
    for dimension in DIMENSIONS:
        covering = coverer[dimension]
        if covering is None:
            continue
        outside = current[dimension] - covering
        if outside:
            pieces.append({**current, dimension: outside})
        current[dimension] = current[dimension] & covering
    return pieces


def _remainder(start: list[Box], coverers: list[tuple[str, Box]]) -> tuple[list[Box], list[str]]:
    """What no coverer covers, and the keys of the coverers that removed part
    of it.  Exact union semantics, independent of coverer order for the
    covered/uncovered answer."""
    remaining = list(start)
    used: list[str] = []
    for key, coverer in coverers:
        after: list[Box] = []
        touched = False
        for box in remaining:
            pieces = _subtract(box, coverer)
            if pieces is None:
                after.append(box)
            else:
                after.extend(pieces)
                touched = True
        if len(after) > _REMAINDER_LIMIT:
            return list(start), []
        if touched:
            used.append(key)
        remaining = after
    return remaining, used


def _context_boxes(boxes: list[Box]) -> tuple[dict[str, tuple[str, ...]], ...]:
    rendered = [
        {dimension: tuple(sorted(values)) for dimension, values in box.items() if values is not None}
        for box in boxes
    ]
    return tuple(sorted(rendered, key=lambda box: sorted(box.items())))


def _supersession(
    visible: list[KnowledgeItem], base: dict[str, ApplicabilityResult], context: BusinessContext,
) -> tuple[dict[str, ApplicabilityResult], set[str], frozenset[frozenset[str]], dict[str, list[Box]]]:
    """Resolve supersession from the immutable per-item applicability.

    ``S.supersedes = [T]`` is transitive (C supersedes B supersedes A means C
    supersedes A, even when B itself is not in force).  T stops applying when
    the later revisions that apply together cover T's whole bound context; it
    becomes unknown when a later revision's applicability is unknown or the
    lineage is cyclic; later revisions covering only part of T leave T in
    force for the rest.  Independent of input order.  Returns final
    applicability, the ids that were superseded, the revision pairs ordered by
    supersession, and the remaining in-force context of partially superseded
    items.
    """
    by_id = {item.id: item for item in visible}
    direct: dict[str, set[str]] = {}
    for item in visible:
        for target in item.supersedes:
            if target in by_id:
                direct.setdefault(target, set()).add(item.id)

    def later(item_id: str) -> set[str]:
        seen: set[str] = set()
        stack = list(direct.get(item_id, ()))
        while stack:
            current = stack.pop()
            if current not in seen:
                seen.add(current)
                stack.extend(direct.get(current, ()))
        return seen

    final = dict(base)
    superseded: set[str] = set()
    ordered: set[frozenset[str]] = set()
    remaining_context: dict[str, list[Box]] = {}
    for item in visible:
        revisions = later(item.id)
        ordered.update(frozenset({item.id, other}) for other in revisions if other != item.id)
        if not revisions:
            continue
        current = base[item.id]
        if item.id in revisions:
            if current.status != ApplicabilityStatus.DOES_NOT_APPLY:
                final[item.id] = ApplicabilityResult(
                    status=ApplicabilityStatus.UNKNOWN, matched=current.matched,
                    reasons=(ApplicabilityReason(code="supersession_cycle", dimension="time",
                                                 detail="the supersession lineage is cyclic"),),
                )
            continue
        if current.status != ApplicabilityStatus.APPLIES:
            continue
        unknown = ""
        coverers: list[tuple[str, Box]] = []
        for other_id in sorted(revisions):
            other, other_base = by_id[other_id], base[other_id]
            if other_base.status == ApplicabilityStatus.APPLIES:
                coverers.append((other.key, _box(other.applicability, other_base, context)))
            elif other_base.status == ApplicabilityStatus.UNKNOWN:
                unknown = unknown or other.key
        remaining, used = _remainder([_box(item.applicability, current, context)], coverers)
        if not remaining:
            superseded.add(item.id)
            final[item.id] = ApplicabilityResult(
                status=ApplicabilityStatus.DOES_NOT_APPLY, matched=current.matched,
                reasons=(ApplicabilityReason(code="superseded", dimension="time",
                                             detail=f"superseded by {', '.join(used)}"),),
            )
        elif unknown:
            final[item.id] = ApplicabilityResult(
                status=ApplicabilityStatus.UNKNOWN, matched=current.matched,
                reasons=(ApplicabilityReason(code="supersession_undetermined", dimension="time",
                                             detail=f"may be superseded by {unknown}"),),
            )
        elif used:
            remaining_context[item.id] = remaining
            final[item.id] = current.model_copy(update={"reasons": (*current.reasons, ApplicabilityReason(
                code="partially_superseded", dimension="time",
                detail=(f"{', '.join(used)} cover{'s' if len(used) == 1 else ''} only part of this "
                        f"context; this revision stays in force for {list(_context_boxes(remaining))}")),)})
    return final, superseded, frozenset(ordered), remaining_context


# ---------------------------------------------------------------------------
# the evaluator


def evaluate_knowledge(
    *,
    items: Iterable[KnowledgeItem],
    exceptions: Iterable[KnowledgeException] = (),
    context: BusinessContext,
    facts: Mapping[str, Fact] | None = None,
    tenant: str,
    access: AccessContext,
    domain: str,
    release: KnowledgeReleaseRef,
    expect_applicable_knowledge: bool = False,
    semantic_candidates: Iterable[str] | None = None,
) -> EvaluationDecision:
    """Evaluate every visible published requirement in a release.

    ``semantic_candidates`` (item ids a retriever ranked as relevant) only
    orders the returned results.  It never limits which requirements are
    evaluated: a mandatory rule outside a semantic top-k is still applied.
    """
    tenant = require_tenant(tenant)
    facts = dict(facts or {})
    visible = visible_items(items, tenant=tenant, access=access)
    visible_exc = visible_exceptions(exceptions, tenant=tenant, access=access)

    # -- applicability ------------------------------------------------------
    own_scope = {
        item.id: evaluate_applicability(item.applicability, context, recorded_at=item.recorded_at)
        for item in visible
    }
    applicability, superseded, ordered, in_force = _supersession(visible, own_scope, context)

    by_id = {item.id: item for item in visible}
    applying = [item for item in visible if applicability[item.id].status == ApplicabilityStatus.APPLIES]

    # -- authorized exceptions ---------------------------------------------
    # An exception changes a requirement only when it applies, covers the
    # requirement's whole bound context and -- for precedence -- names a winner
    # that is itself in force here.  Waivers may cover jointly: together they
    # must cover every combination of context values the requirement still
    # binds after supersession.  Anything less leaves the requirement in
    # force; a precedence that cannot be confirmed escalates.
    applied: list[AppliedException] = []
    waived: dict[str, tuple[str, ...]] = {}
    partly_waived: set[str] = set()
    losers: dict[str, str] = {}
    precedence: dict[frozenset[str], str] = {}
    partial_refs: dict[str, list[str]] = {}
    review_reasons: list[str] = []
    unresolved_exceptions = 0
    in_play = (ApplicabilityStatus.APPLIES, ApplicabilityStatus.UNKNOWN)

    exc_results = {
        exc.id: evaluate_applicability(exc.applicability, context, recorded_at=exc.recorded_at)
        for exc in visible_exc
    }
    candidates: list[KnowledgeException] = []
    for exc in sorted(visible_exc, key=lambda e: e.id):
        targets = [t for t in exc.target_item_ids
                   if t in by_id and applicability[t].status in in_play]
        if not targets:
            continue
        status = exc_results[exc.id].status
        if status == ApplicabilityStatus.UNKNOWN:
            unresolved_exceptions += 1
            review_reasons.append(f"exception {exc.key} may apply but its scope cannot be determined")
        elif status == ApplicabilityStatus.APPLIES:
            candidates.append(exc)

    def coverage_gaps(exc: KnowledgeException, target: str) -> list[str]:
        return _uncovered(exc.applicability, exc_results[exc.id],
                          by_id[target].applicability, applicability[target], context)

    def applying_targets(exc: KnowledgeException) -> list[str]:
        return [t for t in exc.target_item_ids
                if t in by_id and applicability[t].status == ApplicabilityStatus.APPLIES]

    waivers = [e for e in candidates if e.kind == "waiver"]
    waiver_boxes: dict[str, list[tuple[str, Box]]] = {}
    for exc in waivers:
        exc_box = _box(exc.applicability, exc_results[exc.id], context)
        for target in applying_targets(exc):
            waiver_boxes.setdefault(target, []).append((exc.key, exc_box))
    waived_targets: dict[str, list[str]] = {}
    for target, boxes in sorted(waiver_boxes.items()):
        start = in_force.get(target) or [_box(by_id[target].applicability, applicability[target], context)]
        remaining, used = _remainder(start, boxes)
        if not remaining:
            waived[target] = tuple(used)
            for key in used:
                waived_targets.setdefault(key, []).append(target)
        elif used:
            partly_waived.add(target)
            partial_refs.setdefault(target, []).extend(used)
            in_force[target] = remaining
    for exc in waivers:
        if exc.key in waived_targets:
            applied.append(AppliedException(
                exception_key=exc.key, kind=exc.kind, target_item_ids=tuple(waived_targets[exc.key]),
                granted_by=exc.granted_by, approval_ref=exc.approval_ref, citations=exc.sources,
            ))

    precedences = [e for e in candidates if e.kind == "precedence"]
    precedence_losers = {t for e in precedences for t in e.target_item_ids}
    for exc in precedences:
        winner_id = exc.winner_item_id
        resolved: list[str] = []
        unconfirmed = ""
        for target in applying_targets(exc):
            if target in waived:
                continue
            if coverage_gaps(exc, target):
                partial_refs.setdefault(target, []).append(exc.key)
                continue
            winner = by_id.get(winner_id)
            if winner is None:
                unconfirmed = "names a winning requirement that is not available to this caller"
                break
            winner_status = applicability[winner_id].status
            if winner_id in superseded:
                unconfirmed = f"names {winner.key}, which is superseded"
                break
            if winner_status == ApplicabilityStatus.UNKNOWN:
                unconfirmed = f"names {winner.key}, whose applicability is unknown"
                break
            if winner_status == ApplicabilityStatus.DOES_NOT_APPLY or winner_id in waived:
                break  # moot: no effective winner, the target simply stays in force
            if winner_id in partly_waived:
                unconfirmed = f"names {winner.key}, which is only partly waived"
                break
            if winner_id in precedence_losers:
                unconfirmed = "is contradicted or chained by another precedence exception"
                break
            if _uncovered(winner.applicability, applicability[winner_id],
                          by_id[target].applicability, applicability[target], context):
                partial_refs.setdefault(target, []).append(exc.key)
                continue
            resolved.append(target)
        if unconfirmed:
            unresolved_exceptions += 1
            review_reasons.append(f"precedence exception {exc.key} {unconfirmed}")
            continue
        for target in resolved:
            losers[target] = exc.key
            precedence[frozenset({target, winner_id})] = exc.key
        if resolved:
            applied.append(AppliedException(
                exception_key=exc.key, kind=exc.kind, target_item_ids=tuple(resolved),
                winner_item_id=winner_id, granted_by=exc.granted_by,
                approval_ref=exc.approval_ref, citations=exc.sources,
            ))

    # -- conflicts ------------------------------------------------------------
    contenders = [item for item in applying if item.id not in waived]
    conflicts: list[KnowledgeConflict] = []
    for conflict in detect_conflicts(contenders, ordered):
        ids = frozenset(key.split("@v")[0] for key in conflict.item_keys)
        ref = precedence.get(ids, "")
        conflicts.append(conflict.model_copy(update={"resolved": bool(ref), "resolution_ref": ref}))
    unresolved_by_item: dict[str, list[str]] = {}
    for conflict in conflicts:
        if not conflict.resolved:
            for key in conflict.item_keys:
                unresolved_by_item.setdefault(key.split("@v")[0], []).append(conflict.conflict_id)

    # -- validation -----------------------------------------------------------
    results: list[ConstraintResult] = []
    citations: dict[tuple[str, str], EvidenceRef] = {}
    missing_evidence: list[dict[str, str]] = []
    facts_used: dict[str, Any] = {}
    order = {item_id: rank for rank, item_id in enumerate(semantic_candidates or ())}

    for item in sorted(visible, key=lambda i: (order.get(i.id, len(order)), i.id)):
        app = applicability[item.id]
        blocking = item.modality in BLOCKING_MODALITIES
        base = dict(
            item_id=item.id, item_version=item.version, title=item.title, subject_key=item.subject_key,
            modality=item.modality, authority=item.authority.value, blocking=blocking,
            applicability=app.status, applicability_reasons=app.reasons, matched_context=app.matched,
            partial_exception_refs=tuple(partial_refs.get(item.id, ())),
            in_force_context=_context_boxes(in_force[item.id]) if item.id in in_force else (),
        )
        if app.status == ApplicabilityStatus.DOES_NOT_APPLY:
            results.append(ConstraintResult(**base, outcome=ValidationOutcome.NOT_EVALUATED,
                                            outcome_reason="not_applicable", validation_performed=False))
            continue
        if app.status == ApplicabilityStatus.UNKNOWN:
            results.append(ConstraintResult(**base, outcome=ValidationOutcome.NOT_EVALUATED,
                                            outcome_reason="applicability_unknown",
                                            validation_performed=False, citations=item.sources))
            continue
        if item.id in waived:
            results.append(ConstraintResult(**base, outcome=ValidationOutcome.NOT_EVALUATED,
                                            outcome_reason="waived_by_exception", validation_performed=False,
                                            citations=item.sources, exception_refs=waived[item.id]))
            continue
        if item.id in losers:
            results.append(ConstraintResult(**base, outcome=ValidationOutcome.NOT_EVALUATED,
                                            outcome_reason="resolved_by_precedence", validation_performed=False,
                                            citations=item.sources, exception_refs=(losers[item.id],)))
            continue

        predicates: tuple[PredicateResult, ...] = ()
        missing: tuple[str, ...] = ()
        cited = list(item.sources)
        if item.modality == NormativeModality.PERMITTED:
            outcome, reason, performed = ValidationOutcome.NOT_EVALUATED, "permission_has_no_compliance_check", False
        elif item.constraint is None:
            outcome, reason, performed = ValidationOutcome.NOT_EVALUATED, "no_executable_constraint", False
        else:
            outcome, reason, pred_results, missing_list, fact_evidence = evaluate_constraint(
                item.constraint, item.modality, facts,
            )
            predicates, missing, performed = tuple(pred_results), tuple(missing_list), True
            cited.extend(fact_evidence)
            for predicate in pred_results:
                if predicate.result in ("true", "false"):
                    facts_used[predicate.fact] = _bounded_repr(facts[predicate.fact].value)
            for pr in pred_results:
                if pr.result in ("unknown", "invalid"):
                    missing_evidence.append({"item": item.key, "fact": pr.fact, "reason": pr.reason})
        conflict_ids = tuple(unresolved_by_item.get(item.id, ()))
        if conflict_ids:
            outcome, reason = ValidationOutcome.CONFLICT, "unresolved_conflict"
        for ref in cited:
            citations[(ref.source_id, ref.locator)] = ref
        results.append(ConstraintResult(
            **base, outcome=outcome, outcome_reason=reason, validation_performed=performed,
            predicates=predicates, missing_facts=missing, citations=tuple(cited),
            conflict_ids=conflict_ids,
        ))

    for applied_exc in applied:
        for ref in applied_exc.citations:
            citations[(ref.source_id, ref.locator)] = ref

    # -- decision --------------------------------------------------------------
    blocking_results = [r for r in results if r.blocking]
    summary = {
        "unresolved_conflicts": sum(1 for c in conflicts if not c.resolved),
        "unresolved_exceptions": unresolved_exceptions,
        "blocking_failed": sum(1 for r in blocking_results if r.outcome == ValidationOutcome.FAIL),
        "blocking_applicability_unknown": sum(
            1 for r in blocking_results if r.applicability == ApplicabilityStatus.UNKNOWN),
        "blocking_insufficient": sum(
            1 for r in blocking_results if r.outcome == ValidationOutcome.INSUFFICIENT_EVIDENCE),
        "blocking_unchecked": sum(
            1 for r in blocking_results if r.outcome_reason == "no_executable_constraint"),
        "expected_knowledge_missing": int(
            expect_applicable_knowledge and not applying),
    }
    if summary["unresolved_conflicts"]:
        review_reasons.extend(
            f"unresolved {c.kind} between {' and '.join(c.item_keys)}" for c in conflicts if not c.resolved)
    for r in blocking_results:
        if r.applicability == ApplicabilityStatus.UNKNOWN:
            codes = sorted({reason.code for reason in r.applicability_reasons})
            review_reasons.append(f"{r.item_id}@v{r.item_version} applicability unknown ({', '.join(codes)})")
        elif r.outcome == ValidationOutcome.INSUFFICIENT_EVIDENCE:
            review_reasons.append(f"{r.item_id}@v{r.item_version} lacks evidence for {', '.join(r.missing_facts)}")
        elif r.outcome_reason == "no_executable_constraint":
            review_reasons.append(f"{r.item_id}@v{r.item_version} has no machine-checkable constraint")
    if summary["expected_knowledge_missing"]:
        review_reasons.append("no published knowledge applies to this context")

    policy = escalation_policy(tenant)
    decision_seed = content_digest({
        "tenant": tenant, "domain": domain, "release": release.model_dump(mode="json"),
        "context": context.model_dump(mode="json"),
        "facts": {k: v.model_dump(mode="json") for k, v in sorted(facts.items())},
        "results": [r.model_dump(mode="json") for r in results],
    })
    evaluation = evaluate_policy(
        policy, {"summary": summary},
        decision_id=f"KD-{decision_seed[:16]}", evaluation_id=f"KE-{decision_seed[:16]}",
    )
    if evaluation.result == PolicyResult.ESCALATE:
        overall = OverallOutcome.NEEDS_REVIEW
    elif evaluation.result == PolicyResult.DENY:
        overall = OverallOutcome.NON_COMPLIANT
    elif not applying:
        overall = OverallOutcome.NO_APPLICABLE_KNOWLEDGE
    else:
        overall = OverallOutcome.COMPLIANT

    return EvaluationDecision(
        decision_id=f"KD-{decision_seed[:16]}", tenant=tenant, domain=domain, release=release,
        context=context, facts_used=facts_used,
        knowledge=tuple(sorted(item.key for item in visible)),
        applicable=tuple(sorted(item.key for item in applying)),
        unknown_applicability=tuple(sorted(
            item.key for item in visible if applicability[item.id].status == ApplicabilityStatus.UNKNOWN)),
        results=tuple(results), conflicts=tuple(conflicts), exceptions_applied=tuple(applied),
        missing_evidence=tuple(missing_evidence),
        citations=tuple(sorted(citations.values(), key=lambda ref: (ref.source_id, ref.locator))),
        overall=overall, human_review_required=bool(review_reasons),
        human_review_reasons=tuple(dict.fromkeys(review_reasons)),
        escalation=EscalationResult(
            policy_id=ESCALATION_POLICY_ID, policy_version=ESCALATION_POLICY_VERSION,
            result=evaluation.result, matched_rule=evaluation.matched_rule,
            reason_code=evaluation.reason_code,
        ),
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "AppliedException", "ConstraintResult", "EscalationResult", "EvaluationDecision", "KnowledgeConflict",
    "KnowledgeReleaseRef", "OverallOutcome", "PredicateResult", "ValidationOutcome", "can_read",
    "detect_conflicts", "escalation_policy", "evaluate_constraint", "evaluate_knowledge",
    "evaluate_predicate", "visible_exceptions", "visible_items",
]
