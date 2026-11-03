"""Durable decision records: replay, explanation fidelity, re-review impact.

A *decision record* stores what is needed to reproduce one evaluation exactly:
the decision, the release it cites (id and content hash), and the inputs (the
business context, the full facts with evidence, the caller's resolved access
context, and the evaluation options).  Records are tenant-scoped and
append-only in the governance store (``record_knowledge_decision``); the same
inputs, access and release give the same ``record_id``, so recording is
idempotent and the first ``recorded_at`` is kept.

Three uses, all deterministic and offline:

* :func:`replay` re-runs a record against a release; :func:`replays_exactly`
  checks the record against the release it cites.
* :func:`influence` removes each visible item and exception in turn (an
  item together with the exceptions that name it, so every comparison is
  with a release the gate would accept) and reports which ones change the
  outcome ("decisive").  :func:`fidelity`
  checks the decision's own explanation against that: no decisive entry may
  be missing from what the decision cites.  The converse does not hold and is
  not required: a decision lists every applicable requirement, and a
  requirement that passes is usually not decisive on its own.
* :func:`review_impact` answers "which past decisions need another look"
  after a release change, when an exception expires, or when a cited item's
  human-set ``review_due`` date has passed.  It keeps what was decided then
  apart from what would be decided now; missing evidence, pending review and
  refusal stay distinct outcomes.

Records hold the caller's access context so a replay sees exactly what the
caller saw.  Reading the log is therefore an administrative operation (see
``api/routes/knowledge.py``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from graphrag.context_graph.knowledge import BusinessContext, Fact, KnowledgeException, KnowledgeItem, content_digest
from graphrag.context_graph.knowledge_evaluation import (
    EvaluationDecision, KnowledgeReleaseRef, evaluate_knowledge, visible_exceptions, visible_items,
)
from graphrag.enterprise.models import AccessContext

DECISION_RECORD_SCHEMA = "knowledge-decision-record/v1"


def decision_record(
    decision: EvaluationDecision, *, access: AccessContext, context: BusinessContext,
    facts: Mapping[str, Fact] | None = None, expect_applicable_knowledge: bool = False,
    semantic_candidates: list[str] | None = None, recorded_at: datetime | None = None,
) -> dict[str, Any]:
    facts = dict(facts or {})
    inputs = {
        "context": context.model_dump(mode="json"),
        "facts": {name: fact.model_dump(mode="json") for name, fact in sorted(facts.items())},
        "expect_applicable_knowledge": bool(expect_applicable_knowledge),
        "semantic_candidates": list(semantic_candidates) if semantic_candidates is not None else None,
        "access": access.model_dump(mode="json"),
    }
    record_id = "KR-" + content_digest({"decision": decision.decision_id, "access": access.fingerprint,
                                         "inputs": inputs})[:24]
    return {
        "schema_version": DECISION_RECORD_SCHEMA, "record_id": record_id, "decision_id": decision.decision_id,
        "tenant": decision.tenant, "domain": decision.domain,
        "release_id": decision.release.release_id, "content_hash": decision.release.content_hash,
        "recorded_at": (recorded_at or datetime.now(timezone.utc)).isoformat(),
        "subject": access.subject_id, "inputs": inputs, "decision": decision.model_dump(mode="json"),
    }


@dataclass(frozen=True)
class _Inputs:
    context: BusinessContext
    facts: dict[str, Fact]
    access: AccessContext
    expect: bool
    semantic: list[str] | None


def _inputs(record: Mapping[str, Any]) -> _Inputs:
    raw = record["inputs"]
    return _Inputs(
        context=BusinessContext.model_validate(raw["context"]),
        facts={name: Fact.model_validate(value) for name, value in raw["facts"].items()},
        access=AccessContext.model_validate(raw["access"]),
        expect=bool(raw.get("expect_applicable_knowledge", False)),
        semantic=raw.get("semantic_candidates"),
    )


def _evaluate(items, exceptions, release: KnowledgeReleaseRef, record: Mapping[str, Any],
              inputs: _Inputs, context: BusinessContext | None = None) -> EvaluationDecision:
    return evaluate_knowledge(
        items=items, exceptions=exceptions, context=context or inputs.context, facts=inputs.facts,
        tenant=str(record["tenant"]), access=inputs.access, domain=str(record["domain"]), release=release,
        expect_applicable_knowledge=inputs.expect, semantic_candidates=inputs.semantic,
    )


def replay(record: Mapping[str, Any], catalog, *, as_of: datetime | None = None) -> EvaluationDecision:
    """Re-run a record's inputs against ``catalog`` (a ``ReleaseCatalog``),
    optionally viewed at another valid time ``as_of``."""
    inputs = _inputs(record)
    context = inputs.context
    if as_of is not None:
        context = context.model_copy(update={"as_of": as_of})
    return _evaluate(catalog.items, catalog.exceptions, catalog.ref, record, inputs, context)


def replays_exactly(record: Mapping[str, Any], catalog) -> bool:
    """Whether the record is reproduced byte for byte by the release it cites."""
    if catalog.ref.release_id != record["release_id"] or catalog.ref.content_hash != record["content_hash"]:
        return False
    return replay(record, catalog).model_dump(mode="json") == record["decision"]


# ---------------------------------------------------------------------------
# explanation fidelity (counterfactual influence)


def _signature(decision: EvaluationDecision) -> tuple:
    failing = tuple(sorted(r.item_id for r in decision.results if r.outcome.value == "fail"))
    return decision.overall.value, decision.human_review_required, failing


def cited_keys(decision: EvaluationDecision) -> set[str]:
    """Every item or exception the decision itself names as bearing on it."""
    cited = set(decision.applicable) | set(decision.unknown_applicability)
    cited |= {applied.exception_key for applied in decision.exceptions_applied}
    cited |= set(decision.exceptions_unresolved)
    for result in decision.results:
        cited |= set(result.exception_refs) | set(result.partial_exception_refs)
    for conflict in decision.conflicts:
        cited |= set(conflict.item_keys)
        if conflict.resolution_ref:
            cited.add(conflict.resolution_ref)
    return cited


@dataclass(frozen=True)
class Influence:
    key: str
    kind: str          # "item" | "exception"
    cited: bool
    decisive: bool
    overall_without: str


def influence(
    *, items: Iterable[KnowledgeItem], exceptions: Iterable[KnowledgeException] = (), context: BusinessContext,
    facts: Mapping[str, Fact] | None = None, tenant: str, access: AccessContext, domain: str,
    release: KnowledgeReleaseRef, expect_applicable_knowledge: bool = False,
    semantic_candidates: list[str] | None = None,
) -> tuple[EvaluationDecision, list[Influence]]:
    """Remove each visible item and exception once; report which change the
    outcome (overall result, review flag or the set of failing requirements).
    Costs one evaluation per visible entry: an audit tool, not a request path."""
    items, exceptions = list(items), list(exceptions)
    common = dict(context=context, facts=facts, tenant=tenant, access=access, domain=domain, release=release,
                  expect_applicable_knowledge=expect_applicable_knowledge,
                  semantic_candidates=semantic_candidates)
    base = evaluate_knowledge(items=items, exceptions=exceptions, **common)
    reference = _signature(base)
    cited = cited_keys(base)
    rows: list[Influence] = []
    for item in visible_items(items, tenant=tenant, access=access):
        # Only valid releases are compared: the publication gate rejects an
        # exception whose target or winner is missing, so removing an item
        # also removes the exceptions that name it.
        remaining = [e for e in exceptions
                     if item.id not in e.target_item_ids and e.winner_item_id != item.id]
        without = evaluate_knowledge(items=[i for i in items if i.id != item.id], exceptions=remaining, **common)
        rows.append(Influence(item.key, "item", item.key in cited, _signature(without) != reference,
                              without.overall.value))
    for exc in visible_exceptions(exceptions, tenant=tenant, access=access):
        without = evaluate_knowledge(items=items, exceptions=[e for e in exceptions if e.id != exc.id], **common)
        rows.append(Influence(exc.key, "exception", exc.key in cited, _signature(without) != reference,
                              without.overall.value))
    return base, rows


def fidelity(rows: Iterable[Influence]) -> dict[str, Any]:
    """Faithful when no decisive entry is missing from the decision's citations."""
    rows = list(rows)
    uncited = sorted(row.key for row in rows if row.decisive and not row.cited)
    return {
        "entries": len(rows), "cited": sum(row.cited for row in rows),
        "decisive": sum(row.decisive for row in rows), "uncited_decisive": uncited,
        "faithful": not uncited,
    }


# ---------------------------------------------------------------------------
# re-review impact


@dataclass(frozen=True)
class ReviewImpact:
    record_id: str
    decision_id: str
    recorded_release: str
    triggers: tuple[str, ...]
    then: dict[str, Any]
    now: dict[str, Any]
    changed: bool
    changed_items: tuple[str, ...] = ()
    review_due_items: tuple[str, ...] = ()
    expiring_exceptions: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _summary(decision: Mapping[str, Any]) -> dict[str, Any]:
    return {"overall": decision["overall"], "human_review_required": decision["human_review_required"],
            "release_id": decision["release"]["release_id"]}


def _outcomes(decision: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    return {r["item_id"]: (r["applicability"], r["outcome"]) for r in decision["results"]}


def expiring_exceptions(catalog, *, start: datetime, end: datetime) -> list[str]:
    """Exception keys whose validity ends within ``(start, end]``."""
    return sorted(e.key for e in catalog.exceptions
                  if e.applicability.valid_to is not None and start < e.applicability.valid_to <= end)


def review_impact(
    records: Iterable[Mapping[str, Any]], catalog, *, as_of: datetime | None = None,
    now: datetime | None = None, expiring: Iterable[str] = (),
) -> list[ReviewImpact]:
    """Re-evaluate stored decisions against ``catalog`` and report what moved.

    ``as_of`` views every decision at another valid time (e.g. the end of an
    expiry window); ``now`` is the reference for ``review_due``; ``expiring``
    names exceptions about to lapse (see :func:`expiring_exceptions`).
    """
    now = now or datetime.now(timezone.utc)
    expiring = set(expiring)
    due = {item.id for item in catalog.items if item.review_due is not None and item.review_due <= now}
    impacts: list[ReviewImpact] = []
    for record in records:
        then = record["decision"]
        current = replay(record, catalog, as_of=as_of).model_dump(mode="json")
        before, after = _outcomes(then), _outcomes(current)
        changed_items = tuple(sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k)))
        applied = {e["exception_key"] for e in then.get("exceptions_applied", ())}
        cited_ids = {key.split("@v")[0] for key in (*then["applicable"], *then["unknown_applicability"])}
        due_items = tuple(sorted(cited_ids & due))
        lapsing = tuple(sorted(applied & expiring))
        triggers = []
        if catalog.ref.release_id != record["release_id"]:
            triggers.append("release_changed")
        if as_of is not None:
            triggers.append("valid_time_moved")
        if due_items:
            triggers.append("review_due")
        if lapsing:
            triggers.append("exception_expiring")
        changed = (_summary(then) | {"release_id": ""}) != (_summary(current) | {"release_id": ""}) or bool(changed_items)
        impacts.append(ReviewImpact(
            record_id=str(record["record_id"]), decision_id=str(record["decision_id"]),
            recorded_release=str(record["release_id"]), triggers=tuple(triggers),
            then=_summary(then), now=_summary(current), changed=changed, changed_items=changed_items,
            review_due_items=due_items, expiring_exceptions=lapsing,
        ))
    return impacts


def needs_review(impacts: Iterable[ReviewImpact]) -> list[ReviewImpact]:
    """Decisions whose outcome moved, or that cite a due item or lapsing exception."""
    return [i for i in impacts if i.changed or i.review_due_items or i.expiring_exceptions]


__all__ = [
    "DECISION_RECORD_SCHEMA", "Influence", "ReviewImpact", "cited_keys", "decision_record",
    "expiring_exceptions", "fidelity", "influence", "needs_review", "replay", "replays_exactly", "review_impact",
]
