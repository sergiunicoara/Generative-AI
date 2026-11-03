"""Knowledge packs, review, and gated releases.

A *knowledge pack* is the version-controlled curation surface for one domain
and tenant: items and exceptions in every lifecycle state (draft, in review,
approved, published, withdrawn).  A *knowledge release* is the immutable,
content-addressed snapshot of the pack's **published** entries that the
evaluator serves.  Releases are persisted in the existing durable governance
store (``GovernanceStore.publish_knowledge_release``) with the same
append-only + active-pointer discipline as Energy publications.

Publication runs gate checks before anything is written:

* contract checks (Pydantic validation of every entry, done at load time);
* version checks: at most one published version per item or exception id,
  and it must be the latest version of that id;
* referential checks: supersession, override and exception targets exist and
  are published; no supersession cycles; exceptions target published items;
* SHACL: the RDF projection of the published content must conform to the
  shapes generated from ``ontology/models/contextual-knowledge.yaml``;
* regression (mandatory -- an empty case set is refused): the domain's
  deterministic evaluation cases must reproduce their expected applicability,
  outcomes, conflicts and escalation against the *candidate* release.

The release ``content_hash`` covers everything served, including pack
metadata and the disclaimer.

Any failure raises :class:`KnowledgePublicationError` and leaves the active
release untouched.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
import yaml

from graphrag.context_graph.knowledge import (
    BusinessContext, Fact, KnowledgeException, KnowledgeItem, PublicationStatus, content_digest,
)
from graphrag.context_graph.knowledge_evaluation import (
    EvaluationDecision, KnowledgeReleaseRef, evaluate_knowledge,
)
from graphrag.context_graph.knowledge_rdf import projection_failures
from graphrag.context_graph.knowledge_render import render_item
from graphrag.enterprise.models import AccessContext

RELEASE_SCHEMA_VERSION = "knowledge-release/v1"


class KnowledgePublicationError(RuntimeError):
    """A candidate release failed its gate; the active release is unchanged."""

    def __init__(self, message: str, failures: list[str] | None = None, *,
                 issues: list["GateIssue"] | None = None):
        issues = list(issues or [])
        failures = list(failures) if failures is not None else [issue.message for issue in issues]
        super().__init__(message + ": " + "; ".join(failures[:10]))
        self.failures = failures
        self.issues = issues


class KnowledgePack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    tenant: str = Field(min_length=1, max_length=128)
    pack_version: str = Field(min_length=1, max_length=64)
    description: str = ""
    disclaimer: str = ""
    items: list[KnowledgeItem] = Field(default_factory=list)
    exceptions: list[KnowledgeException] = Field(default_factory=list)

    def published_items(self) -> list[KnowledgeItem]:
        return [i for i in self.items if i.status == PublicationStatus.PUBLISHED]

    def published_exceptions(self) -> list[KnowledgeException]:
        return [e for e in self.exceptions if e.status == PublicationStatus.PUBLISHED]


def load_pack(path: str | Path) -> KnowledgePack:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    # ``x-*`` keys hold YAML anchors shared by entries; they are not content.
    raw = {key: value for key, value in raw.items() if not str(key).startswith("x-")}
    return KnowledgePack.model_validate(raw)


# ---------------------------------------------------------------------------
# review (curation) helpers


def review_queue(pack: KnowledgePack) -> list[dict[str, Any]]:
    """Entries awaiting a human decision, with what they still need."""
    queue = []
    for item in pack.items:
        if item.status in (PublicationStatus.DRAFT, PublicationStatus.IN_REVIEW):
            needs = []
            if item.constraint is None:
                needs.append("executable constraint (optional; otherwise not machine-checkable)")
            missing_dims = [d for d in ("customer", "brand", "market", "product", "product_family", "workflow")
                            if item.applicability.scope(d) is None]
            if missing_dims:
                needs.append("applicability for " + ", ".join(missing_dims))
            if not item.sources:
                needs.append("a source citation")
            queue.append({"key": item.key, "status": item.status.value, "origin": item.origin.value,
                          "title": item.title, "statement": item.statement,
                          "executable": render_item(item), "needs": needs})
    return queue


def approve_item(
    item: KnowledgeItem, *, reviewed_by: str, review_ref: str, publish: bool = True,
) -> KnowledgeItem:
    """Return the next version of an item, approved by a trusted reviewer.

    ``reviewed_by`` must come from the authenticated identity (token ``sub``),
    never from a request body.  Validation of the new version re-runs the
    contract checks, so an extracted item with a constraint but no
    constraint provenance still cannot be approved.
    """
    if not reviewed_by or not review_ref:
        raise ValueError("approval needs a reviewer identity and a review reference")
    data = item.model_dump()
    data.update(
        version=item.version + 1, reviewed_by=reviewed_by, review_ref=review_ref,
        status=PublicationStatus.PUBLISHED if publish else PublicationStatus.APPROVED,
        recorded_at=datetime.now(timezone.utc),
    )
    return KnowledgeItem.model_validate(data)


def withdraw_item(item: KnowledgeItem, *, reviewed_by: str, review_ref: str) -> KnowledgeItem:
    data = item.model_dump()
    data.update(version=item.version + 1, status=PublicationStatus.WITHDRAWN,
                reviewed_by=reviewed_by, review_ref=review_ref, recorded_at=datetime.now(timezone.utc))
    return KnowledgeItem.model_validate(data)


# ---------------------------------------------------------------------------
# regression cases


class RegressionExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applicable: list[str] = Field(default_factory=list)
    unknown: list[str] | None = None
    outcomes: dict[str, str] = Field(default_factory=dict)
    conflicts: list[dict[str, Any]] | None = None
    human_review_required: bool | None = None
    overall: str | None = None
    citations: dict[str, list[str]] = Field(default_factory=dict)


class RegressionCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str = ""
    covers: list[str] = Field(default_factory=list)
    tenant: str
    principal: dict[str, Any] = Field(default_factory=lambda: {"sub": "eval-runner"})
    context: dict[str, Any] = Field(default_factory=dict)
    facts: dict[str, Any] = Field(default_factory=dict)
    expect_applicable_knowledge: bool = False
    semantic_candidates: list[str] | None = None
    expected: RegressionExpectation


def load_cases(path: str | Path, *, domain: str | None = None) -> list[RegressionCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if domain is not None and raw.get("domain") not in (None, domain):
        return []
    return [RegressionCase.model_validate(case) for case in raw["cases"]]


def run_case(
    case: RegressionCase, *, items: Iterable[KnowledgeItem], exceptions: Iterable[KnowledgeException],
    domain: str, release: KnowledgeReleaseRef,
) -> EvaluationDecision:
    access = AccessContext.from_claims(case.principal)
    context = BusinessContext.model_validate(case.context)
    facts = {name: Fact.model_validate(value) for name, value in case.facts.items()}
    return evaluate_knowledge(
        items=items, exceptions=exceptions, context=context, facts=facts, tenant=case.tenant,
        access=access, domain=domain, release=release,
        expect_applicable_knowledge=case.expect_applicable_knowledge,
        semantic_candidates=case.semantic_candidates,
    )


def case_mismatches(case: RegressionCase, decision: EvaluationDecision) -> list[str]:
    """Differences between a decision and a case's expectation (empty = pass)."""
    exp = case.expected
    out: list[str] = []
    applicable = sorted(key.split("@v")[0] for key in decision.applicable)
    if applicable != sorted(exp.applicable):
        out.append(f"{case.id}: applicable {applicable} != {sorted(exp.applicable)}")
    if exp.unknown is not None:
        unknown = sorted(key.split("@v")[0] for key in decision.unknown_applicability)
        if unknown != sorted(exp.unknown):
            out.append(f"{case.id}: unknown {unknown} != {sorted(exp.unknown)}")
    by_item = {r.item_id: r for r in decision.results}
    for item_id, outcome in exp.outcomes.items():
        actual = by_item[item_id].outcome.value if item_id in by_item else "absent"
        if actual != outcome:
            out.append(f"{case.id}: {item_id} outcome {actual} != {outcome}")
    if exp.conflicts is not None:
        actual_conflicts = sorted(
            (c.kind, tuple(sorted(k.split("@v")[0] for k in c.item_keys)), c.resolved)
            for c in decision.conflicts)
        expected_conflicts = sorted(
            (c["kind"], tuple(sorted(c["items"])), bool(c.get("resolved", False))) for c in exp.conflicts)
        if actual_conflicts != expected_conflicts:
            out.append(f"{case.id}: conflicts {actual_conflicts} != {expected_conflicts}")
    if exp.human_review_required is not None and decision.human_review_required != exp.human_review_required:
        out.append(f"{case.id}: human_review_required {decision.human_review_required} != "
                   f"{exp.human_review_required}")
    if exp.overall is not None and decision.overall.value != exp.overall:
        out.append(f"{case.id}: overall {decision.overall.value} != {exp.overall}")
    for item_id, sources in exp.citations.items():
        cited = {c.source_id for c in by_item[item_id].citations} if item_id in by_item else set()
        if not set(sources) <= cited:
            out.append(f"{case.id}: {item_id} citations {sorted(cited)} missing {sorted(set(sources) - cited)}")
    return out


# ---------------------------------------------------------------------------
# publication gate


@dataclass(frozen=True)
class GateIssue:
    """One gate failure in machine-readable form: a stable ``code``, the
    ``path`` of the offending entry, the human ``message`` (the same text the
    gate has always produced) and the smallest corrective ``hint``."""

    code: str
    path: str
    message: str
    hint: str


_HINTS = {
    "multiple_published_versions": "Keep one published version per id; withdraw the older ones (withdraw_item).",
    "published_version_not_latest": "Publish the latest version, or drop the newer unpublished or withdrawn one.",
    "tenant_mismatch": "Set the entry tenant to the pack tenant, or move the entry to that tenant pack.",
    "supersedes_unknown_item": "Add the superseded item to the pack, or remove its id from supersedes.",
    "overrides_unknown_item": "Add the overridden item to the pack, or remove its id from overrides.",
    "supersession_cycle": "Remove one supersedes edge so revisions form a chain.",
    "exception_target_unpublished": "Publish the target item, or remove it from target_item_ids/winner_item_id.",
    "shacl_violation": ("Fix the field named in the message on the published entry; the shapes are generated "
                        "from ontology/models/contextual-knowledge.yaml."),
    "regression_mismatch": ("Fix the pack, or change the case expectation only after review: cases encode "
                            "intended behaviour."),
    "regression_cases_missing": "Supply the domain regression cases (evals/knowledge_eval/<domain>_cases.json).",
}


def _issue(code: str, path: str, message: str) -> GateIssue:
    return GateIssue(code=code, path=path, message=message, hint=_HINTS[code])


def _version_issues(entries: Iterable, kind: str) -> list[GateIssue]:
    """Exactly one effective version per id: at most one published version,
    and it must be the latest -- a newer withdrawn or unpublished version
    cannot leave an older one in force."""
    by_id: dict[str, list] = {}
    for entry in entries:
        by_id.setdefault(entry.id, []).append(entry)
    issues: list[GateIssue] = []
    for entry_id, versions in sorted(by_id.items()):
        path = f"{kind}s/{entry_id}"
        published = sorted(e.version for e in versions if e.status == PublicationStatus.PUBLISHED)
        if len(published) > 1:
            issues.append(_issue("multiple_published_versions", path,
                                 f"{kind} {entry_id}: more than one published version {published}"))
        latest = max(versions, key=lambda e: e.version)
        if published and latest.status != PublicationStatus.PUBLISHED:
            issues.append(_issue("published_version_not_latest", path,
                                 f"{kind} {entry_id}: published v{published[-1]} is older than "
                                 f"v{latest.version} ({latest.status.value})"))
    return issues


def check_pack_issues(pack: KnowledgePack, cases: Iterable[RegressionCase] = ()) -> list[GateIssue]:
    """Gate failures for the pack's published content, structured (empty = pass)."""
    issues: list[GateIssue] = []
    published = pack.published_items()
    published_ids = {item.id for item in published}
    issues.extend(_version_issues(pack.items, "item"))
    issues.extend(_version_issues(pack.exceptions, "exception"))
    all_ids = {item.id for item in pack.items}
    for item in pack.items + pack.exceptions:  # type: ignore[operator]
        if item.tenant != pack.tenant:
            issues.append(_issue("tenant_mismatch", item.key,
                                 f"{item.key}: tenant {item.tenant!r} differs from pack tenant {pack.tenant!r}"))
    for item in published:
        for target in item.supersedes:
            if target not in all_ids:
                issues.append(_issue("supersedes_unknown_item", f"{item.key}/supersedes",
                                     f"{item.key}: supersedes unknown item {target}"))
        for target in item.overrides:
            if target not in all_ids:
                issues.append(_issue("overrides_unknown_item", f"{item.key}/overrides",
                                     f"{item.key}: overrides unknown item {target}"))
    # supersession cycles among published items
    graph = {item.id: set(item.supersedes) & published_ids for item in published}
    visiting: set[str] = set()
    done: set[str] = set()

    def _cycle(node: str) -> bool:
        if node in done:
            return False
        if node in visiting:
            return True
        visiting.add(node)
        found = any(_cycle(n) for n in graph.get(node, ()))
        visiting.discard(node)
        done.add(node)
        return found

    if any(_cycle(node) for node in graph):
        issues.append(_issue("supersession_cycle", "items", "supersession cycle among published items"))
    for exc in pack.published_exceptions():
        for target in (*exc.target_item_ids, *([exc.winner_item_id] if exc.winner_item_id else [])):
            if target not in published_ids:
                issues.append(_issue("exception_target_unpublished", f"{exc.key}/targets",
                                     f"{exc.key}: targets {target}, which is not a published item"))
    if not issues:
        # the projection assumes one effective version per id, checked above
        issues.extend(_issue("shacl_violation", "rdf", failure)
                      for failure in projection_failures(published, pack.published_exceptions()))
    if issues:
        return issues

    release = KnowledgeReleaseRef(release_id="candidate", domain=pack.domain, version="candidate")
    for case in cases:
        decision = run_case(case, items=published, exceptions=pack.published_exceptions(),
                            domain=pack.domain, release=release)
        issues.extend(_issue("regression_mismatch", f"cases/{case.id}", mismatch)
                      for mismatch in case_mismatches(case, decision))
    return issues


def check_pack(pack: KnowledgePack, cases: Iterable[RegressionCase] = ()) -> list[str]:
    """Return gate failures for the pack's published content (empty = pass)."""
    return [issue.message for issue in check_pack_issues(pack, cases)]


def release_record(pack: KnowledgePack, *, published_by: str, checks: Mapping[str, Any]) -> dict[str, Any]:
    items = [item.model_dump(mode="json") for item in sorted(pack.published_items(), key=lambda i: i.id)]
    exceptions = [e.model_dump(mode="json") for e in sorted(pack.published_exceptions(), key=lambda e: e.id)]
    # Everything the release serves is in its identity, including the
    # disclaimer and pack metadata: changing any of them is a new release.
    content_hash = content_digest({
        "schema": RELEASE_SCHEMA_VERSION, "tenant": pack.tenant, "domain": pack.domain,
        "pack_version": pack.pack_version, "description": pack.description,
        "disclaimer": pack.disclaimer, "items": items, "exceptions": exceptions,
    })
    return {
        "schema_version": RELEASE_SCHEMA_VERSION, "release_id": uuid4().hex, "tenant": pack.tenant,
        "domain": pack.domain, "pack_version": pack.pack_version, "content_hash": content_hash,
        "published_at": datetime.now(timezone.utc).isoformat(), "published_by": published_by,
        "rolled_back_from": None, "checks": dict(checks), "disclaimer": pack.disclaimer,
        "item_keys": [f"{i['id']}@v{i['version']}" for i in items],
        "exception_keys": [f"{e['id']}@v{e['version']}" for e in exceptions],
        "items": items, "exceptions": exceptions,
    }


async def publish_pack(
    store, pack: KnowledgePack, *, published_by: str, cases: Iterable[RegressionCase] = (),
) -> dict[str, Any]:
    """Gate, then persist and activate a release.  Raises on any gate failure.

    Regression cases are mandatory: a release whose behaviour was never
    checked is not recorded as having passed a regression gate.
    """
    if not published_by:
        raise ValueError("publication needs the publishing identity")
    cases = list(cases)
    rejected = f"knowledge release for {pack.tenant}/{pack.domain} rejected; the active release is unchanged"
    if not cases:
        raise KnowledgePublicationError(rejected, issues=[_issue(
            "regression_cases_missing", "cases",
            "no regression cases supplied; a release cannot pass an empty regression gate")])
    issues = check_pack_issues(pack, cases)
    if issues:
        raise KnowledgePublicationError(rejected, issues=issues)
    checks = {"contract": "passed", "referential": "passed", "regression_cases": len(cases),
              "regression": "passed"}
    return await store.publish_knowledge_release(release_record(pack, published_by=published_by, checks=checks))


@dataclass(frozen=True)
class ReleaseCatalog:
    """What the evaluator serves from one release."""

    ref: KnowledgeReleaseRef
    items: tuple[KnowledgeItem, ...]
    exceptions: tuple[KnowledgeException, ...] = field(default_factory=tuple)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ReleaseCatalog":
        return cls(
            ref=KnowledgeReleaseRef(
                release_id=str(record["release_id"]), domain=str(record["domain"]),
                version=record.get("version", ""), content_hash=str(record["content_hash"]),
            ),
            items=tuple(KnowledgeItem.model_validate(i) for i in record["items"]),
            exceptions=tuple(KnowledgeException.model_validate(e) for e in record.get("exceptions", [])),
        )


__all__ = [
    "GateIssue", "KnowledgePack", "KnowledgePublicationError", "RegressionCase", "ReleaseCatalog", "approve_item",
    "case_mismatches", "check_pack", "check_pack_issues", "load_cases", "load_pack", "publish_pack",
    "release_record",
    "review_queue", "run_case", "withdraw_item",
]
