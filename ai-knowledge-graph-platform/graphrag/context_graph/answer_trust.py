"""Answer-trust investigation: bundle, classify, track to a fix.

When a user marks a decision as not trusted, one bundle is assembled from the
decision log: the question (business context), the evidence cited, the
permission filters that applied, the release and definition versions, and the
trace (decision and record ids, whether it replays exactly).  The bundle is
built **under the reporting user's access**: knowledge the reporter cannot
read is omitted (only a count of hidden entries remains), and the submitted
facts are included only when the reporter is the user who asked.

The root cause is classified deterministically from a fixed list, first match
wins:

1. ``permission_filter``  restricted knowledge that applies here was hidden
   from the caller (for example groups were not resolved);
2. ``wrong_or_missing_definition``  an applicable item names a definition that
   is missing, unpublished or has a newer version now (checked before
   staleness: a changed definition always makes a new release, and the
   definition is the more specific cause);
3. ``stale_source``       the cited release is no longer active, or a cited
   item is past its human-set review date;
4. ``retrieval_miss``     no knowledge applied to the question;
5. ``extraction_error``   inputs were missing, unevidenced or of the wrong type;
6. ``model_error``        only for answers produced by a model (not for
   deterministic knowledge decisions);
7. ``expectation_mismatch``  the system did what its knowledge says: the
   decision replays exactly against a current release with complete inputs.

Counts per class and time to fix are computed from the stored reports.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from statistics import median
from typing import Any

from graphrag.context_graph.knowledge import BusinessContext, content_digest
from graphrag.context_graph.knowledge_decisions import replays_exactly
from graphrag.context_graph.knowledge_evaluation import can_read, visible_items
from graphrag.context_graph.applicability import ApplicabilityStatus, evaluate_applicability
from graphrag.enterprise.models import AccessContext

ROOT_CAUSES = (
    "permission_filter", "wrong_or_missing_definition", "stale_source", "retrieval_miss",
    "extraction_error", "model_error", "expectation_mismatch",
)


def assemble_bundle(
    record: Mapping[str, Any], *, cited_catalog, active_catalog, reporter: AccessContext,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Everything needed to investigate one decision, as the reporter may see it."""
    now = now or datetime.now(timezone.utc)
    decision = record["decision"]
    tenant = str(record["tenant"])
    readable = {item.id for item in visible_items(cited_catalog.items, tenant=tenant, access=reporter)}
    results = [r for r in decision["results"] if r["item_id"] in readable]
    used = {r["item_id"] for r in decision["results"]}
    is_asker = reporter.subject_id == record["subject"]
    original_access = AccessContext.model_validate(record["inputs"]["access"])
    return {
        "record_id": record["record_id"], "decision_id": record["decision_id"], "tenant": tenant,
        "domain": record["domain"], "recorded_at": record["recorded_at"],
        "question": {"context": record["inputs"]["context"],
                     "facts": record["inputs"]["facts"] if is_asker else sorted(record["inputs"]["facts"])},
        "evidence": sorted({(c["source_id"], c.get("locator", "")) for r in results for c in r["citations"]}),
        "results": [{"item_id": r["item_id"], "applicability": r["applicability"], "outcome": r["outcome"],
                     "outcome_reason": r["outcome_reason"]} for r in results],
        "overall": decision["overall"], "human_review_required": decision["human_review_required"],
        "missing_evidence": [m for m in decision.get("missing_evidence", ())
                             if m["item"].split("@v")[0] in readable],
        "permission_filters": {
            "groups_resolved": original_access.groups_resolved,
            "hidden_entries": len(used - readable),
        },
        "release": {
            "cited": {"release_id": record["release_id"], "content_hash": record["content_hash"]},
            "active": {"release_id": active_catalog.ref.release_id, "content_hash": active_catalog.ref.content_hash},
            "current": record["content_hash"] == active_catalog.ref.content_hash,
        },
        "definitions_used": list(decision.get("definitions_used", ())),
        "replays_exactly": replays_exactly(record, cited_catalog),
        "assembled_at": now.isoformat(),
    }


def classify(
    record: Mapping[str, Any], bundle: Mapping[str, Any], *, cited_catalog, active_catalog,
    answer_source: str = "knowledge_decision", now: datetime | None = None,
) -> str:
    """The first matching root cause (see module docstring)."""
    now = now or datetime.now(timezone.utc)
    decision = record["decision"]
    tenant = str(record["tenant"])
    original = AccessContext.model_validate(record["inputs"]["access"])
    context = BusinessContext.model_validate(record["inputs"]["context"])

    hidden_applying = [
        item for item in cited_catalog.items
        if item.tenant == tenant and not can_read(item.access, original)
        and evaluate_applicability(item.applicability, context, recorded_at=item.recorded_at).status
        == ApplicabilityStatus.APPLIES
    ]
    if hidden_applying:
        return "permission_filter"

    cited_ids = {key.split("@v")[0] for key in (*decision["applicable"], *decision["unknown_applicability"])}
    defined = {d.id: d for d in getattr(cited_catalog, "definitions", ())}
    newest = {}
    for d in getattr(active_catalog, "definitions", ()):
        newest[d.id] = max(newest.get(d.id, 0), d.version)
    for item in cited_catalog.items:
        if item.id not in cited_ids:
            continue
        for ref in getattr(item, "definition_refs", ()):
            current = defined.get(ref)
            if current is None or current.status.value != "published" or newest.get(ref, 0) > current.version:
                return "wrong_or_missing_definition"

    due = any(item.id in cited_ids and item.review_due is not None and item.review_due <= now
              for item in cited_catalog.items)
    if not bundle["release"]["current"] or due:
        return "stale_source"

    if decision["overall"] == "no_applicable_knowledge" or not decision["applicable"]:
        return "retrieval_miss"
    if decision.get("missing_evidence"):
        return "extraction_error"
    if answer_source != "knowledge_decision":
        return "model_error"
    return "expectation_mismatch"


def trust_report(
    record: Mapping[str, Any], *, reporter: AccessContext, reason: str, root_cause: str, bundle: Mapping[str, Any],
    reported_at: datetime | None = None,
) -> dict[str, Any]:
    if root_cause not in ROOT_CAUSES:
        raise ValueError(f"unknown root cause {root_cause!r}")
    reported_at = reported_at or datetime.now(timezone.utc)
    return {
        "report_id": "TR-" + content_digest({"record": record["record_id"], "reporter": reporter.subject_id,
                                              "at": reported_at.isoformat()})[:24],
        "tenant": record["tenant"], "domain": record["domain"], "record_id": record["record_id"],
        "reporter": reporter.subject_id, "reason": reason[:2000], "root_cause": root_cause,
        "reported_at": reported_at.isoformat(), "status": "open", "fixed_at": None, "fixed_by": None,
        "bundle": dict(bundle),
    }


def trust_statistics(reports: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts per root cause and median hours from report to fix."""
    reports = list(reports)
    counts = {cause: 0 for cause in ROOT_CAUSES}
    hours: dict[str, list[float]] = {cause: [] for cause in ROOT_CAUSES}
    for report in reports:
        counts[report["root_cause"]] += 1
        if report.get("fixed_at"):
            opened = datetime.fromisoformat(report["reported_at"])
            fixed = datetime.fromisoformat(report["fixed_at"])
            hours[report["root_cause"]].append((fixed - opened).total_seconds() / 3600)
    return {
        "reports": len(reports), "open": sum(r["status"] == "open" for r in reports),
        "by_root_cause": counts,
        "median_hours_to_fix": {cause: round(median(v), 2) if v else None for cause, v in hours.items()},
    }


__all__ = ["ROOT_CAUSES", "assemble_bundle", "classify", "trust_report", "trust_statistics"]
