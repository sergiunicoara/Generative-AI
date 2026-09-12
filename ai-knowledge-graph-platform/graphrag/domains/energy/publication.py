"""SHACL as a publication gate: stage a candidate RDF graph, validate it,
quarantine the records that violate the domain's shapes, publish the
conformant remainder as a new version, and support rolling back to any
earlier published version.

Closes a gap a follow-up platform review named explicitly: nothing in this
repo had a real stage -> validate -> quarantine -> publish -> rollback
lifecycle for an RDF dataset. `EnergyDemoService.graph`
(`graphrag/domains/energy/demo.py`) used to be a single in-memory graph,
built once and immediately "live" -- queries, `export_turtle()`, and the
`/rdf` route all read it directly, with no conformance gate anywhere.
`validate_candidate()` demonstrated SHACL rejection only against one
hardcoded synthetic record, by calling `pyshacl.validate()` directly and
bypassing `graphrag/graph/shacl_validator.py`'s `SHACLValidator` entirely.

This module uses the real `SHACLValidator` (now genuinely reusable via its
`shapes_path` parameter -- see that module) against the actual candidate
graph, not a synthetic probe.

Quarantine granularity and a stated limitation
------------------------------------------------
Validation runs once against the whole candidate graph; every `sh:Violation`
result's `focus_node` names a record (subject) to quarantine -- its own
triples are excluded from the published graph, and the reason is recorded.
Everything else publishes, including subjects no shape targets at all (a
`Site` or `Component`, say, which this domain's shapes don't constrain).

Known, deliberate limitation: quarantining a subject removes only *that
subject's own* triples. A still-published record that merely references a
quarantined subject (e.g. a valid WorkOrder's `energy:concernsAsset`
pointing at a quarantined Asset) is not itself quarantined or rewritten.
A referential-integrity cascade is materially bigger scope than this gate
is trying to close -- this is a bounded, disclosed choice, not a silently
cut corner.

Version history is in-memory and append-only, matching this session's
earlier audit-trail-style patterns (e.g. `OntologyEvent` nodes): a
`rollback()` never deletes or rewrites a prior entry, it appends a new one
whose content matches the target version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from rdflib import Graph

from graphrag.graph.shacl_validator import SHACLValidator


class PublicationRollbackError(RuntimeError):
    """Raised when `rollback()`'s target version does not exist -- including
    the case where only one version exists and no explicit target was given
    (there is nothing before the current version to roll back to)."""


@dataclass(frozen=True)
class QuarantinedRecord:
    subject: str
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PublicationReport:
    version_id: str
    published_at: str
    published_triple_count: int
    candidate_record_count: int
    quarantined_records: list[QuarantinedRecord] = field(default_factory=list)
    rolled_back_from: str | None = None

    @property
    def quarantined_count(self) -> int:
        return len(self.quarantined_records)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatasetPublisher:
    """Holds the version history for one RDF dataset (in-memory, per
    instance -- matches `EnergyDemoService`'s existing pure-in-memory,
    per-service-instance model; nothing here persists across processes)."""

    def __init__(self, shapes_path: Path) -> None:
        self._shapes_path = shapes_path
        # Append-only: (report, published_graph), oldest first. `current`/
        # `current_report` always read the last entry; nothing here ever
        # mutates or removes an earlier one.
        self._versions: list[tuple[PublicationReport, Graph]] = []

    def stage_and_publish(self, candidate: Graph) -> PublicationReport:
        """Validate `candidate` against this dataset's SHACL shapes, quarantine
        the records that violate them, and publish the conformant remainder
        as a new version. Always succeeds -- "publish" here means "publish
        whatever conforms," not "refuse to publish anything if one record is
        bad," matching the critique's own wording ("quarantine invalid
        records, publish complete version": the *complete* version is the
        valid subset, not a synthetic gate everyone must pass wholesale)."""
        report_shacl = SHACLValidator(candidate, shapes_path=self._shapes_path).validate_report(target="energy")

        violating_subjects: dict[str, list[str]] = {}
        for result in report_shacl.results:
            if result.severity != "Violation" or not result.focus_node:
                continue
            violating_subjects.setdefault(result.focus_node, []).append(result.message)

        candidate_subjects = {str(s) for s in candidate.subjects()}
        published = Graph()
        for prefix, namespace in candidate.namespaces():
            published.bind(prefix, namespace)
        for subject, predicate, obj in candidate:
            if str(subject) in violating_subjects:
                continue
            published.add((subject, predicate, obj))

        quarantined = [
            QuarantinedRecord(subject=subject, reasons=reasons)
            for subject, reasons in sorted(violating_subjects.items())
        ]
        report = PublicationReport(
            version_id=uuid4().hex,
            published_at=_now_iso(),
            published_triple_count=len(published),
            candidate_record_count=len(candidate_subjects),
            quarantined_records=quarantined,
        )
        self._versions.append((report, published))
        return report

    @property
    def current(self) -> Graph:
        if not self._versions:
            raise PublicationRollbackError("no version has been published yet")
        return self._versions[-1][1]

    @property
    def current_report(self) -> PublicationReport:
        if not self._versions:
            raise PublicationRollbackError("no version has been published yet")
        return self._versions[-1][0]

    def history(self) -> list[PublicationReport]:
        """Every report published so far, oldest first -- a read-only view;
        callers cannot mutate the publisher's own history through this."""
        return [report for report, _graph in self._versions]

    def rollback(self, version_id: str | None = None) -> PublicationReport:
        """Roll back to `version_id`, or the version immediately before the
        current one when omitted. Appends a NEW version entry carrying the
        target's exact graph content (`rolled_back_from` records which
        version it came from) -- history is never rewritten."""
        if not self._versions:
            raise PublicationRollbackError("no version has been published yet")

        if version_id is None:
            if len(self._versions) < 2:
                raise PublicationRollbackError(
                    "only one version has been published; there is nothing to roll back to"
                )
            target_report, target_graph = self._versions[-2]
        else:
            match = next((entry for entry in self._versions if entry[0].version_id == version_id), None)
            if match is None:
                raise PublicationRollbackError(f"no published version with id {version_id!r}")
            target_report, target_graph = match

        restored = Graph()
        for prefix, namespace in target_graph.namespaces():
            restored.bind(prefix, namespace)
        for triple in target_graph:
            restored.add(triple)

        new_report = PublicationReport(
            version_id=uuid4().hex,
            published_at=_now_iso(),
            published_triple_count=target_report.published_triple_count,
            candidate_record_count=target_report.candidate_record_count,
            quarantined_records=target_report.quarantined_records,
            rolled_back_from=target_report.version_id,
        )
        self._versions.append((new_report, restored))
        return new_report


__all__ = [
    "DatasetPublisher",
    "PublicationReport",
    "PublicationRollbackError",
    "QuarantinedRecord",
]
