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

Quarantine granularity: a prune -> revalidate fixpoint
------------------------------------------------------
Validation used to run exactly once. Every `sh:Violation`'s `focus_node`
named a record to quarantine, its triples were dropped, and whatever
remained was published unchecked. That left a real hole, disclosed at the
time as a bounded choice: a still-published record referencing a quarantined
subject (a valid WorkOrder whose `energy:concernsAsset` points at an Asset
that was just removed) was neither quarantined nor rewritten, so the
"published" graph could still violate the very shapes it was gated on.

It no longer can. Validation now iterates: validate, remove what the report
condemns, revalidate, and repeat until the graph conforms. Publication
asserts conformance as a post-condition, so a non-conformant graph cannot be
published at all.

Two removal granularities, because they answer different failures:

* When a result carries `sh:resultPath` *and* an IRI `sh:value` naming a
  triple that is actually present, that one triple is pruned and the subject
  stays published. This is the referential case -- `sh:class` on
  `energy:concernsAsset` -- and pruning the reference rather than the record
  matters: `SiteShape`'s `energy:hasAsset sh:class energy:Asset` means
  subject-granularity alone would quarantine the entire Site over one bad
  turbine, destroying the topology of nine healthy ones.
* Otherwise the subject is quarantined whole. A `sh:minCount` failure has a
  path but no value -- the absence *is* the violation -- and a bad
  `sh:datatype` prunes to a missing required property, which the next pass
  then quarantines. Both converge; neither guesses.

Termination: each pass removes at least one triple from a graph that only
ever shrinks, or aborts on "no progress". The quarantine set is monotone --
a subject is never un-quarantined -- which is what makes the bound real; the
violation count itself is not monotone, since removing a triple can satisfy
a `sh:maxCount`. `max_iterations` fails closed well before the theoretical
bound of one pass per triple.

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

from rdflib import Graph, URIRef

from graphrag.graph.shacl_validator import SHACLValidator, ShaclResult

# A sh:value naming an IRI is a reference this gate can prune on its own.
# A literal value cannot be reconstructed faithfully from SHACL's string form
# (its datatype and language tag are not carried on the result), so those
# fall through to subject quarantine rather than being guessed at.
_IRI_PREFIXES = ("http://", "https://", "urn:")


def _referential_triple(graph: Graph, result: ShaclResult) -> tuple | None:
    """The single triple this violation condemns, when it names one.

    Requires a focus node, a result path and an IRI value that together
    identify a triple actually present in `graph`. Returns None for every
    other shape of violation -- notably `sh:minCount`, where the missing
    value is the whole point and there is nothing to prune.
    """
    if not (result.focus_node and result.result_path and result.value):
        return None
    if not result.value.startswith(_IRI_PREFIXES):
        return None
    triple = (URIRef(result.focus_node), URIRef(result.result_path), URIRef(result.value))
    return triple if triple in graph else None


class PublicationRollbackError(RuntimeError):
    """Raised when `rollback()`'s target version does not exist -- including
    the case where only one version exists and no explicit target was given
    (there is nothing before the current version to roll back to)."""


class PublicationGateError(RuntimeError):
    """The fixpoint could not reach a conformant graph.

    Raised rather than publishing what it has: the entire purpose of this
    gate is that a published graph conforms, so a graph that will not
    converge must not be published at all.
    """


@dataclass(frozen=True)
class QuarantinedRecord:
    subject: str
    reasons: list[str] = field(default_factory=list)
    # 0 = intrinsically invalid on the first pass. >=1 = collateral, removed
    # only once an earlier pass took away something it depended on. Recorded
    # because "my WorkOrder was valid, why was it dropped?" is otherwise an
    # unanswerable question.
    iteration: int = 0


@dataclass(frozen=True)
class PrunedReference:
    """One referential triple removed so its subject could stay published."""

    subject: str
    predicate: str
    obj: str
    reason: str
    iteration: int


@dataclass(frozen=True)
class PublicationReport:
    version_id: str
    published_at: str
    published_triple_count: int
    candidate_record_count: int
    quarantined_records: list[QuarantinedRecord] = field(default_factory=list)
    rolled_back_from: str | None = None
    pruned_references: list[PrunedReference] = field(default_factory=list)
    revalidation_passes: int = 1
    # The asserted post-condition, persisted rather than assumed: this graph
    # was validated to conform after the last removal, not merely before.
    conforms: bool = True

    @property
    def quarantined_count(self) -> int:
        return len(self.quarantined_records)

    @property
    def pruned_count(self) -> int:
        return len(self.pruned_references)


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

    def stage_and_publish(self, candidate: Graph, *, max_passes: int = 10) -> PublicationReport:
        """Validate, remove what fails, revalidate, and publish what conforms.

        "Publish" means "publish whatever conforms" -- one bad record does
        not block the rest -- but unlike the single-pass version this
        replaced, what gets published is *verified* to conform rather than
        assumed to. See the module docstring for the two removal
        granularities and the termination argument.
        """
        candidate_subjects = {str(subject) for subject in candidate.subjects()}
        working = Graph()
        for prefix, namespace in candidate.namespaces():
            working.bind(prefix, namespace)
        for triple in candidate:
            working.add(triple)

        quarantined: dict[str, QuarantinedRecord] = {}
        pruned: list[PrunedReference] = []

        for iteration in range(max_passes):
            report_shacl = SHACLValidator(working, shapes_path=self._shapes_path).validate_report(target="energy")
            violations = [
                result for result in report_shacl.results
                if result.severity == "Violation" and result.focus_node
            ]
            if not violations:
                return self._publish(
                    working, candidate_subjects, quarantined, pruned, iteration + 1,
                )

            removed_any = False
            for result in violations:
                triple = _referential_triple(working, result)
                if triple is not None:
                    working.remove(triple)
                    pruned.append(PrunedReference(
                        subject=str(triple[0]), predicate=str(triple[1]), obj=str(triple[2]),
                        reason=result.message, iteration=iteration,
                    ))
                    removed_any = True
                    continue
                subject = URIRef(result.focus_node)
                existing = quarantined.get(result.focus_node)
                reasons = list(existing.reasons) if existing else []
                reasons.append(result.message)
                quarantined[result.focus_node] = QuarantinedRecord(
                    subject=result.focus_node, reasons=reasons,
                    iteration=existing.iteration if existing else iteration,
                )
                for owned in list(working.triples((subject, None, None))):
                    working.remove(owned)
                    removed_any = True

            if not removed_any:
                raise PublicationGateError(
                    f"SHACL reported {len(violations)} violation(s) that identify nothing "
                    "removable; refusing to publish a non-conformant graph"
                )

        raise PublicationGateError(
            f"validation did not reach a fixpoint within {max_passes} passes; "
            "refusing to publish a non-conformant graph"
        )

    def _publish(
        self, published: Graph, candidate_subjects: set[str],
        quarantined: dict[str, QuarantinedRecord], pruned: list[PrunedReference],
        passes: int,
    ) -> PublicationReport:
        report = PublicationReport(
            version_id=uuid4().hex,
            published_at=_now_iso(),
            published_triple_count=len(published),
            candidate_record_count=len(candidate_subjects),
            quarantined_records=[quarantined[key] for key in sorted(quarantined)],
            pruned_references=pruned,
            revalidation_passes=passes,
            conforms=True,
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
    "PrunedReference",
    "PublicationGateError",
    "PublicationReport",
    "PublicationRollbackError",
    "QuarantinedRecord",
]
