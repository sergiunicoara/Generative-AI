"""Query-derived Energy answers and citations.

Every supported business answer and every evidence field is derived here from
the published RDF graph, via version-controlled SPARQL under
``evals/energy_demo/sparql/``. Question *ids* are fixed (they are the demo's
reviewable question set); answers are not.

What this replaced, and why it mattered
---------------------------------------
``EnergyDemoService.answer()`` used to pick the authoritative bulletin with a
hard-coded date branch (``effective >= 2026-06-01``), interpolate two
hard-coded revision ids and thresholds, and then assert prose containing a
literal ``96°C`` and a literal ``WO-9001`` regardless of what its one SPARQL
query actually returned. Three of the five questions ran no query at all, and
every question returned the same three hand-written evidence rows with fixed
timestamps -- including for historical questions, where those timestamps were
*later* than the instant being asked about.

The practical consequence: change the source data and the prose kept its old
claims. That is the defect this module exists to remove. A regression suite
(tests/unit/test_energy_answer_derivation.py) mutates the affected turbine,
the temperature, the threshold, the work-order status, the bulletin revision
and evidence availability, and asserts the answers and citations move with
the data.

Bulletin selection here is valid-time only (``energy:validFrom`` /
``energy:validTo``). The recorded-time axis (``energy:recordedAt``) is carried
on every citation but is not yet used to filter, and the missing/conflicting/
expired/superseded outcomes are not yet distinguished -- both are the
temporal-correctness work, deliberately kept separate from this change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from rdflib import Graph, URIRef

from graphrag.domains.energy.vocabulary import ENERGY, PROV, format_instant, parse_instant
from graphrag.graph.sparql_bridge import SPARQLBridge

ROOT = Path(__file__).resolve().parents[3]
QUERY_DIR = ROOT / "evals" / "energy_demo" / "sparql"

QUESTIONS = {
    "maintenance_review": "Which assets need maintenance review, and why?",
    "open_work_orders": "Which open work orders concern components mentioned in the latest bulletin?",
    "revision_change": "What changed when the revised manufacturer bulletin became effective?",
    "historical_state": "What would the answer have been at a specified earlier date?",
    "insufficient_evidence": "Which assets cannot be assessed because required evidence is missing?",
}

ANSWER_SOURCE = "version-controlled SPARQL query"
QUERY_VERSION = "energy-demo/v2"
MAPPING_VERSION = "energy-r2rml/1.0.0"

# Substituted values are always drawn from the published graph or from an
# instant this module itself formatted -- never passed through from a caller
# verbatim. This guard keeps that true by construction rather than by
# convention, since the substitution lands inside a SPARQL string literal.
_SAFE_SUBSTITUTION = re.compile(r"^[A-Za-z0-9:_.+-]+$")


class EnergyAnswerError(ValueError):
    """A question id that this module has no derivation for."""


@dataclass(frozen=True)
class Evidence:
    """One cited source record. Every field is read from the published graph."""

    source_id: str
    source_type: str
    source_document: str
    field_or_span: str
    observed_at: str
    valid_from: str
    valid_to: str | None
    recorded_at: str
    access_scope: str
    value: str


@lru_cache(maxsize=None)
def _query_text(name: str) -> str:
    return (QUERY_DIR / name).read_text(encoding="utf-8")


def _render(name: str, substitutions: dict[str, str]) -> str:
    query = _query_text(name)
    for key, value in substitutions.items():
        if not _SAFE_SUBSTITUTION.match(value):
            raise EnergyAnswerError(f"unsafe value {value!r} for {key} in {name}")
        query = query.replace("{{" + key + "}}", value)
    return query


def _run(graph: Graph, name: str, substitutions: dict[str, str] | None = None) -> list[dict[str, Any]]:
    return SPARQLBridge(graph).query(_render(name, substitutions or {}))


def _local_name(iri: str) -> str:
    return iri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def _source_type(provenance_iri: str) -> str:
    """Derive a readable source label from the record's own provenance IRI.

    A mechanical transform of ``prov:wasDerivedFrom`` (e.g.
    ``urn:synthetic:sap:work-orders`` -> ``sap_work_orders``), not a lookup
    table of invented names -- so a new synthetic source shows up correctly
    without anything here being edited.
    """
    text = provenance_iri
    for prefix in ("urn:synthetic:", "urn:"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() or "unknown_source"


def _provenance(graph: Graph, subject_iri: str) -> str:
    value = graph.value(URIRef(subject_iri), PROV.wasDerivedFrom)
    return str(value) if value is not None else ""


def _measure(value: str, unit: str) -> str:
    return f"{value} {unit}".strip() if unit else value


def _instant(value: str) -> str:
    """Render an instant in the ``...Z`` form the source data uses.

    rdflib string-coerces an ``xsd:dateTime`` to its ``+00:00`` canonical
    form; the synthetic exports all write ``Z``. Normalising keeps one
    spelling across answers, citations and the fixtures they came from.
    Anything unparseable is passed through untouched rather than guessed at.
    """
    if not value:
        return value
    try:
        return format_instant(parse_instant(value))
    except ValueError:
        return value


def _metric_unit(graph: Graph, metric: str) -> str:
    """The unit recorded for a metric, when the graph agrees on exactly one.

    Used to render a bulletin's threshold in the same unit the readings it
    governs are expressed in, without inventing a unit the data never states.
    """
    units = {
        str(unit)
        for observation in graph.subjects(ENERGY.metric, None)
        if str(graph.value(observation, ENERGY.metric)) == metric
        for unit in [graph.value(observation, ENERGY.unit)]
        if unit is not None
    }
    return units.pop() if len(units) == 1 else ""


def _observation_evidence(graph: Graph, row: dict[str, Any], tenant: str) -> Evidence:
    provenance = _provenance(graph, row["observation"])
    return Evidence(
        source_id=_local_name(row["observation"]),
        source_type=_source_type(provenance),
        source_document=provenance,
        field_or_span=row["metric"],
        observed_at=_instant(row["observedAt"]),
        valid_from=_instant(row["observedAt"]),
        valid_to=None,
        recorded_at=_instant(row["observationRecordedAt"]),
        access_scope=tenant,
        value=_measure(row["temperature"], row.get("unit", "")),
    )


def _work_order_evidence(graph: Graph, row: dict[str, Any], tenant: str) -> Evidence:
    provenance = _provenance(graph, row["workOrder"])
    recorded_at = row.get("workOrderRecordedAt") or row.get("recordedAt", "")
    valid_from = row.get("workOrderValidFrom") or row.get("validFrom", "")
    return Evidence(
        source_id=row["workOrderId"],
        source_type=_source_type(provenance),
        source_document=provenance,
        field_or_span="status",
        observed_at=_instant(valid_from),
        valid_from=_instant(valid_from),
        valid_to=None,
        recorded_at=_instant(recorded_at),
        access_scope=tenant,
        value=row["status"],
    )


def _bulletin_evidence(graph: Graph, bulletin: dict[str, Any], tenant: str, unit: str) -> Evidence:
    provenance = _provenance(graph, bulletin["bulletin"])
    return Evidence(
        source_id=bulletin["bulletinId"],
        source_type=_source_type(provenance),
        source_document=provenance,
        field_or_span="temperatureReviewThreshold",
        observed_at=_instant(bulletin["validFrom"]),
        valid_from=_instant(bulletin["validFrom"]),
        valid_to=_instant(bulletin["validTo"]) if bulletin.get("validTo") else None,
        recorded_at=_instant(bulletin["recordedAt"]),
        access_scope=tenant,
        value=_measure(bulletin["threshold"], unit),
    )


@dataclass(frozen=True)
class BulletinSelection:
    """The outcome of resolving which guidance applies at a temporal view.

    `status` is one of:

    ``current``     exactly one revision applies; `bulletin` is it.
    ``missing``     no revision is in force at this instant, and none ever
                    was -- nothing to expire, nothing recorded yet.
    ``expired``     no revision is in force, but one was and its validity
                    ended before this instant. `candidates` carries it, so
                    the abstention can name what lapsed and when.
    ``conflicting`` two or more revisions claim the same instant and no
                    supersession edge orders them. Deliberately abstains
                    rather than picking one.

    A revision that is superseded by another *also in force at the same
    instant* is resolved rather than treated as a conflict -- the superseding
    revision wins, and `resolution` records that it happened so the
    overlap is visible instead of silent.
    """

    status: str
    bulletin: dict[str, Any] | None
    reason: str
    candidates: list[dict[str, Any]]
    resolution: str = ""

    @property
    def is_current(self) -> bool:
        return self.status == "current"


def select_bulletin(graph: Graph, *, as_of: datetime, known_as: datetime) -> BulletinSelection:
    """Resolve the applicable guidance revision at (`as_of`, `known_as`).

    Replaces a hard-coded ``effective >= 2026-06-01`` date branch. Candidates
    come from the graph filtered by recorded time; which of them is in force
    is decided here against their stored `validFrom`/`validTo` windows, and
    every non-resolvable case is reported explicitly rather than collapsing
    to "no guidance".
    """
    rows = _run(graph, "bulletin_candidates.rq", {"KNOWN_AS": format_instant(known_as)})

    applicable: list[dict[str, Any]] = []
    expired: list[dict[str, Any]] = []
    for row in rows:
        valid_from = parse_instant(row["validFrom"])
        valid_to = parse_instant(row["validTo"]) if row.get("validTo") else None
        if valid_from <= as_of and (valid_to is None or as_of < valid_to):
            applicable.append(row)
        elif valid_to is not None and valid_to <= as_of:
            expired.append(row)

    if not applicable:
        if expired:
            latest = max(expired, key=lambda row: parse_instant(row["validTo"]))
            return BulletinSelection(
                status="expired", bulletin=None, candidates=expired,
                reason=(
                    f"{latest['bulletinId']} was the last applicable "
                    f"{latest['componentType']} guidance and its validity ended "
                    f"{_instant(latest['validTo'])}, before "
                    f"{format_instant(as_of)}"
                ),
            )
        return BulletinSelection(
            status="missing", bulletin=None, candidates=[],
            reason=(
                f"no guidance revision recorded by {format_instant(known_as)} is in "
                f"force at {format_instant(as_of)}"
            ),
        )

    # A revision superseded by another that is ALSO in force here is ordered
    # by that edge; one left standing is the answer.
    in_force_ids = {row["bulletin"] for row in applicable}
    heads = [row for row in applicable if row.get("supersededBy") not in in_force_ids]
    resolution = ""
    if len(applicable) > 1 and len(heads) == 1:
        superseded = [row["bulletinId"] for row in applicable if row not in heads]
        resolution = (
            f"{heads[0]['bulletinId']} supersedes {', '.join(sorted(superseded))}, "
            "whose validity windows overlap it"
        )

    if len(heads) > 1:
        names = sorted(row["bulletinId"] for row in heads)
        return BulletinSelection(
            status="conflicting", bulletin=None, candidates=heads,
            reason=(
                f"{' and '.join(names)} both claim validity at "
                f"{format_instant(as_of)} and no supersession orders them"
            ),
        )

    return BulletinSelection(
        status="current", bulletin=heads[0], candidates=applicable,
        reason="", resolution=resolution,
    )


class TemporalInvariantError(RuntimeError):
    """A citation fell outside the temporal view it was produced for.

    Raised rather than silently dropped: a citation outside the asking view
    means a query lost one of its temporal filters, and quietly hiding the
    row would leave the answer subtly wrong with nothing to notice. Same
    fail-closed stance as the R2RML/LPG executors, which reject rather than
    approximate.
    """


def _assert_within_view(
    evidence: list[Evidence], *, as_of: datetime, known_as: datetime,
) -> None:
    """Every citation must be valid at `as_of` and recorded by `known_as`.

    The invariant gap C names: a historical answer must not cite a fact that
    was not yet knowable. Enforced here over the assembled citations, so it
    holds regardless of which query produced them.
    """
    for item in evidence:
        if item.valid_from and parse_instant(item.valid_from) > as_of:
            raise TemporalInvariantError(
                f"{item.source_id} is valid from {item.valid_from}, after as_of {format_instant(as_of)}"
            )
        if item.valid_to and parse_instant(item.valid_to) <= as_of:
            raise TemporalInvariantError(
                f"{item.source_id} stopped being valid at {item.valid_to}, at or before as_of {format_instant(as_of)}"
            )
        if item.recorded_at and parse_instant(item.recorded_at) > known_as:
            raise TemporalInvariantError(
                f"{item.source_id} was recorded {item.recorded_at}, after known_as {format_instant(known_as)}"
            )


def _result(
    *, answer: str, evidence: list[Evidence], effective: datetime, known_as: datetime,
    bulletin_id: str, rows: list[dict[str, Any]], status: str = "advisory",
    guidance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _assert_within_view(evidence, as_of=effective, known_as=known_as)
    return {
        "status": status,
        "answer": answer,
        "current_as_of": format_instant(effective),
        "known_as": format_instant(known_as),
        "authoritative_bulletin": bulletin_id,
        "evidence": [item.__dict__ for item in evidence],
        "query_rows": rows,
        "answer_source": ANSWER_SOURCE,
        "query_version": QUERY_VERSION,
        "mapping_version": MAPPING_VERSION,
        "guidance": guidance or {"status": "current", "reason": "", "resolution": ""},
    }


def _abstain_on_guidance(
    graph: Graph, selection: BulletinSelection, effective: datetime,
    known_as: datetime, tenant: str,
) -> dict[str, Any]:
    """Abstain, naming which guidance problem blocked the answer.

    `missing`, `expired` and `conflicting` produce visibly different
    explanations and different citations -- an expired revision is still
    cited so a reader can see what lapsed, and a conflict cites every
    revision that claims the instant.
    """
    evidence = [
        _bulletin_evidence(graph, row, tenant, _metric_unit(graph, "temperature_c"))
        for row in selection.candidates
    ]
    # Candidates are cited for explanation, not as support for a conclusion,
    # and an expired revision is by definition outside the asking view -- so
    # the within-view invariant deliberately does not apply to them.
    return {
        "status": "insufficient_evidence",
        "answer": (
            f"No maintenance conclusion is made: {selection.reason}."
        ),
        "current_as_of": format_instant(effective),
        "known_as": format_instant(known_as),
        "authoritative_bulletin": "",
        "evidence": [item.__dict__ for item in evidence],
        "query_rows": selection.candidates,
        "answer_source": ANSWER_SOURCE,
        "query_version": QUERY_VERSION,
        "mapping_version": MAPPING_VERSION,
        "guidance": {
            "status": selection.status,
            "reason": selection.reason,
            "resolution": selection.resolution,
        },
    }


def _maintenance_review(graph, bulletin, effective, known_as, tenant) -> dict[str, Any]:
    rows = _run(graph, "maintenance_review.rq", _view(bulletin, effective, known_as))
    unit = _metric_unit(graph, "temperature_c")
    if not rows:
        return _result(
            answer=(
                f"No asset exceeds the {_measure(bulletin['threshold'], unit)} "
                f"{bulletin['componentType']} review threshold in "
                f"{bulletin['bulletinId']} from the evidence available at "
                f"{format_instant(effective)}."
            ),
            evidence=[_bulletin_evidence(graph, bulletin, tenant, unit)],
            effective=effective, known_as=known_as,
            bulletin_id=bulletin["bulletinId"], rows=rows,
        )

    findings = [
        f"{_local_name(row['asset'])}"
        + (f" ({row['assetLabel']})" if row.get("assetLabel") else "")
        + f": {row['componentType']} {row['metric']} "
        + f"{_measure(row['temperature'], row.get('unit', ''))} observed {_instant(row['observedAt'])} "
        + f"exceeds the {_measure(row['threshold'], row.get('unit', ''))} threshold in "
        + f"{row['bulletinId']}; work order {row['workOrderId']} is {row['status']}"
        for row in rows
    ]
    evidence: list[Evidence] = []
    for row in rows:
        evidence.append(_observation_evidence(graph, row, tenant))
        evidence.append(_work_order_evidence(graph, row, tenant))
    evidence.append(_bulletin_evidence(graph, bulletin, tenant, unit))

    subject = "Advisory review is required for"
    return _result(
        answer=f"{subject} {len(rows)} asset{'s' if len(rows) != 1 else ''}. " + ". ".join(findings) + ".",
        evidence=evidence, effective=effective, known_as=known_as,
        bulletin_id=bulletin["bulletinId"], rows=rows,
    )


def _open_work_orders(graph, bulletin, effective, known_as, tenant) -> dict[str, Any]:
    rows = _run(graph, "open_work_orders.rq", _view(bulletin, effective, known_as))
    unit = _metric_unit(graph, "temperature_c")
    if not rows:
        return _result(
            answer=(
                f"No open work order concerns a {bulletin['componentType']} component "
                f"covered by {bulletin['bulletinId']} at {format_instant(effective)}."
            ),
            evidence=[_bulletin_evidence(graph, bulletin, tenant, unit)],
            effective=effective, known_as=known_as,
            bulletin_id=bulletin["bulletinId"], rows=rows,
        )

    described = [
        f"{row['workOrderId']} ({_local_name(row['asset'])} {row['componentType']})"
        for row in rows
    ]
    evidence = [_work_order_evidence(graph, row, tenant) for row in rows]
    evidence.append(_bulletin_evidence(graph, bulletin, tenant, unit))
    return _result(
        answer=(
            f"{', '.join(described)} {'is' if len(rows) == 1 else 'are'} open and "
            f"concern{'s' if len(rows) == 1 else ''} components covered by "
            f"{bulletin['bulletinId']}."
        ),
        evidence=evidence, effective=effective, known_as=known_as,
        bulletin_id=bulletin["bulletinId"], rows=rows,
    )


def _revision_change(graph, bulletin, effective, known_as, tenant) -> dict[str, Any]:
    rows = _run(graph, "revision_change.rq", _view(bulletin, effective, known_as))
    unit = _metric_unit(graph, "temperature_c")
    bulletin_evidence = _bulletin_evidence(graph, bulletin, tenant, unit)
    if not rows:
        return _result(
            answer=(
                f"{bulletin['bulletinId']} is the earliest {bulletin['componentType']} "
                f"guidance revision known at {format_instant(known_as)}; it supersedes no "
                "earlier revision, so nothing changed at its effective date."
            ),
            evidence=[bulletin_evidence], effective=effective, known_as=known_as,
            bulletin_id=bulletin["bulletinId"], rows=rows,
        )

    row = rows[0]
    return _result(
        answer=(
            f"{row['bulletinId']} superseded {row['supersededId']} effective "
            f"{_instant(row['validFrom'])} and changed the {row['componentType']} temperature "
            f"review threshold from {_measure(row['supersededThreshold'], unit)} to "
            f"{_measure(row['threshold'], unit)}."
        ),
        evidence=[bulletin_evidence], effective=effective, known_as=known_as,
        bulletin_id=bulletin["bulletinId"], rows=rows,
    )


def _historical_state(graph, bulletin, effective, known_as, tenant) -> dict[str, Any]:
    unit = _metric_unit(graph, "temperature_c")
    window = (
        f"in force from {_instant(bulletin['validFrom'])}"
        + (f" until {_instant(bulletin['validTo'])}" if bulletin.get("validTo") else " with no end date")
    )
    knowledge = (
        "" if known_as == effective
        else f", using only what was recorded by {format_instant(known_as)}"
    )
    return _result(
        answer=(
            f"As of {format_instant(effective)}{knowledge}, {bulletin['bulletinId']} was "
            f"authoritative ({window}) and the {bulletin['componentType']} temperature "
            f"review threshold was {_measure(bulletin['threshold'], unit)}."
        ),
        evidence=[_bulletin_evidence(graph, bulletin, tenant, unit)],
        effective=effective, known_as=known_as,
        bulletin_id=bulletin["bulletinId"], rows=[bulletin],
    )


def _insufficient_evidence(graph, bulletin, effective, known_as, tenant) -> dict[str, Any]:
    rows = _run(graph, "insufficient_evidence.rq", _view(bulletin, effective, known_as))
    unit = _metric_unit(graph, "temperature_c")
    if not rows:
        return _result(
            answer=(
                "Every asset has both a gearbox temperature reading and a work-order "
                "status at this instant, so none is blocked on missing evidence."
            ),
            evidence=[_bulletin_evidence(graph, bulletin, tenant, unit)],
            effective=effective, known_as=known_as,
            bulletin_id=bulletin["bulletinId"], rows=rows,
            status="advisory",
        )

    def _missing(row: dict[str, Any]) -> str:
        gaps = []
        if row.get("hasTemperature", "").lower() != "true":
            gaps.append("no temperature_c observation")
        if row.get("hasWorkOrder", "").lower() != "true":
            gaps.append("no work-order status")
        return " and ".join(gaps)

    described = [f"{_local_name(row['asset'])} ({_missing(row)})" for row in rows]
    return _result(
        answer=(
            f"{len(rows)} asset{'s' if len(rows) != 1 else ''} cannot be assessed against "
            f"{bulletin['bulletinId']} because required evidence is missing at "
            f"{format_instant(effective)}: {'; '.join(described)}. No maintenance "
            f"conclusion is made for {'them' if len(rows) != 1 else 'it'}."
        ),
        evidence=[_bulletin_evidence(graph, bulletin, tenant, unit)],
        effective=effective, known_as=known_as,
        bulletin_id=bulletin["bulletinId"], rows=rows,
        status="insufficient_evidence",
    )


_DERIVATIONS = {
    "maintenance_review": _maintenance_review,
    "open_work_orders": _open_work_orders,
    "revision_change": _revision_change,
    "historical_state": _historical_state,
    "insufficient_evidence": _insufficient_evidence,
}


def _view(bulletin: dict[str, Any], as_of: datetime, known_as: datetime) -> dict[str, str]:
    """The substitutions every question's query needs: bulletin + both axes."""
    return {
        "BULLETIN_ID": bulletin["bulletinId"],
        "AS_OF": format_instant(as_of),
        "KNOWN_AS": format_instant(known_as),
    }


def resolve_view(
    graph: Graph, *, as_of: str | None = None, known_as: str | None = None,
    now: str | None = None,
) -> tuple[datetime, datetime]:
    """Resolve the (valid-time, recorded-time) pair a question is asked at.

    The defaulting rule is the whole point of gap C:

    * neither given -> both are "now": today's answer from everything known.
    * `as_of` alone -> `known_as` defaults to **`as_of`**, i.e. "as we knew
      it then". This is what makes a historical answer structurally unable
      to cite a fact recorded later -- the defect being fixed, where a
      question about May returned evidence timestamped in August.
    * `known_as` alone -> ask about now through older knowledge.
    * both given -> the fully general bitemporal query.
    """
    default = _default_instant(graph, now)
    if as_of is not None and known_as is None:
        effective = parse_instant(as_of)
        return effective, effective
    return (
        parse_instant(as_of) if as_of else default,
        parse_instant(known_as) if known_as else default,
    )


def answer(
    graph: Graph, question_id: str, *, tenant: str, dataset_tenant: str,
    as_of: str | None = None, known_as: str | None = None, now: str | None = None,
) -> dict[str, Any]:
    """Derive the answer and citations for `question_id` from `graph`.

    `tenant` is the caller's tenant; `dataset_tenant` is the published
    dataset's own. A mismatch means this dataset is not the caller's to read,
    which is reported as not-found with no evidence rather than as a denial
    that would itself confirm the dataset exists.

    `as_of` is valid time (when the fact applies) and `known_as` is recorded
    time (when the platform knew it); see `resolve_view` for how they default.
    """
    if tenant != dataset_tenant:
        return {
            "status": "not_found",
            "answer": "No energy demonstration is available for this tenant.",
            "evidence": [],
        }
    if question_id not in QUESTIONS:
        raise EnergyAnswerError("unknown energy demonstration question")

    effective, knowledge = resolve_view(graph, as_of=as_of, known_as=known_as, now=now)
    selection = select_bulletin(graph, as_of=effective, known_as=knowledge)
    if not selection.is_current:
        return _abstain_on_guidance(graph, selection, effective, knowledge, tenant)
    result = _DERIVATIONS[question_id](
        graph, selection.bulletin, effective, knowledge, tenant,
    )
    if selection.resolution:
        result["guidance"] = {
            "status": "current", "reason": "", "resolution": selection.resolution,
        }
    return result


def _default_instant(graph: Graph, now: str | None) -> datetime:
    """The instant an un-dated question is answered at.

    Derived from the data rather than a hard-coded "today": the latest
    recorded-time in the published graph. That keeps the demo reproducible
    (its answers do not drift with the wall clock) without pinning a literal
    date in code that silently goes stale as the fixtures move.
    """
    if now is not None:
        return parse_instant(now)
    recorded = [str(value) for value in graph.objects(None, ENERGY.recordedAt)]
    if not recorded:
        raise EnergyAnswerError("published graph carries no energy:recordedAt to anchor 'now'")
    return max(parse_instant(value) for value in recorded)


__all__ = [
    "ANSWER_SOURCE", "BulletinSelection", "EnergyAnswerError", "Evidence",
    "MAPPING_VERSION", "QUERY_VERSION", "QUESTIONS", "TemporalInvariantError",
    "answer", "render_query", "resolve_view", "select_bulletin",
]


def render_query(name: str, **substitutions: str) -> str:
    """Render a committed query exactly as production renders it.

    Exported so tests that execute these files against a live triplestore
    (tests/e2e/test_live_graphdb.py) run the same text this module does,
    instead of maintaining their own substitution logic that can drift.
    """
    return _render(name, substitutions)
