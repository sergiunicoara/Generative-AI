"""Bitemporal regression: valid time and recorded time are separate axes.

Gap C named two defects. First, bulletin selection was a hard-coded date
branch (``effective >= 2026-06-01``) rather than a query over the
``energy:validFrom``/``energy:validTo`` the revisions already carried.
Second -- and worse -- a historical answer returned evidence timestamped
*after* the instant it was asked about, because the citation list was a fixed
hand-written trio with fixed timestamps.

The fix gives every question two explicit axes: `as_of` (when the fact
applies) and `known_as` (when the platform knew it). The load-bearing
default is that `as_of` alone implies `known_as = as_of` -- "as we knew it
then" -- which makes citing a later-recorded fact structurally impossible
rather than merely unlikely.

The fixture carries a real late-arriving correction for this: WT-01's
2026-08-28T08:00Z gearbox reading was first recorded that same instant as
96.0 C, then corrected to 91.5 C in a record not written until
2026-09-02T10:00Z.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from graphrag.domains.energy.answers import (
    TemporalInvariantError,
    _assert_within_view,
    select_bulletin,
)
from graphrag.domains.energy.demo import EnergyDemoService, TENANT
from graphrag.domains.energy.fixtures import create_sap_fixture_sqlite
from graphrag.domains.energy.vocabulary import DOC, ENERGY, PROV, parse_instant

ROOT = Path(__file__).resolve().parents[2]

# The correction's own coordinates, as committed in
# data/energy_demo/snowflake_observations.json.
READING_INSTANT = "2026-08-28T08:00:00Z"
BEFORE_CORRECTION = "2026-08-28T12:00:00Z"
AFTER_CORRECTION = "2026-09-05T00:00:00Z"
ORIGINAL_ID, ORIGINAL_VALUE = "SNOW-OBS-WT-01", "96.0"
CORRECTED_ID, CORRECTED_VALUE = "SNOW-OBS-WT-01-R2", "91.5"


@pytest.fixture
def source(tmp_path) -> Path:
    database = tmp_path / "energy.sqlite"
    create_sap_fixture_sqlite(database)
    return database


@pytest.fixture
def service(source) -> EnergyDemoService:
    return EnergyDemoService(source_db=source)


def _ids(result: dict) -> set[str]:
    return {item["source_id"] for item in result["evidence"]}


class TestLateArrivingCorrection:
    def test_a_view_before_the_correction_sees_only_the_original_reading(self, service):
        result = service.answer(
            "maintenance_review", tenant=TENANT,
            as_of=BEFORE_CORRECTION, known_as=BEFORE_CORRECTION,
        )

        assert ORIGINAL_VALUE in result["answer"]
        assert CORRECTED_VALUE not in result["answer"]
        assert ORIGINAL_ID in _ids(result)
        assert CORRECTED_ID not in _ids(result)

    def test_the_same_instant_seen_later_returns_the_corrected_reading(self, service):
        """Identical valid time, later recorded time -- a different answer.

        This is the bitemporal distinction itself: nothing about the physical
        moment being asked about changed, only what the platform knows about
        it, and the answer moves accordingly.
        """
        result = service.answer(
            "maintenance_review", tenant=TENANT,
            as_of=BEFORE_CORRECTION, known_as=AFTER_CORRECTION,
        )

        assert CORRECTED_VALUE in result["answer"]
        assert ORIGINAL_VALUE not in result["answer"]
        assert CORRECTED_ID in _ids(result)
        assert ORIGINAL_ID not in _ids(result)

    def test_as_of_alone_defaults_to_what_was_known_then(self, service):
        """The defect gap C reported: a historical question must not return
        evidence recorded after the instant it asks about. Supplying `as_of`
        without `known_as` defaults the knowledge axis to the same instant,
        so the later correction is invisible by construction."""
        result = service.answer("maintenance_review", tenant=TENANT, as_of=BEFORE_CORRECTION)

        assert result["known_as"] == BEFORE_CORRECTION
        assert ORIGINAL_ID in _ids(result)
        assert CORRECTED_ID not in _ids(result)

    def test_no_citation_is_ever_dated_after_the_view_that_produced_it(self, service):
        """The general invariant, over every question and several views --
        not just the one case the correction demonstrates."""
        for as_of in ("2026-06-15T00:00:00Z", BEFORE_CORRECTION, AFTER_CORRECTION):
            for question_id in service.questions:
                result = service.answer(question_id, tenant=TENANT, as_of=as_of)
                view = parse_instant(result["known_as"])
                effective = parse_instant(result["current_as_of"])
                for item in result["evidence"]:
                    assert parse_instant(item["recorded_at"]) <= view, (
                        f"{question_id} at {as_of} cited {item['source_id']} "
                        f"recorded {item['recorded_at']}"
                    )
                    assert parse_instant(item["valid_from"]) <= effective

    def test_the_correction_does_not_silently_change_the_conclusion(self, service):
        """Both readings sit above the 85.0 threshold, so the correction
        changes the cited value without flipping the operational outcome --
        stated explicitly so a reader is not left assuming a corrected
        reading always means a withdrawn advisory."""
        before = service.answer("maintenance_review", tenant=TENANT, as_of=BEFORE_CORRECTION)
        after = service.answer(
            "maintenance_review", tenant=TENANT,
            as_of=BEFORE_CORRECTION, known_as=AFTER_CORRECTION,
        )

        assert before["status"] == after["status"] == "advisory"
        assert "WT-01" in before["answer"] and "WT-01" in after["answer"]


class TestTemporalInvariantIsEnforcedNotAssumed:
    def test_a_citation_recorded_after_the_view_raises(self):
        from graphrag.domains.energy.answers import Evidence

        late = Evidence(
            source_id="LATE-1", source_type="t", source_document="urn:x",
            field_or_span="f", observed_at="2026-01-01T00:00:00Z",
            valid_from="2026-01-01T00:00:00Z", valid_to=None,
            recorded_at="2026-12-31T00:00:00Z", access_scope=TENANT, value="v",
        )

        with pytest.raises(TemporalInvariantError, match="recorded"):
            _assert_within_view(
                [late],
                as_of=parse_instant("2026-06-01T00:00:00Z"),
                known_as=parse_instant("2026-06-01T00:00:00Z"),
            )

    def test_a_citation_not_yet_valid_at_the_instant_raises(self):
        from graphrag.domains.energy.answers import Evidence

        future = Evidence(
            source_id="FUTURE-1", source_type="t", source_document="urn:x",
            field_or_span="f", observed_at="2026-12-01T00:00:00Z",
            valid_from="2026-12-01T00:00:00Z", valid_to=None,
            recorded_at="2026-01-01T00:00:00Z", access_scope=TENANT, value="v",
        )

        with pytest.raises(TemporalInvariantError, match="valid from"):
            _assert_within_view(
                [future],
                as_of=parse_instant("2026-06-01T00:00:00Z"),
                known_as=parse_instant("2026-06-01T00:00:00Z"),
            )


def _bulletin_graph(revisions: list[dict]) -> Graph:
    """A minimal graph of guidance revisions, for the selection edge cases.

    Built directly rather than through the fixture pipeline because these
    cases -- two revisions claiming one instant, everything expired, nothing
    recorded -- are data states the shipped synthetic export deliberately
    does not contain.
    """
    graph = Graph()
    for revision in revisions:
        node = DOC[revision["id"]]
        graph.add((node, RDF.type, ENERGY.DocumentRevision))
        graph.add((node, ENERGY.documentId, Literal(revision["id"])))
        graph.add((node, ENERGY.appliesToComponentType, Literal("gearbox")))
        graph.add((node, ENERGY.temperatureReviewThreshold,
                   Literal(str(revision["threshold"]), datatype=XSD.decimal)))
        graph.add((node, ENERGY.validFrom, Literal(revision["valid_from"], datatype=XSD.dateTime)))
        if revision.get("valid_to"):
            graph.add((node, ENERGY.validTo, Literal(revision["valid_to"], datatype=XSD.dateTime)))
        graph.add((node, ENERGY.recordedAt, Literal(revision["recorded_at"], datatype=XSD.dateTime)))
        if revision.get("supersedes"):
            graph.add((node, ENERGY.supersedes, DOC[revision["supersedes"]]))
        graph.add((node, PROV.wasDerivedFrom, URIRef("urn:synthetic:sharepoint:technical-guidance")))
    return graph


class TestGuidanceOutcomesAreExplicit:
    """Missing, expired, conflicting and superseded guidance are each
    reported as themselves rather than collapsing into one silent
    "no answer"."""

    def test_nothing_recorded_yet_is_missing(self):
        graph = _bulletin_graph([
            {"id": "R1", "threshold": 90.0, "valid_from": "2026-01-01T00:00:00Z",
             "recorded_at": "2026-01-01T00:00:00Z"},
        ])

        selection = select_bulletin(
            graph, as_of=parse_instant("2025-06-01T00:00:00Z"),
            known_as=parse_instant("2025-06-01T00:00:00Z"),
        )

        assert selection.status == "missing"
        assert selection.bulletin is None
        assert "no guidance revision recorded" in selection.reason

    def test_guidance_whose_window_has_closed_is_expired_not_missing(self):
        graph = _bulletin_graph([
            {"id": "R1", "threshold": 90.0, "valid_from": "2026-01-01T00:00:00Z",
             "valid_to": "2026-06-01T00:00:00Z", "recorded_at": "2026-01-01T00:00:00Z"},
        ])

        selection = select_bulletin(
            graph, as_of=parse_instant("2026-08-01T00:00:00Z"),
            known_as=parse_instant("2026-08-01T00:00:00Z"),
        )

        assert selection.status == "expired"
        assert selection.bulletin is None
        assert "R1" in selection.reason and "2026-06-01" in selection.reason
        # The lapsed revision is still cited, so a reader can see what ended.
        assert [row["bulletinId"] for row in selection.candidates] == ["R1"]

    def test_two_unordered_revisions_claiming_one_instant_conflict(self):
        graph = _bulletin_graph([
            {"id": "R1", "threshold": 90.0, "valid_from": "2026-01-01T00:00:00Z",
             "recorded_at": "2026-01-01T00:00:00Z"},
            {"id": "RX", "threshold": 80.0, "valid_from": "2026-02-01T00:00:00Z",
             "recorded_at": "2026-02-01T00:00:00Z"},
        ])

        selection = select_bulletin(
            graph, as_of=parse_instant("2026-08-01T00:00:00Z"),
            known_as=parse_instant("2026-08-01T00:00:00Z"),
        )

        assert selection.status == "conflicting"
        assert selection.bulletin is None
        assert "R1" in selection.reason and "RX" in selection.reason
        assert {row["bulletinId"] for row in selection.candidates} == {"R1", "RX"}

    def test_an_overlapping_supersession_resolves_and_says_so(self):
        """Same overlap, but with a supersedes edge ordering the two. That
        is resolvable, so it resolves -- and records that it did, rather
        than silently picking one."""
        graph = _bulletin_graph([
            {"id": "R1", "threshold": 90.0, "valid_from": "2026-01-01T00:00:00Z",
             "recorded_at": "2026-01-01T00:00:00Z"},
            {"id": "R2", "threshold": 85.0, "valid_from": "2026-02-01T00:00:00Z",
             "recorded_at": "2026-02-01T00:00:00Z", "supersedes": "R1"},
        ])

        selection = select_bulletin(
            graph, as_of=parse_instant("2026-08-01T00:00:00Z"),
            known_as=parse_instant("2026-08-01T00:00:00Z"),
        )

        assert selection.status == "current"
        assert selection.bulletin["bulletinId"] == "R2"
        assert "R2 supersedes R1" in selection.resolution

    def test_a_revision_not_yet_recorded_is_invisible_to_an_earlier_view(self):
        """Recorded-time filtering applies to guidance too, not only
        telemetry: a revision published later cannot inform an earlier
        view, even about an instant its validity window covers."""
        graph = _bulletin_graph([
            {"id": "R1", "threshold": 90.0, "valid_from": "2026-01-01T00:00:00Z",
             "valid_to": "2026-06-01T00:00:00Z", "recorded_at": "2026-01-01T00:00:00Z"},
            {"id": "R2", "threshold": 85.0, "valid_from": "2026-06-01T00:00:00Z",
             "recorded_at": "2026-09-01T00:00:00Z", "supersedes": "R1"},
        ])
        instant = parse_instant("2026-07-01T00:00:00Z")

        early = select_bulletin(graph, as_of=instant, known_as=parse_instant("2026-07-15T00:00:00Z"))
        later = select_bulletin(graph, as_of=instant, known_as=parse_instant("2026-09-15T00:00:00Z"))

        # R2 governs 2026-07-01 but was not recorded until September.
        assert early.status == "expired"
        assert later.status == "current" and later.bulletin["bulletinId"] == "R2"


class TestAbstentionSurfacesTheGuidanceProblem:
    def test_an_instant_with_no_applicable_guidance_abstains_and_explains(self, service):
        result = service.answer("maintenance_review", tenant=TENANT, as_of="2025-01-01T00:00:00Z")

        assert result["status"] == "insufficient_evidence"
        assert result["guidance"]["status"] == "missing"
        assert result["authoritative_bulletin"] == ""
        assert "No maintenance conclusion is made" in result["answer"]

    def test_every_question_abstains_the_same_way_when_guidance_is_unavailable(self, service):
        for question_id in service.questions:
            result = service.answer(question_id, tenant=TENANT, as_of="2025-01-01T00:00:00Z")
            assert result["status"] == "insufficient_evidence", question_id
            assert result["guidance"]["status"] == "missing", question_id
