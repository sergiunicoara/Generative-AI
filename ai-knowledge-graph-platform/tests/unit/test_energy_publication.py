"""graphrag/domains/energy/publication.py's DatasetPublisher: the real
stage -> validate -> quarantine -> publish -> rollback lifecycle for an RDF
dataset, built against the Energy domain's real SHACL shapes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

from graphrag.domains.energy.publication import (
    DatasetPublisher,
    PublicationRollbackError,
)
from graphrag.graph.shacl_validator import SHACLValidator

ROOT = Path(__file__).resolve().parents[2]
SHAPES = ROOT / "ontology" / "shapes" / "energy-asset-intelligence.shapes.ttl"

ENERGY = Namespace("https://example.energy.demo/ontology#")
ASSET = Namespace("https://example.energy.demo/asset/")
REC = Namespace("https://example.energy.demo/record/")


def _valid_candidate() -> Graph:
    g = Graph()
    g.add((ASSET["WT-01"], RDF.type, ENERGY.Asset))
    g.add((ASSET["WT-01"], ENERGY.assetId, Literal("WT-01")))
    g.add((REC["obs-1"], RDF.type, ENERGY.Observation))
    # str() first, not a raw float -- Literal(96.0, datatype=XSD.decimal)
    # is ill-typed under strict SHACL sh:datatype checking (.toPython() is
    # a float, not a Decimal); see rml_rdf.py's identical fix.
    g.add((REC["obs-1"], ENERGY.value, Literal(str(96.0), datatype=XSD.decimal)))
    g.add((REC["obs-1"], ENERGY.unit, Literal("C")))
    g.add((REC["obs-1"], ENERGY.observedAt, Literal("2026-08-28T08:00:00Z", datatype=XSD.dateTime)))
    # energy:recordedAt is required on an Observation (the bitemporal
    # recorded-time axis added alongside the valid-time observedAt): without
    # it this "valid" candidate would itself be quarantined, and these tests
    # would be asserting against the wrong baseline.
    g.add((REC["obs-1"], ENERGY.recordedAt, Literal("2026-08-28T08:00:00Z", datatype=XSD.dateTime)))
    return g


def _candidate_with_one_invalid_observation() -> Graph:
    """One valid asset, one valid observation, one observation missing the
    required energy:value -- exactly the shape validate_candidate() already
    demonstrates elsewhere, reused here as a real staged candidate."""
    g = _valid_candidate()
    g.add((REC["obs-invalid"], RDF.type, ENERGY.Observation))
    g.add((REC["obs-invalid"], ENERGY.observedAsset, ASSET["WT-01"]))
    g.add((REC["obs-invalid"], ENERGY.observedAt, Literal("2026-08-28T08:00:00Z", datatype=XSD.dateTime)))
    g.add((REC["obs-invalid"], ENERGY.recordedAt, Literal("2026-08-28T08:00:00Z", datatype=XSD.dateTime)))
    # Deliberately missing energy:value and energy:unit -- and only those, so
    # this record is invalid for exactly the reason the test names.
    return g


def _candidate_with_a_dangling_reference() -> Graph:
    """A perfectly valid WorkOrder pointing at an Asset that is not.

    The Asset is missing its required energy:assetId, so it is quarantined.
    The WorkOrder itself violates nothing -- until the Asset's rdf:type goes
    with it, at which point the WorkOrder's own
    `energy:concernsAsset sh:class energy:Asset` can no longer be satisfied.
    A single validation pass never sees that second failure.
    """
    g = _valid_candidate()
    g.add((ASSET["WT-99"], RDF.type, ENERGY.Asset))
    # Deliberately no energy:assetId -- this is what makes WT-99 invalid.
    g.add((REC["WO-DANGLING"], RDF.type, ENERGY.WorkOrder))
    g.add((REC["WO-DANGLING"], ENERGY.workOrderId, Literal("WO-DANGLING")))
    g.add((REC["WO-DANGLING"], ENERGY.status, Literal("open")))
    g.add((REC["WO-DANGLING"], ENERGY.validFrom, Literal("2026-08-01T00:00:00Z", datatype=XSD.dateTime)))
    g.add((REC["WO-DANGLING"], ENERGY.recordedAt, Literal("2026-08-01T00:00:00Z", datatype=XSD.dateTime)))
    g.add((REC["WO-DANGLING"], ENERGY.concernsAsset, ASSET["WT-99"]))
    return g


class TestRevalidationFixpointClosesTheDanglingReferenceHole:
    """The executable refutation of this module's former disclosed
    limitation: quarantining a subject used to leave records referencing it
    published and non-conformant. These fail against the single-pass
    implementation this replaced."""

    def test_the_published_graph_actually_conforms_after_quarantine(self):
        publisher = DatasetPublisher(SHAPES)

        report = publisher.stage_and_publish(_candidate_with_a_dangling_reference())

        recheck = SHACLValidator(publisher.current, shapes_path=SHAPES).validate_report(target="energy")
        assert recheck.conforms, [r.message for r in recheck.results]
        assert report.conforms is True

    def test_the_dangling_reference_is_pruned_and_its_record_survives(self):
        publisher = DatasetPublisher(SHAPES)

        report = publisher.stage_and_publish(_candidate_with_a_dangling_reference())

        # The invalid asset is quarantined on the first pass...
        assert {r.subject for r in report.quarantined_records} == {str(ASSET["WT-99"])}
        # ...and the reference to it is pruned on a later one, rather than
        # the whole (otherwise valid) work order being discarded.
        assert report.revalidation_passes >= 2
        pruned = [
            reference for reference in report.pruned_references
            if reference.predicate == str(ENERGY.concernsAsset)
        ]
        assert len(pruned) == 1
        assert pruned[0].subject == str(REC["WO-DANGLING"])
        assert pruned[0].iteration >= 1

        assert (REC["WO-DANGLING"], RDF.type, ENERGY.WorkOrder) in publisher.current
        assert (REC["WO-DANGLING"], ENERGY.concernsAsset, ASSET["WT-99"]) not in publisher.current

    def test_a_clean_candidate_still_converges_in_one_pass(self):
        publisher = DatasetPublisher(SHAPES)

        report = publisher.stage_and_publish(_valid_candidate())

        assert report.revalidation_passes == 1
        assert report.quarantined_records == []
        assert report.pruned_references == []


class TestStageAndPublish:
    def test_an_all_valid_candidate_publishes_everything_with_no_quarantine(self):
        publisher = DatasetPublisher(SHAPES)
        candidate = _valid_candidate()

        report = publisher.stage_and_publish(candidate)

        assert report.quarantined_records == []
        assert report.published_triple_count == len(candidate)
        assert set(publisher.current) == set(candidate)

    def test_a_candidate_with_one_bad_record_quarantines_only_that_record(self):
        publisher = DatasetPublisher(SHAPES)
        candidate = _candidate_with_one_invalid_observation()

        report = publisher.stage_and_publish(candidate)

        quarantined_subjects = {r.subject for r in report.quarantined_records}
        assert quarantined_subjects == {str(REC["obs-invalid"])}
        assert report.quarantined_records[0].reasons  # real SHACL messages, not empty

        # The bad record's own triples are gone from the published graph...
        assert (REC["obs-invalid"], RDF.type, ENERGY.Observation) not in publisher.current
        # ...but the good asset and good observation are still published.
        assert (ASSET["WT-01"], ENERGY.assetId, Literal("WT-01")) in publisher.current
        assert (REC["obs-1"], ENERGY.value, Literal(96.0, datatype=XSD.decimal)) in publisher.current

    def test_published_triple_count_excludes_the_quarantined_records_triples(self):
        publisher = DatasetPublisher(SHAPES)
        candidate = _candidate_with_one_invalid_observation()
        report = publisher.stage_and_publish(candidate)
        assert report.published_triple_count == len(publisher.current)
        assert report.published_triple_count < len(candidate)

    def test_a_subject_no_shape_targets_at_all_is_still_published(self):
        """A Site/Component-shaped subject has no SHACL shape targeting it
        in this domain -- it must publish untouched, not get swept up as
        quarantined just because it wasn't explicitly validated."""
        publisher = DatasetPublisher(SHAPES)
        candidate = _valid_candidate()
        site = URIRef("https://example.energy.demo/asset/north-sea-wind-farm")
        candidate.add((site, RDF.type, ENERGY.Site))
        report = publisher.stage_and_publish(candidate)
        assert report.quarantined_records == []
        assert (site, RDF.type, ENERGY.Site) in publisher.current


class TestHistory:
    def test_history_grows_across_multiple_publications(self):
        publisher = DatasetPublisher(SHAPES)
        publisher.stage_and_publish(_valid_candidate())
        publisher.stage_and_publish(_candidate_with_one_invalid_observation())
        assert len(publisher.history()) == 2
        assert publisher.history()[-1] is publisher.current_report

    def test_current_report_before_any_publish_raises(self):
        publisher = DatasetPublisher(SHAPES)
        with pytest.raises(PublicationRollbackError):
            _ = publisher.current_report


class TestRollback:
    def test_implicit_rollback_restores_the_immediately_prior_version(self):
        publisher = DatasetPublisher(SHAPES)
        v1 = publisher.stage_and_publish(_valid_candidate())
        publisher.stage_and_publish(_candidate_with_one_invalid_observation())
        assert publisher.current_report.quarantined_records  # v2 has one

        rolled_back = publisher.rollback()

        assert rolled_back.rolled_back_from == v1.version_id
        assert rolled_back.quarantined_records == []  # v1 had none
        assert set(publisher.current) == set(_valid_candidate())
        # Rollback APPENDS a new version; it does not delete v2's history entry.
        assert len(publisher.history()) == 3

    def test_explicit_version_id_rollback(self):
        publisher = DatasetPublisher(SHAPES)
        v1 = publisher.stage_and_publish(_valid_candidate())
        publisher.stage_and_publish(_candidate_with_one_invalid_observation())
        publisher.stage_and_publish(_valid_candidate())

        rolled_back = publisher.rollback(version_id=v1.version_id)

        assert rolled_back.rolled_back_from == v1.version_id

    def test_rollback_with_only_one_version_and_no_explicit_target_raises(self):
        publisher = DatasetPublisher(SHAPES)
        publisher.stage_and_publish(_valid_candidate())
        with pytest.raises(PublicationRollbackError, match="nothing to roll back"):
            publisher.rollback()

    def test_rollback_to_an_unknown_version_id_raises(self):
        publisher = DatasetPublisher(SHAPES)
        publisher.stage_and_publish(_valid_candidate())
        publisher.stage_and_publish(_valid_candidate())
        with pytest.raises(PublicationRollbackError, match="no published version"):
            publisher.rollback(version_id="not-a-real-version-id")

    def test_rollback_before_any_publish_raises(self):
        publisher = DatasetPublisher(SHAPES)
        with pytest.raises(PublicationRollbackError):
            publisher.rollback()
