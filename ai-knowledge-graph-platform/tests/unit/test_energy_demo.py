from rdflib import Graph
from rdflib.namespace import RDF

from graphrag.domains.energy.demo import ENERGY, REC, EnergyDemoService, TENANT
from graphrag.ingestion.r2rml import r2rml_to_mapping
from scripts.create_energy_demo_sqlite import create


def test_energy_demo_answers_current_and_historical_questions_with_evidence():
    service = EnergyDemoService()
    current = service.answer("maintenance_review", tenant=TENANT)
    historical = service.answer("historical_state", tenant=TENANT, as_of="2026-05-01T00:00:00Z")
    assert "WT-01" in current["answer"]
    assert current["authoritative_bulletin"] == "MFG-GBX-17-R2"
    assert current["evidence"]
    assert historical["authoritative_bulletin"] == "MFG-GBX-17-R1"


def test_energy_demo_does_not_leak_evidence_to_another_tenant():
    result = EnergyDemoService().answer("maintenance_review", tenant="another-tenant")
    assert result == {"status": "not_found", "answer": "No energy demonstration is available for this tenant.", "evidence": []}


def test_energy_demo_exports_parseable_rdf_and_rejects_invalid_candidate():
    service = EnergyDemoService()
    graph = Graph().parse(data=service.export_turtle(), format="turtle")
    assert len(graph) > 50
    report = service.validate_candidate()
    assert report["conforms"] is False
    assert report["rejected_records"]
    assert any("unit" in violation.lower() for violation in report["violations"])


def test_energy_r2rml_mapping_parses_against_reproducible_source_contract(tmp_path):
    database = tmp_path / "energy.sqlite"
    create(database)
    mapping = r2rml_to_mapping(
        "ontology/mappings/energy-assets.r2rml.ttl", mapping_id="energy-assets",
        version="1.0.0", source_id="synthetic-sap", tenant=TENANT,
    )
    assert [entity.table for entity in mapping.entities] == ["sap_assets", "sap_work_orders"]
    assert database.exists()


def test_energy_demo_materializes_the_sqlite_source_and_uses_sparql_for_review(tmp_path):
    database = tmp_path / "energy.sqlite"
    create(database)
    result = EnergyDemoService(source_db=database).answer("maintenance_review", tenant=TENANT)
    assert result["answer_source"] == "version-controlled SPARQL query"
    assert result["query_rows"] == [{
        "asset": "https://example.energy.demo/asset/WT-01",
        "temperature": "96.0",
        "threshold": "85.0",
        "bulletin": "https://example.energy.demo/document/MFG-GBX-17-R2",
        "workOrder": "https://example.energy.demo/record/WO-9001",
    }]


def test_the_real_pipeline_publishes_with_zero_quarantined_records():
    """A genuine regression pin: the real R2RML/RML-materialized data is
    fully SHACL-conformant today (item 2's mapping-completion work, plus
    the ill-typed-decimal-literal fix this publication-gate work found),
    so nothing should ever be quarantined in normal operation."""
    service = EnergyDemoService()
    report = service.publication_report()
    assert report.quarantined_records == []
    assert report.published_triple_count > 0


def test_rollback_is_wired_through_the_service_and_refreshes_self_graph():
    """EnergyDemoService only ever publishes once during __init__, so this
    proves the wiring (not DatasetPublisher's own rollback logic, already
    covered by tests/unit/test_energy_publication.py) by staging a second
    version directly through the same publisher the service already holds,
    then rolling back through the service's own public method."""
    service = EnergyDemoService()
    first_report = service.publication_report()
    graph_before_second_publish = set(service.graph)

    second_candidate = Graph()
    for triple in service.graph:
        second_candidate.add(triple)
    second_candidate.add((REC["extra"], RDF.type, ENERGY.Site))
    service._publisher.stage_and_publish(second_candidate)
    service.graph = service._publisher.current
    assert set(service.graph) != graph_before_second_publish

    rolled_back_report = service.rollback()

    assert rolled_back_report.rolled_back_from == first_report.version_id
    assert set(service.graph) == graph_before_second_publish
    assert len(service.publication_history()) == 3


def test_explicit_and_ephemeral_source_db_produce_the_identical_graph(tmp_path):
    """No source_db (the default) now builds an ephemeral SQLite fixture and
    runs the same real R2RML materialization an explicit source_db does --
    the two code paths that used to independently hand-write the same data
    are now genuinely one path. Proves that directly: same triple content
    either way, not just individually-passing assertions."""
    database = tmp_path / "energy.sqlite"
    create(database)

    explicit = EnergyDemoService(source_db=database).graph
    ephemeral = EnergyDemoService().graph

    assert set(explicit) == set(ephemeral)
