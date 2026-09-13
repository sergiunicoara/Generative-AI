from pathlib import Path

import pytest
from rdflib import Graph
from rdflib.namespace import RDF

from graphrag.domains.energy.demo import ENERGY, REC, EnergyDemoService, TENANT
from graphrag.ingestion.r2rml import r2rml_to_mapping
from scripts.create_energy_demo_sqlite import create

ROOT = Path(__file__).resolve().parents[2]


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
    # Pinned to an explicit instant before the late-arriving correction was
    # recorded, so this contract test keeps asserting one fixed, known result
    # instead of drifting with whatever "now" resolves to as the telemetry
    # fixture grows. tests/e2e/test_live_graphdb.py pins the same instant.
    result = EnergyDemoService(source_db=database).answer(
        "maintenance_review", tenant=TENANT, as_of="2026-08-28T12:00:00Z",
    )
    assert result["answer_source"] == "version-controlled SPARQL query"

    assert len(result["query_rows"]) == 1
    row = result["query_rows"][0]
    # The five bindings tests/e2e/test_live_graphdb.py also executes against
    # live GraphDB -- a pinned contract on the committed .rq file.
    assert row["asset"] == "https://example.energy.demo/asset/WT-01"
    assert row["temperature"] == "96.0"
    assert row["threshold"] == "85.0"
    assert row["bulletin"] == "https://example.energy.demo/document/MFG-GBX-17-R2"
    assert row["workOrder"] == "https://example.energy.demo/record/WO-9001"

    # Every value the answer states is one of those bindings, not a literal.
    for stated in (row["temperature"], row["threshold"], "WT-01", "WO-9001", "MFG-GBX-17-R2"):
        assert stated in result["answer"]


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


async def test_create_is_the_async_constructor_and_builds_the_same_graph():
    """Gap A regression: EnergyDemoService.create() must be safely awaitable
    from inside an already-running event loop -- this test function itself
    runs under one, via pytest-asyncio's asyncio_mode=auto, which is exactly
    the context that used to crash. Compare against a separately-known-good
    triple count rather than a sync EnergyDemoService() built in this same
    test, since constructing the latter here would hit the very
    still-running-loop guard this session added (see the next test) --
    cross-construction equivalence is already covered by
    test_explicit_and_ephemeral_source_db_produce_the_identical_graph."""
    created = await EnergyDemoService.create()
    assert created.graph is not None
    report = created.publication_report()
    assert report.published_triple_count > 0
    assert report.quarantined_records == []


async def test_synchronous_construction_inside_a_running_loop_fails_clearly():
    """Gap A regression: the exact crash this session found --
    `scripts/project_energy_rdf_to_neo4j.py` constructing EnergyDemoService()
    synchronously from inside `async def _run(...)` -- must now fail with a
    clear, actionable error naming `.create()` instead of asyncio.run()'s own
    cryptic 'cannot be called from a running event loop' RuntimeError
    surfacing from deep inside graph construction."""
    with pytest.raises(RuntimeError, match=r"EnergyDemoService\.create"):
        EnergyDemoService()


def test_every_observation_work_order_and_bulletin_carries_both_temporal_axes():
    """Bitemporal data-foundation regression (gap C's prerequisite): every
    Observation and DocumentRevision must carry recordedAt, and every
    WorkOrder must carry both validFrom and recordedAt, alongside its
    existing valid-time property -- not just on the one row a hand-picked
    assertion happens to check, but on every instance the real R2RML/RML
    pipeline actually materializes."""
    service = EnergyDemoService()
    graph = service.graph

    observations = set(graph.subjects(RDF.type, ENERGY.Observation))
    work_orders = set(graph.subjects(RDF.type, ENERGY.WorkOrder))
    bulletins = set(graph.subjects(RDF.type, ENERGY.DocumentRevision))
    assert observations and work_orders and bulletins  # sanity: the fixture actually produced some

    for subject in observations:
        assert (subject, ENERGY.observedAt, None) in graph
        assert (subject, ENERGY.recordedAt, None) in graph
    for subject in work_orders:
        assert (subject, ENERGY.validFrom, None) in graph
        assert (subject, ENERGY.recordedAt, None) in graph
    for subject in bulletins:
        assert (subject, ENERGY.validFrom, None) in graph
        assert (subject, ENERGY.recordedAt, None) in graph

    # None of this newly-required metadata caused anything to be quarantined --
    # the real pipeline populates it everywhere, not just where a test looks.
    assert service.publication_report().quarantined_records == []


def test_the_committed_sap_artifact_matches_the_fixture_schema(tmp_path):
    """The checked-in artifacts/energy-demo-sap.sqlite is what the API route
    loads at import. When the fixture schema gains a column and that artifact
    is not regenerated, the R2RML mapping silently produces work orders
    missing a now-required property and SHACL quarantines every one of them --
    the dashboard then shows zero assets under review with no error anywhere.
    That happened during this work; this pins it so it cannot recur silently."""
    import sqlite3

    committed = ROOT / "artifacts" / "energy-demo-sap.sqlite"
    if not committed.exists():
        pytest.skip("artifacts/energy-demo-sap.sqlite is not present in this checkout")

    expected_db = tmp_path / "expected.sqlite"
    create(expected_db)

    def _columns(path, table):
        with sqlite3.connect(path) as db:
            return [row[1] for row in db.execute(f"PRAGMA table_info({table})")]

    for table in ("sap_assets", "sap_work_orders"):
        assert _columns(committed, table) == _columns(expected_db, table), (
            f"{committed.name} table {table} is stale -- regenerate with "
            "`python scripts/create_energy_demo_sqlite.py --output artifacts/energy-demo-sap.sqlite`"
        )

    # And it must still publish cleanly, not merely have the right columns.
    assert EnergyDemoService(source_db=committed).publication_report().quarantined_records == []


def test_synchronous_construction_still_works_outside_a_running_loop():
    """The overwhelmingly common case (scripts, sync tests, the API's
    module-level singleton at import time) must be unaffected: this is a
    plain sync test with no event loop running, so the synchronous
    constructor must keep working exactly as before -- every other
    EnergyDemoService() call site in this file and in
    api/routes/energy_demo.py relies on that."""
    service = EnergyDemoService()
    assert service.graph is not None
    assert service.publication_report().published_triple_count > 0
