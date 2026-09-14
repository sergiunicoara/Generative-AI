"""Validation and RDF-projection gates for the evidence-remediation workflow."""

from __future__ import annotations

from pathlib import Path

import pytest
from rdflib import Literal, URIRef
from rdflib.namespace import RDF

from graphrag.domains.energy.demo import EnergyDemoService
from graphrag.domains.energy.evidence_requests import (
    EvidenceRequestError, EvidenceRequestService, UnknownAssetError,
)
from graphrag.domains.energy.governance_store import GovernanceStore
from graphrag.domains.energy.vocabulary import ENERGY


async def _runtime(tmp_path: Path) -> tuple[GovernanceStore, EnergyDemoService, EvidenceRequestService]:
    store = GovernanceStore(
        f"sqlite+aiosqlite:///{(tmp_path / 'governance.sqlite').as_posix()}",
        blob_root=tmp_path / "published",
    )
    await store.open()
    service = await EnergyDemoService.create(governance_store=store)
    return store, service, EvidenceRequestService(store, tenant=service.tenant)


def _query_rows(service: EnergyDemoService) -> list[dict[str, str]]:
    return service.answer("insufficient_evidence", tenant=service.tenant)["query_rows"]


async def test_create_evidence_request_for_wt02_missing_temperature_c_selects_snowflake(tmp_path: Path):
    store, service, evidence = await _runtime(tmp_path)
    record, _wire, replayed = await evidence.create(
        asset_id="WT-02", missing_field="temperature_c", source_system="Snowflake",
        owner="ops-team", priority="high", reason="Confirm gearbox temperature",
        created_by="operator-1", command_id="req-wt02", graph=service.graph,
        query_rows=_query_rows(service),
    )

    assert replayed is False
    assert record.asset_id == "WT-02"
    assert record.missing_field == "temperature_c"
    assert record.target_source_system == "Snowflake"
    assert record.state == "open"
    assert record.object_version == 0
    await store.close()


async def test_create_rejects_field_that_is_not_actually_missing(tmp_path: Path):
    """WT-01 has both a temperature_c observation and a work order -- not blocked at all."""
    store, service, evidence = await _runtime(tmp_path)
    with pytest.raises(EvidenceRequestError, match="not currently blocked"):
        await evidence.create(
            asset_id="WT-01", missing_field="temperature_c", source_system="Snowflake",
            owner="ops-team", priority="medium", reason="not actually missing",
            created_by="operator-1", command_id="req-wt01", graph=service.graph,
            query_rows=_query_rows(service),
        )
    await store.close()


async def test_create_rejects_wrong_source_system_mapping(tmp_path: Path):
    store, service, evidence = await _runtime(tmp_path)
    with pytest.raises(EvidenceRequestError, match="owned by Snowflake"):
        await evidence.create(
            asset_id="WT-02", missing_field="temperature_c", source_system="SAP",
            owner="ops-team", priority="medium", reason="wrong system",
            created_by="operator-1", command_id="req-wt02-bad-system", graph=service.graph,
            query_rows=_query_rows(service),
        )
    await store.close()


async def test_create_rejects_unsupported_missing_field(tmp_path: Path):
    store, service, evidence = await _runtime(tmp_path)
    with pytest.raises(EvidenceRequestError, match="unsupported missing_field"):
        await evidence.create(
            asset_id="WT-02", missing_field="pressure_bar", source_system="Snowflake",
            owner="ops-team", priority="medium", reason="unsupported field",
            created_by="operator-1", command_id="req-wt02-bad-field", graph=service.graph,
            query_rows=_query_rows(service),
        )
    await store.close()


async def test_create_rejects_unknown_asset(tmp_path: Path):
    store, service, evidence = await _runtime(tmp_path)
    with pytest.raises(UnknownAssetError):
        await evidence.create(
            asset_id="WT-99", missing_field="temperature_c", source_system="Snowflake",
            owner="ops-team", priority="medium", reason="unknown asset",
            created_by="operator-1", command_id="req-unknown", graph=service.graph,
            query_rows=_query_rows(service),
        )
    await store.close()


async def test_rdf_projection_keeps_each_transition_as_an_independent_resource(tmp_path: Path):
    store, service, evidence = await _runtime(tmp_path)
    record, _wire, _replayed = await evidence.create(
        asset_id="WT-02", missing_field="temperature_c", source_system="Snowflake",
        owner="ops-team", priority="high", reason="Confirm gearbox temperature",
        created_by="operator-1", command_id="req-wt02-rdf", graph=service.graph,
        query_rows=_query_rows(service),
    )
    await evidence.transition(
        record.request_id, to_state="in_progress", changed_by="operator-2",
        reason="Snowflake export requested", expected_version=record.object_version,
        command_id="req-wt02-progress",
    )
    graph = await evidence.rdf_projection(record.request_id)

    transitions = set(graph.subjects(RDF.type, ENERGY.EvidenceRequestTransition))
    assert len(transitions) == 1
    requests = set(graph.subjects(RDF.type, ENERGY.EvidenceRequest))
    expected_request = URIRef(f"https://example.energy.demo/evidence-request/{record.request_id}")
    assert requests == {expected_request}
    transition = transitions.pop()
    assert graph.value(transition, ENERGY.changesRequest) == expected_request
    assert graph.value(transition, ENERGY.toState) == Literal("in_progress")
    await store.close()
