"""Regression tests for governed ontology drift and fuzzy candidate indexing."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graphrag.core.models import Chunk, Entity, Relation
from graphrag.graph.alias_registry import AliasRegistry, _normalize
from graphrag.graph.ontology_proposals import (
    OntologyProposalService,
    build_ontology_proposals,
)
from graphrag.graph.ontology_registry import OntologyRegistry


def _entity(name: str, entity_type: str) -> Entity:
    return Entity(name=name, type=entity_type)


def test_entity_canonical_identity_is_a_scoped_natural_key():
    entity = _entity("Space Exploration Technologies", "ORG")
    assert entity.canonical_identity is None
    entity.redirect_to("SpaceX", "ORG")
    assert entity.canonical_identity == ("SpaceX", "ORG")
    assert "canonical_id" not in entity.model_dump()


def test_fuzzy_length_index_keeps_only_theoretically_possible_candidates():
    registry = AliasRegistry(AsyncMock())
    registry._exact = {
        _normalize("Acme Corporation"): ("Acme Corporation", "ORG"),
        "x" * 100: ("Very Long", "ORG"),
        "y" * 120: ("Very Long Two", "ORG"),
    }

    candidates = registry._fuzzy_candidates(_normalize("Acme Corp"), minimum_score=70)

    assert _normalize("Acme Corporation") in candidates
    assert "x" * 100 not in candidates
    assert "y" * 120 not in candidates


@pytest.mark.asyncio
async def test_unknown_relation_is_rejected_in_strict_mode_not_promoted():
    registry = OntologyRegistry(AsyncMock(), tenant="acme")
    registry._allowed_types = {"ORG"}
    registry._loaded = True
    source = _entity("Acme", "ORG")
    target = _entity("Beta", "ORG")
    relation = Relation(source_entity_id=source.id, target_entity_id=target.id, relation="PARTNERS_WITH")

    report = registry.validate_extraction([source, target], [relation], strict=True)

    assert report["new_relations"] == ["PARTNERS_WITH"]
    assert report["rejected_relation_ids"] == [relation.id]
    assert "PARTNERS_WITH" not in registry._known_relations


def test_build_proposals_keeps_source_grounded_schema_candidates():
    chunk = Chunk(id="chunk-1", document_id="doc-1", text="Acme partners with Beta.", chunk_index=0, tenant="acme")
    source = _entity("Acme", "ORG")
    target = _entity("Beta", "VENDOR")
    relation = Relation(source_entity_id=source.id, target_entity_id=target.id, relation="PARTNERS_WITH")
    report = {
        "rejected_entity_ids": [target.id],
        "rejected_relation_ids": [relation.id],
        "new_relations": ["PARTNERS_WITH"],
    }

    proposals = build_ontology_proposals(report, [source, target], [relation], chunk)

    assert {(p["kind"], p["proposed_value"]) for p in proposals} == {
        ("entity_type", "VENDOR"), ("relation", "PARTNERS_WITH"),
    }
    # Neither value was proposed under more than one kind, so no conflicts.
    assert all(p["conflicts"] == [] for p in proposals)
    assert all(p["confidence"] == 1.0 for p in proposals)


def test_build_proposals_flags_same_value_proposed_under_two_kinds():
    chunk = Chunk(id="chunk-1", document_id="doc-1", text="text", chunk_index=0, tenant="acme")
    source = _entity("Acme", "ORG")
    target = _entity("Beta", "SUPPLIER")
    entity_target = _entity("Gamma", "SUPPLIER")
    relation = Relation(source_entity_id=source.id, target_entity_id=target.id, relation="SUPPLIER")
    report = {
        "rejected_entity_ids": [target.id, entity_target.id],
        "rejected_relation_ids": [relation.id],
        "new_relations": ["SUPPLIER"],
    }

    proposals = build_ontology_proposals(report, [source, target, entity_target], [relation], chunk)

    by_kind = {p["kind"]: p for p in proposals}
    assert by_kind["entity_type"]["conflicts"] == ["relation"]
    assert by_kind["relation"]["conflicts"] == ["entity_type"]


@pytest.mark.asyncio
async def test_proposal_service_deduplicates_by_tenant_fingerprint_and_decides():
    neo4j = AsyncMock()
    neo4j.run = AsyncMock(side_effect=[
        [{"id": "proposal-1"}],
        [{"id": "proposal-1", "kind": "entity_type", "proposed_value": "VENDOR", "status": "approved"}],
    ])
    service = OntologyProposalService(neo4j)
    chunk = Chunk(id="chunk-1", document_id="doc-1", text="evidence", chunk_index=0, tenant="acme")

    proposal_ids = await service.submit(
        [{"kind": "entity_type", "proposed_value": "VENDOR", "entity_name": "Beta", "source_type": "", "target_type": "", "reason": "unknown_entity_type"}],
        chunk,
        ontology_version_id="version-1",
    )
    decision = await service.decide("proposal-1", approve=True, reviewed_by="architect", tenant="acme")

    assert proposal_ids == ["proposal-1"]
    assert decision["status"] == "approved"
    submit_query = neo4j.run.await_args_list[0].args[0]
    assert "MERGE (p:OntologyProposal {tenant: $tenant, fingerprint: $fingerprint})" in submit_query
    assert neo4j.run.await_args_list[0].kwargs["tenant"] == "acme"
    # Legacy approve=True path still carries actor + timestamp evidence.
    assert neo4j.run.await_args_list[1].kwargs["reviewed_by"] == "architect"


@pytest.mark.asyncio
async def test_decide_rejects_when_neither_action_nor_approve_given():
    service = OntologyProposalService(AsyncMock())
    result = await service.decide("proposal-1", reviewed_by="architect", tenant="acme")
    assert "error" in result


@pytest.mark.asyncio
async def test_decide_rejects_unknown_action():
    service = OntologyProposalService(AsyncMock())
    result = await service.decide("proposal-1", action="delete", reviewed_by="architect", tenant="acme")
    assert "error" in result


@pytest.mark.asyncio
async def test_decide_edit_action_overrides_proposed_value_and_records_reason():
    neo4j = AsyncMock()
    neo4j.run = AsyncMock(return_value=[{"id": "proposal-1", "kind": "entity_type", "proposed_value": "SUPPLIER", "status": "edited"}])
    service = OntologyProposalService(neo4j)

    result = await service.decide(
        "proposal-1", action="edit", reviewed_by="architect", tenant="acme",
        reason="typo in extraction", model_version="groq-llama-4", edited_value="SUPPLIER",
    )

    assert result["status"] == "edited"
    cypher = neo4j.run.await_args.args[0]
    kwargs = neo4j.run.await_args.kwargs
    assert "p.proposed_value = $edited_value" in cypher
    assert kwargs["edited_value"] == "SUPPLIER"
    assert kwargs["reason"] == "typo in extraction"
    assert kwargs["model_version"] == "groq-llama-4"


@pytest.mark.asyncio
async def test_decide_merge_action_records_merge_target():
    neo4j = AsyncMock()
    neo4j.run = AsyncMock(return_value=[{"id": "proposal-1", "kind": "relation", "proposed_value": "SUPPLIES", "status": "merged"}])
    service = OntologyProposalService(neo4j)

    result = await service.decide(
        "proposal-1", action="merge", reviewed_by="architect", tenant="acme", merge_target="SUPPLIER_OF",
    )

    assert result["status"] == "merged"
    kwargs = neo4j.run.await_args.kwargs
    assert kwargs["merge_target"] == "SUPPLIER_OF"


@pytest.mark.asyncio
async def test_submit_passes_intake_and_prioritisation_fields_through():
    neo4j = AsyncMock()
    neo4j.run = AsyncMock(return_value=[{"id": "proposal-1"}])
    service = OntologyProposalService(neo4j)
    chunk = Chunk(id="chunk-1", document_id="doc-1", text="evidence", chunk_index=0, tenant="acme")

    await service.submit(
        [{
            "kind": "entity_type", "proposed_value": "VENDOR",
            "business_impact": "high", "urgency": "next_release", "effort": "small",
            "risk": "low", "dependencies": ["schema-migration-12"], "requesting_team": "supply-chain",
        }],
        chunk,
    )

    kwargs = neo4j.run.await_args.kwargs
    assert kwargs["business_impact"] == "high"
    assert kwargs["urgency"] == "next_release"
    assert kwargs["effort"] == "small"
    assert kwargs["risk"] == "low"
    assert kwargs["dependencies"] == ["schema-migration-12"]
    assert kwargs["requesting_team"] == "supply-chain"


@pytest.mark.asyncio
async def test_decide_defer_and_quarantine_set_matching_status():
    neo4j = AsyncMock()
    neo4j.run = AsyncMock(side_effect=[
        [{"id": "p1", "kind": "entity_type", "proposed_value": "X", "status": "deferred"}],
        [{"id": "p2", "kind": "entity_type", "proposed_value": "Y", "status": "quarantined"}],
    ])
    service = OntologyProposalService(neo4j)

    deferred = await service.decide("p1", action="defer", reviewed_by="architect", tenant="acme")
    quarantined = await service.decide("p2", action="quarantine", reviewed_by="architect", tenant="acme")

    assert deferred["status"] == "deferred"
    assert quarantined["status"] == "quarantined"


@pytest.mark.asyncio
async def test_export_golden_set_shapes_decided_proposals_as_input_label_pairs():
    neo4j = AsyncMock()
    neo4j.run = AsyncMock(return_value=[
        {
            "id": "p1", "kind": "entity_type", "proposed_value": "SUPPLIER",
            "entity_name": "Beta", "source_type": "", "target_type": "",
            "reason": "unknown_entity_type", "confidence": 0.8, "status": "edited",
            "reviewed_by": "architect", "reviewed_at": "2026-09-18T00:00:00Z",
            "decision_reason": "typo", "decision_model_version": "groq-llama-4",
            "merge_target": None,
        },
        {
            "id": "p2", "kind": "relation", "proposed_value": "SUPPLIES",
            "entity_name": "", "source_type": "ORG", "target_type": "SUPPLIER",
            "reason": "unknown_relation", "confidence": 1.0, "status": "merged",
            "reviewed_by": "architect", "reviewed_at": "2026-09-18T00:01:00Z",
            "decision_reason": "", "decision_model_version": "",
            "merge_target": "SUPPLIER_OF",
        },
    ])
    service = OntologyProposalService(neo4j)

    records = await service.export_golden_set("acme")

    assert len(records) == 2
    edited = next(r for r in records if r["id"] == "p1")
    assert edited["label"] == "edited"
    assert edited["corrected_value"] == "SUPPLIER"
    assert edited["merge_target"] is None
    assert edited["input"]["extraction_confidence"] == 0.8

    merged = next(r for r in records if r["id"] == "p2")
    assert merged["label"] == "merged"
    assert merged["corrected_value"] is None
    assert merged["merge_target"] == "SUPPLIER_OF"

    kwargs = neo4j.run.await_args.kwargs
    assert kwargs["tenant"] == "acme"
    assert set(kwargs["statuses"]) == {"approved", "rejected", "edited", "merged", "deferred", "quarantined"}


@pytest.mark.asyncio
async def test_status_report_aggregates_by_status_and_kind():
    neo4j = AsyncMock()
    neo4j.run = AsyncMock(return_value=[
        {"status": "pending", "kind": "entity_type", "count": 3},
        {"status": "approved", "kind": "entity_type", "count": 5},
        {"status": "approved", "kind": "relation", "count": 2},
    ])
    service = OntologyProposalService(neo4j)

    report = await service.status_report("acme")

    assert report["tenant"] == "acme"
    assert report["total"] == 10
    assert report["by_status"] == {"pending": 3, "approved": 7}
    assert report["by_kind"] == {"entity_type": 8, "relation": 2}
    assert report["pending"] == 3
