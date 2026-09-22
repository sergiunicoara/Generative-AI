from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graphrag.core.models import CitationEvidence, EvalResult, QueryResult
from graphrag.evidence.claim_graph import build_claim_evidence_graph, persist_claim_evidence_graph


def _result() -> QueryResult:
    return QueryResult(
        query_id="q-1", question="What applies?",
        answer="FAA AD 2024-01-02 applies. Southwest must comply.",
        contexts=["FAA AD 2024-01-02 applies to the fleet."],
        citations=["FAA-AD-2024-01-02", "SWA_fleet_registry_2024"],
        retrieval_mode="hybrid",
        correlation_id="corr-q-1", source_trace_id="trace-q-1",
    )


def _result_with_evidence() -> QueryResult:
    result = _result()
    result.evidence = [
        CitationEvidence(
            source_id="FAA-AD-2024-01-02", source_label="FAA-AD-2024-01-02",
            path="[FAA-AD-2024-01-02]", valid_from="2024-01-02", confidence=0.91,
        ),
        CitationEvidence(
            source_id="SWA_fleet_registry_2024", source_label="SWA_fleet_registry_2024",
            path="[SWA_fleet_registry_2024]", confidence=0.5,
        ),
    ]
    return result


def test_claim_graph_preserves_provenance_without_overclaiming_sentence_proof():
    graph = build_claim_evidence_graph(
        _result(), EvalResult(job_id="j-1", query_id="q-1", faithfulness=0.92), tenant="aerospace",
    )
    assert len(graph.claims) == 2
    assert len(graph.artifacts) == 3
    assert graph.supported_by
    assert graph.checks[0].status == "passed"
    assert any(check.check_type == "judge_retrieve_abstain" for check in graph.checks)


def test_claim_graph_persists_versioned_deterministic_rubrics():
    result = _result()
    evaluation = EvalResult(
        job_id="j-1", query_id="q-1", faithfulness=0.92,
        rubric_results=[{
            "rubric_id": "tenant_scope_preserved", "version": "1.0",
            "passed": True, "score": 1.0, "reason": "tenant=t1",
        }],
    )
    graph = build_claim_evidence_graph(result, evaluation, tenant="aerospace")
    check = next(item for item in graph.checks if item.check_type == "tenant_scope_preserved")
    assert check.status == "passed"
    assert check.version == "1.0"
    assert graph.validated_by
    assert graph.actions[0].correlation_id == "corr-q-1"
    assert graph.actions[0].source_trace_id == "trace-q-1"


def test_claim_graph_uses_structured_evidence_when_present():
    graph = build_claim_evidence_graph(
        _result_with_evidence(), EvalResult(job_id="j-1", query_id="q-1", faithfulness=0.92), tenant="aerospace",
    )
    document_artifacts = [a for a in graph.artifacts if a.artifact_type == "document"]
    assert len(document_artifacts) == 2
    faa = next(a for a in document_artifacts if a.external_id == "FAA-AD-2024-01-02")
    assert faa.source_label == "FAA-AD-2024-01-02"
    assert faa.path == "[FAA-AD-2024-01-02]"
    assert faa.valid_from == "2024-01-02"
    assert faa.confidence == 0.91


def test_claim_graph_falls_back_to_opaque_citations_when_evidence_empty():
    # Byte-identical to the pre-`evidence` behavior when a caller only sets
    # `citations` (e.g. every existing test/caller that predates this field).
    graph = build_claim_evidence_graph(
        _result(), EvalResult(job_id="j-1", query_id="q-1", faithfulness=0.92), tenant="aerospace",
    )
    document_artifacts = [a for a in graph.artifacts if a.artifact_type == "document"]
    assert len(document_artifacts) == 2
    assert all(a.source_label == "" and a.confidence is None for a in document_artifacts)


@pytest.mark.asyncio
async def test_claim_graph_persists_tenant_scoped_relationships():
    neo4j = SimpleNamespace(run=AsyncMock(return_value=[]))
    graph = build_claim_evidence_graph(
        _result(), EvalResult(job_id="j-1", query_id="q-1", faithfulness=0.5), tenant="aerospace",
    )
    await persist_claim_evidence_graph(neo4j, graph)
    assert neo4j.run.await_count == 7
    calls = neo4j.run.await_args_list
    assert "SUPPORTED_BY" in calls[4].args[0]
    assert calls[0].kwargs["items"][0]["tenant"] == "aerospace"
