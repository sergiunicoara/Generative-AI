"""Phase 6: explanation and proof traces."""
from __future__ import annotations

import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag.retrieval.explanation import build_explanation, grounding_check

CHUNKS = [
    {"chunk_id": "c1", "_doc_name": "AD-2024-03", "text": "The inspection interval for the main landing gear is 600 flight hours.",
     "final_score": 0.8, "trust": {"current": True, "origin": "EXTRACTED", "verification_status": "UNVERIFIED",
                                   "trust_factor": 0.9, "factors": {"authority": 1.0, "origin": 1.0, "verification": 0.9,
                                                                   "temporal": 1.0, "supersession": 1.0, "staleness": 1.0}},
     "score_components": {"vector_score": 0.82, "bm25_score": 6.1, "rerank_score": 2.3, "gnn_score": 0.4,
                          "final_score": 0.8}},
    {"chunk_id": "c2", "_doc_name": "SB-OLD", "text": "Older bulletin: interval 400 flight hours.", "final_score": 0.3,
     "path_length": 2, "via_entity": "Landing Gear", "path_confidence": 0.7,
     "trust": {"current": False, "superseded": True, "trust_factor": 0.45,
               "factors": {"supersession": 0.5, "temporal": 1.0, "staleness": 1.0, "origin": 1.0, "verification": 0.9}},
     "score_components": {"path_score": 0.35, "final_score": 0.3}},
]
EDGES = [
    {"src": "Acme", "relation": "OWNS", "tgt": "Bolt", "confidence": 0.72, "confidence_before_trust": 0.8,
     "trust": {"origin": "EXTRACTED", "confidence_state": "ASSERTED", "verification_status": "UNVERIFIED",
               "factors": {"verification": 0.9}}, "source_doc_id": "d1"},
    {"src": "Acme", "relation": "OWNS", "tgt": "Nut", "confidence": 0.5, "source_type": "inferred",
     "inferred_by": "owns_transitive", "rule_version": "abc123", "premise_keys": ["ORG:Acme|OWNS|ORG:Bolt",
                                                                                "ORG:Bolt|OWNS|ORG:Nut"],
     "trust": {"origin": "INFERRED", "confidence_state": "INFERRED"}},
]
ENTITIES = [{"entity": "Acme", "type": "ORG", "resolution_status": "auto_resolved", "resolution_method": "alias"}]
LOCAL = {"chunks": CHUNKS, "entity_edges": EDGES, "entities": ENTITIES}
ROUTE = {"route": "RELATIONAL", "reason": "legacy_relational"}


def explain(**kw):
    args = dict(answer="The inspection interval for the main landing gear is 600 flight hours [AD-2024-03].",
                route=ROUTE, local_results=LOCAL, evidence=[], citations=["AD-2024-03"],
                sufficiency={"sufficient": True, "reason_code": "sufficient", "evidence_count": 2},
                schema_version="aero@1.0.0#abcdef123456", fallback={"triggered": False, "reason": None},
                acl_enforced=False, router_policy="enforce")
    args.update(kw)
    return build_explanation(**args)


def test_response_structure_matches_the_spec():
    e = explain()
    for key in ("answer", "confidence", "route", "evidence", "graph_paths", "inferences", "score_breakdown",
                "schema_version", "fallback", "limitations"):
        assert key in e
    assert e["route"] == "RELATIONAL" and e["schema_version"].startswith("aero@")
    assert e["fallback"] == {"triggered": False, "reason": None}
    assert 0.0 <= e["confidence"] <= 1.0
    json.dumps(e)  # serialisable


def test_cited_evidence_supports_the_answer():
    e = explain()
    cited = [i for i in e["evidence"] if i["cited"]]
    assert [i["id"] for i in cited] == ["c1"]
    assert e["grounding"]["unsupported_statements"] == [] and e["grounding"]["grounding_ratio"] == 1.0
    assert "600 flight hours" in cited[0]["excerpt"]


def test_unsupported_statements_are_not_presented_as_grounded():
    e = explain(answer="The interval is 600 flight hours. Boeing also recalled every turbine blade worldwide last spring.")
    assert len(e["grounding"]["unsupported_statements"]) == 1
    assert "turbine" in e["grounding"]["unsupported_statements"][0]
    assert any(l["code"] == "unsupported_statements" for l in e["limitations"])
    assert e["confidence"] < explain()["confidence"]


def test_grounding_ignores_short_fragments():
    g = grounding_check("Yes. See above.", ["anything"])
    assert g["statements"] == 0 and g["grounding_ratio"] == 1.0


def test_inferred_facts_include_their_derivation():
    e = explain()
    [inf] = e["inferences"]
    assert inf["rule"] == "owns_transitive" and inf["rule_version"] == "abc123"
    assert inf["premises"] == ["ORG:Acme|OWNS|ORG:Bolt", "ORG:Bolt|OWNS|ORG:Nut"]
    assert inf["derivation_recorded"] is True


def test_inference_without_premises_is_flagged():
    edges = [{**EDGES[1], "premise_keys": None}]
    e = explain(local_results={**LOCAL, "entity_edges": edges})
    assert any(l["code"] == "inference_without_recorded_premises" for l in e["limitations"])


def test_graph_paths_and_entity_resolution_with_trust():
    e = explain()
    edge = e["graph_paths"][0]["edges"][0]
    assert edge["origin"] == "EXTRACTED" and edge["confidence_before_trust"] == 0.8
    assert any(p.get("via_entity") == "Landing Gear" for p in e["graph_paths"])
    assert e["entity_resolution"][0] == {"entity": "Acme", "type": "ORG", "resolution_status": "auto_resolved",
                                         "resolution_method": "alias"}


def test_score_breakdown_has_separate_components():
    sb = explain()["score_breakdown"]
    assert sb["vector"] == 0.82 and sb["bm25"] == 6.1 and sb["graph"] == 0.4
    assert sb["temporal"] == 1.0 and sb["authority"] == 1.0 and sb["provenance"] == pytest.approx(0.9)
    assert "c1" in sb["per_evidence"]


def test_non_current_evidence_is_disclosed():
    e = explain(citations=["AD-2024-03", "SB-OLD"])
    assert any(l["code"] == "non_current_evidence" for l in e["limitations"])
    old = next(i for i in e["evidence"] if i["id"] == "c2")
    assert old["current"] is False and old["trust"]["superseded"] is True


def test_unauthorized_graph_data_never_appears_under_acl():
    e = explain(acl_enforced=True)
    assert e["graph_paths"] == [] and e["inferences"] == [] and e["entity_resolution"] == []
    blob = json.dumps(e)
    for secret in ("Nut", "owns_transitive", "auto_resolved", "Landing Gear"):
        assert secret not in blob
    assert any(l["code"] == "graph_explanation_withheld" for l in e["limitations"])


def test_insufficient_evidence_produces_an_honest_failure():
    e = explain(answer="I can’t provide a grounded answer because all retrieved evidence is superseded.",
                sufficiency={"sufficient": False, "reason_code": "all_evidence_stale", "evidence_count": 1},
                citations=[])
    assert e["insufficient_context"] == {"reason": "all_evidence_stale", "evidence_count": 1}
    assert e["confidence"] <= 0.25
    assert any(l["code"] == "insufficient_context:all_evidence_stale" for l in e["limitations"])


def test_no_evidence_means_zero_confidence():
    e = explain(local_results={"chunks": [], "entity_edges": [], "entities": []}, citations=[])
    assert e["confidence"] == 0.0 and e["evidence"] == []


def test_fallback_is_reported_with_its_reason():
    e = explain(fallback={"triggered": True, "reason": "low_confidence"})
    assert e["fallback"] == {"triggered": True, "reason": "low_confidence"}
    assert any(l["code"] == "agentic_fallback" for l in e["limitations"])


def test_observe_policy_and_unregistered_schema_are_limitations():
    e = explain(router_policy="observe", schema_version="platform/v1")
    codes = {l["code"] for l in e["limitations"]}
    assert {"route_observed_only", "schema_unregistered"} <= codes


# ── wiring ────────────────────────────────────────────────────────────────────

def test_every_answer_path_attaches_an_explanation():
    from graphrag.retrieval import hybrid_retriever
    src = inspect.getsource(hybrid_retriever)
    assert src.count("result.explanation = build_explanation(") == 2      # synthesis + agentic fallback
    assert 'result.explanation = {' in src                                # structured template path
    assert '"triggered": True, "reason": fallback_reason' in src


def test_async_api_no_longer_drops_explanation_fields():
    from graphrag.messaging import consumers
    src = inspect.getsource(consumers)
    for field in ('"explanation"', '"retrieval_sufficiency"', '"evidence_bundle"', '"retrieval_trajectory"',
                  '"route"', '"schema_version"', '"confidence"'):
        assert field in src


# ── decision traces are authorization-filtered ───────────────────────────────

def _trace():
    return {
        "decision": {"id": "dec1", "tenant": "acme", "answer": "Protected fact X", "status": "final"},
        "manifest": {"id": "m1", "document_ids": ["pub", "secret"], "chunk_ids": ["c-pub", "c-sec"],
                     "task_input": "question"},
        "chunks": [{"id": "c-pub", "document_id": "pub", "text": "public", "embedding": [0.1]},
                   {"id": "c-sec", "document_id": "secret", "text": "secret text", "embedding": [0.2]}],
        "documents": [{"id": "pub"}, {"id": "secret"}],
        "observations": [{"content": "secret text"}],
    }


@pytest.mark.asyncio
async def test_trace_removes_unauthorized_evidence_and_redacts_decision():
    from graphrag.context_graph.repository import ContextGraphRepository

    neo = MagicMock()
    neo.run = AsyncMock(return_value=[{"id": "pub"}])
    repo = ContextGraphRepository.__new__(ContextGraphRepository)
    repo._neo4j = neo
    settings = MagicMock()
    settings.access_control = {"enabled": True}
    with patch("graphrag.core.config.get_settings", return_value=settings):
        out = await repo._authorize_trace(_trace(), "acme", None)
    assert [c["id"] for c in out["chunks"]] == ["c-pub"] and [d["id"] for d in out["documents"]] == ["pub"]
    assert out["manifest"]["document_ids"] == ["pub"] and out["manifest"]["chunk_ids"] == ["c-pub"]
    assert "Protected" not in json.dumps(out) and "secret text" not in json.dumps(out)
    assert out["decision"]["id"] == "dec1" and out["decision"]["status"] == "final"
    assert out["redaction"]["removed_documents"] == 1
    assert all("embedding" not in c for c in out["chunks"])
    q, kw = neo.run.await_args.args[0], neo.run.await_args.kwargs
    assert kw["tenant"] == "acme" and kw["acl_enabled"] is True and "acl_state" in q


@pytest.mark.asyncio
async def test_trace_without_acl_only_strips_embeddings():
    from graphrag.context_graph.repository import ContextGraphRepository

    repo = ContextGraphRepository.__new__(ContextGraphRepository)
    repo._neo4j = MagicMock()
    settings = MagicMock()
    settings.access_control = {"enabled": False}
    with patch("graphrag.core.config.get_settings", return_value=settings):
        out = await repo._authorize_trace(_trace(), "acme", None)
    assert len(out["chunks"]) == 2 and "redaction" not in out
    assert all("embedding" not in c for c in out["chunks"])


def test_trace_routes_pass_the_callers_access_context():
    from api.routes import context_graph
    for fn in (context_graph.load_context_trace, context_graph.replay_context_trace):
        assert "AccessContext.from_claims(user)" in inspect.getsource(fn)
