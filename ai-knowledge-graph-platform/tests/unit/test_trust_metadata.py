"""Phase 5: trust metadata integrated into retrieval.

Covers: conflicting sources, different authority, expired evidence, generated
but unverified claims, verified corrections, score-component explainability,
and the shared validity predicate on every fact-returning read path.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag.graph.trust import (
    Origin,
    VerificationStatus,
    apply_chunk_trust,
    apply_edge_trust,
    assess,
    origin_for,
    rank_conflicting_claims,
)
from graphrag.graph.validity import AT, edge_is_current

NOW = datetime(2026, 10, 10, tzinfo=timezone.utc)
PAST = (NOW - timedelta(days=30)).isoformat()
FUTURE = (NOW + timedelta(days=30)).isoformat()
T = "acme"


# ── origin and verification ──────────────────────────────────────────────────

@pytest.mark.parametrize("source_type,explicit,expected", [
    ("document", None, Origin.EXTRACTED), ("inferred", None, Origin.INFERRED),
    ("llm", None, Origin.GENERATED), ("manual", None, Origin.MANUAL),
    ("document", "IMPORTED", Origin.IMPORTED), (None, None, Origin.EXTRACTED),
])
def test_origin_distinguishes_extracted_imported_inferred_generated_manual(source_type, explicit, expected):
    assert origin_for(source_type, explicit) is expected


def test_having_a_source_never_implies_verified():
    a = assess({"source_type": "document", "source_doc_id": "d1", "confidence": 1.0})
    assert a.verification_status == VerificationStatus.UNVERIFIED.value
    assert a.factors["verification"] < 1.0


def test_generated_unverified_claim_is_trusted_less_than_extracted_or_verified():
    generated = assess({"source_type": "llm"})
    extracted = assess({"source_type": "document"})
    verified = assess({"source_type": "document", "verification_status": "VERIFIED", "verified_by": "rev"})
    assert generated.origin == "GENERATED" and generated.verification_status == "UNVERIFIED"
    assert generated.trust_factor < extracted.trust_factor < verified.trust_factor
    assert verified.verified_by == "rev"


def test_rejected_and_retracted_facts_are_not_current():
    assert not assess({"verification_status": "REJECTED"}).current
    assert not assess({"confidence_state": "RETRACTED"}).current
    assert assess({"confidence_state": "RETRACTED"}).trust_factor == 0.0


# ── temporal validity ─────────────────────────────────────────────────────────

def test_expired_not_yet_valid_and_stale_are_not_current():
    assert assess({"valid_to": PAST}, at=NOW).expired
    assert not assess({"valid_to": PAST}, at=NOW).current
    assert assess({"valid_from": FUTURE}, at=NOW).not_yet_valid
    stale = assess({"stale_after": PAST}, at=NOW)
    assert stale.stale and not stale.current and stale.factors["staleness"] < 1.0
    fresh = assess({"valid_to": FUTURE, "stale_after": FUTURE}, at=NOW)
    assert fresh.current and fresh.factors["temporal"] == 1.0


def test_superseded_document_is_not_treated_as_current():
    a = assess({"superseded": True, "authority_level": 1}, kind="chunk")
    assert not a.current and a.factors["supersession"] == 0.5


def test_shared_predicate_excludes_retracted_and_expired_and_defaults_to_now():
    frag = edge_is_current("r")
    assert "'RETRACTED'" in frag and "r.valid_to" in frag and "r.valid_from" in frag
    assert AT == "coalesce(datetime($as_of), datetime())"
    assert "datetime()" in edge_is_current("r", at="datetime()")


# ── read paths use the shared predicate ──────────────────────────────────────

def _client():
    from graphrag.graph.neo4j_client import Neo4jClient
    c = Neo4jClient.__new__(Neo4jClient)
    c.run = AsyncMock(return_value=[])
    return c


@pytest.mark.asyncio
@pytest.mark.parametrize("call", [
    lambda c: c.get_entity_neighbors(["c1"], tenant=T),
    lambda c: c.get_multihop_chunks(["c1"], tenant=T),
    lambda c: c.get_entity_relations_subgraph([{"name": "A", "type": "ORG"}], tenant=T),
    lambda c: c.get_relations_for_entity("A", "ORG", tenant=T),
    lambda c: c.get_all_relations(tenant=T),
])
async def test_every_graph_fact_read_excludes_retracted_and_expired_edges(call):
    c = _client()
    await call(c)
    q, kw = c.run.await_args.args[0], c.run.await_args.kwargs
    assert "'RETRACTED'" in q and "valid_to" in q
    assert "as_of" in kw  # always judged at as_of or now, never skipped


@pytest.mark.asyncio
async def test_neighbours_are_tenant_scoped_on_chunk_and_entity():
    c = _client()
    await c.get_entity_neighbors(["c1"], tenant=T)
    q = c.run.await_args.args[0]
    assert "Chunk {id: cid, tenant: $tenant}" in q and "Entity {tenant: $tenant}" in q


@pytest.mark.asyncio
async def test_fact_reads_return_trust_fields():
    c = _client()
    await c.get_entity_relations_subgraph([{"name": "A", "type": "ORG"}], tenant=T)
    q = c.run.await_args.args[0]
    for field in ("confidence_state", "origin", "verification_status", "stale_after", "schema_version"):
        assert f"AS {field}" in q
    await c.get_relations_for_entity("A", "ORG", tenant=T)
    assert "AS verification_status" in c.run.await_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["vector_search_chunks", "bm25_search_chunks", "bm25_search_entities"])
async def test_chunk_reads_return_document_trust(method):
    c = _client()
    c._filtered_vector_search = False
    c._filtered_vector_indexes = set()
    arg = [0.1, 0.2] if method == "vector_search_chunks" else "query"
    await getattr(c, method)(arg, top_k=3, tenant=T)
    q = c.run.await_args.args[0]
    for field in ("authority_level", "superseded", "doc_valid_to", "doc_stale_after"):
        assert f"AS {field}" in q


def test_controlled_queries_and_agent_tool_use_the_shared_predicate():
    import inspect

    from graphrag.agents.tools import neo4j_tools
    from graphrag.graph import controlled_query as cq
    for q in (cq._ENTITY_RELATIONS_CYPHER, cq._EVIDENCE_GAP_CYPHER):
        assert "'RETRACTED'" in q and "valid_to" in q and "quarantined" in q
    assert "quarantined" in cq._TYPE_ENTITIES_CYPHER
    assert "'RETRACTED'" in inspect.getsource(neo4j_tools.get_neighbors) or \
           "edge_is_current" in inspect.getsource(neo4j_tools.get_neighbors)


# ── write side ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ingested_relations_record_origin_and_are_never_verified_by_ingestion():
    from graphrag.core.models import Relation

    c = _client()
    await c.merge_relation(Relation(source_entity_id="a", target_entity_id="b", relation="OWNS",
                                    extraction_model="gpt-x"), "A", "ORG", "B", "ORG", tenant=T)
    q, kw = c.run.await_args_list[0].args[0], c.run.await_args_list[0].kwargs
    assert kw["origin"] == "EXTRACTED" and kw["generated_by"] == "gpt-x"
    assert "coalesce(r.verification_status, 'UNVERIFIED')" in q
    assert "'VERIFIED'" not in q


def test_relational_import_marks_origin_imported():
    import inspect

    from graphrag.ingestion import relational
    assert 'origin="IMPORTED"' in inspect.getsource(relational)


@pytest.mark.asyncio
async def test_verified_correction_sets_verification_from_reviewer():
    from graphrag.graph.confidence_lifecycle import ConfidenceLifecycleService

    neo = MagicMock()
    neo.run = AsyncMock(side_effect=[[{"state": "ASSERTED"}], []])
    await ConfidenceLifecycleService(neo).transition_relation(
        "A", "ORG", "OWNS", "B", "ORG", "APPROVED", tenant=T, changed_by="reviewer-7")
    q, kw = neo.run.await_args.args[0], neo.run.await_args.kwargs
    assert "WHEN 'APPROVED' THEN 'VERIFIED'" in q and "WHEN 'RETRACTED' THEN 'REJECTED'" in q
    assert kw["changed_by"] == "reviewer-7" and "r.verified_by" in q


def test_manual_override_is_manual_and_verified_by_the_operator():
    import inspect

    from api.routes import corrections
    src = inspect.getsource(corrections.override_edge)
    assert "r.origin       = 'MANUAL'" in src and "r.verification_status = 'VERIFIED'" in src
    assert "r.verified_by  = $override_by" in src


# ── conflicts and authority ───────────────────────────────────────────────────

def test_conflicting_sources_are_ranked_by_authority_without_collapsing():
    out = rank_conflicting_claims([
        {"claim": "informal", "authority_level": 4},
        {"claim": "regulation", "authority_level": 1},
    ])
    assert out["status"] == "suggested" and out["winner"]["claim"] == "regulation"
    assert [c["claim"] for c in out["claims"]] == ["regulation", "informal"]
    assert all("trust" in c for c in out["claims"])  # both claims kept, with components


def test_equal_trust_conflict_stays_unresolved():
    out = rank_conflicting_claims([{"claim": "a", "authority_level": 2}, {"claim": "b", "authority_level": 2}])
    assert out["status"] == "unresolved" and out["winner"] is None and len(out["claims"]) == 2


def test_superseded_or_expired_source_never_wins():
    out = rank_conflicting_claims([
        {"claim": "old-regulation", "authority_level": 1, "superseded": True, "doc_valid_to": None},
        {"claim": "current-informal", "authority_level": 4, "doc_valid_to": None},
    ])
    assert out["winner"]["claim"] == "current-informal"
    all_stale = rank_conflicting_claims([{"claim": "x", "valid_to": PAST}, {"claim": "y", "valid_to": PAST}])
    assert all_stale["winner"] is None


@pytest.mark.asyncio
async def test_conflict_suggestion_reads_only_the_callers_tenant_and_writes_nothing():
    from graphrag.graph.contradiction_detector import ContradictionDetector

    neo = MagicMock()
    neo.run = AsyncMock(side_effect=[
        [{"sources": "['d1', 'd2']", "status": "open", "src": "A", "tgt": "B", "relation": "R"}],
        [{"claim": "d1", "authority_level": 1}, {"claim": "d2", "authority_level": 4}],
    ])
    out = await ContradictionDetector(neo).suggest_resolution("c1", tenant=T)
    assert out["suggested_winner_doc_id"] == "d1" and len(out["claims"]) == 2
    for call in neo.run.await_args_list:
        assert call.kwargs["tenant"] == T
        assert "SET" not in call.args[0] and "MERGE" not in call.args[0]


# ── graph scoring and explainability ─────────────────────────────────────────

def test_edge_trust_feeds_graph_scoring_and_keeps_components():
    edges = apply_edge_trust([
        {"src": "A", "tgt": "B", "confidence": 0.8, "confidence_state": "DISPUTED"},
        {"src": "A", "tgt": "C", "confidence": 0.8, "confidence_state": "ASSERTED",
         "verification_status": "VERIFIED"},
        {"src": "A", "tgt": "D", "confidence": 0.8, "source_type": "inferred"},
    ])
    disputed, verified, inferred = edges
    assert disputed["confidence"] < inferred["confidence"] < verified["confidence"]
    assert disputed["confidence_before_trust"] == 0.8
    assert disputed["trust"]["factors"]["state"] == 0.5


def test_chunk_trust_ranks_superseded_below_current_and_explains_scores():
    chunks = [
        {"chunk_id": "old", "final_score": 0.9, "vector_score": 0.8, "bm25_score": 7.1, "rrf_score": 0.03,
         "superseded": True, "authority_level": 1},
        {"chunk_id": "new", "final_score": 0.6, "vector_score": 0.7, "superseded": False, "authority_level": 3},
    ]
    assert apply_chunk_trust(chunks, at=NOW) is True
    chunks.sort(key=lambda c: c["final_score"], reverse=True)
    assert [c["chunk_id"] for c in chunks] == ["new", "old"]
    old = next(c for c in chunks if c["chunk_id"] == "old")
    comp = old["score_components"]
    assert comp["vector_score"] == 0.8 and comp["bm25_score"] == 7.1 and comp["rrf_score"] == 0.03
    assert comp["score_before_trust"] == 0.9 and comp["trust_factor_applied"] == 0.5
    assert comp["final_score"] == pytest.approx(0.45)
    assert old["trust"]["superseded"] is True and old["trust"]["current"] is False


def test_chunk_trust_can_be_observed_without_changing_ranking():
    chunks = [{"chunk_id": "old", "final_score": 0.9, "superseded": True}]
    assert apply_chunk_trust(chunks, enabled=False) is False
    assert chunks[0]["final_score"] == 0.9 and chunks[0]["score_components"]["trust_factor_applied"] == 0.5


def test_rrf_keeps_raw_scores_from_every_list():
    from graphrag.retrieval.bm25_search import _reciprocal_rank_fusion

    fused = _reciprocal_rank_fusion(
        [{"chunk_id": "c1", "text": "x", "score": 0.91, "vector_score": 0.91}],
        [{"chunk_id": "c1", "text": "x", "score": 7.5, "bm25_score": 7.5}],
    )
    assert fused[0]["vector_score"] == 0.91 and fused[0]["bm25_score"] == 7.5
    assert fused[0]["score"] == fused[0]["rrf_score"]


def test_citation_evidence_carries_trust_and_components():
    from graphrag.retrieval.context_builder import _trust_fields

    chunk = {"trust": {"valid_from": "2025-01-01", "valid_to": None, "current": True},
             "score_components": {"vector_score": 0.8}}
    out = _trust_fields(chunk)
    assert out["trust"]["current"] is True and out["valid_from"] == "2025-01-01"
    assert out["score_components"] == {"vector_score": 0.8}
    assert _trust_fields({}) == {}


# ── insufficient context when all evidence is stale ──────────────────────────

def test_all_stale_evidence_is_insufficient():
    from graphrag.retrieval.sufficiency import abstention_message, assess_retrieval_sufficiency

    stale = [{"chunk_id": "a", "final_score": 0.9, "trust": {"current": False}},
             {"chunk_id": "b", "final_score": 0.8, "trust": {"current": False}}]
    s = assess_retrieval_sufficiency(chunks=stale, citations=["a"], conflicts=[])
    assert not s.sufficient and s.reason_code == "all_evidence_stale"
    assert "superseded" in abstention_message("all_evidence_stale")
    mixed = stale + [{"chunk_id": "c", "final_score": 0.5, "trust": {"current": True}}]
    assert assess_retrieval_sufficiency(chunks=mixed, citations=["a"], conflicts=[]).sufficient
    # chunks without assessments (e.g. agentic) are not treated as stale
    assert assess_retrieval_sufficiency(chunks=[{"chunk_id": "x"}], citations=["x"], conflicts=[]).sufficient


def test_stale_evidence_abstains_even_when_general_abstention_is_off():
    import inspect

    from graphrag.retrieval import hybrid_retriever
    src = inspect.getsource(hybrid_retriever)
    assert 'sufficiency.reason_code == "all_evidence_stale"' in src
    assert 'cfg.get("retrieval_sufficiency_abstain_on_stale", True)' in src
