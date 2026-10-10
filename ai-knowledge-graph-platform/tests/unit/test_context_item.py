"""Phase 8 (optional): shared ContextItem contract with adapters."""
from __future__ import annotations

import pytest

from graphrag.retrieval.context_item import (
    ContextItem,
    context_items_from_results,
    from_chunk,
    from_community,
    from_episode,
    from_graph_edge,
    from_structured_row,
    fuse,
)

T = "acme"
CHUNK = {"chunk_id": "c1", "text": "600 flight hours", "document_id": "d1", "_doc_name": "AD-1",
         "authority_level": 1, "doc_valid_from": "2024-01-01T00:00:00+00:00",
         "doc_valid_to": "2026-01-01T00:00:00+00:00", "retrieval": "vector+bm25",
         "trust": {"origin": "EXTRACTED", "trust_factor": 0.9}, "score_components": {"vector_score": 0.8}}
EDGE = {"src": "Acme", "src_type": "ORG", "relation": "OWNS", "tgt": "Bolt", "tgt_type": "ORG",
        "confidence": 0.7, "source_doc_id": "d1", "valid_from": "2024-03-01T00:00:00+00:00",
        "trust": {"origin": "INFERRED"}, "inferred_by": "r", "premise_keys": ["p"]}


def test_every_surface_maps_to_the_contract_without_losing_provenance():
    c = from_chunk(CHUNK, tenant=T, scope=["tenant"])
    assert c.identity == "chunk:c1" and c.context_type == "document_chunk"
    assert c.provenance[0]["document_id"] == "d1" and c.provenance[0]["authority_level"] == 1
    assert c.valid_to.year == 2026 and c.confidence == 0.9 and c.scope == ["tenant"]
    f = from_graph_edge(EDGE, tenant=T)
    assert f.identity == "fact:ORG:Acme|OWNS|ORG:Bolt" and f.provenance[0]["premises"] == ["p"]
    assert from_community({"id": "k1", "summary": "s", "score": 0.4}, tenant=T).context_type == "community_summary"
    assert from_structured_row({"source_doc_id": "d1"}, intent="x", tenant=T, index=0).identity == "row:x:0"
    assert from_episode({"id": "e1", "summary": "m"}, tenant=T).provenance[0]["origin"] == "GENERATED"


def test_fusion_keeps_identity_and_merges_provenance_instead_of_replacing():
    a = from_chunk(CHUNK, tenant=T, scope=["tenant"])
    b = from_chunk({**CHUNK, "retrieval": "graph", "document_id": "d1", "_doc_name": "AD-1",
                    "doc_valid_to": "2025-06-01T00:00:00+00:00", "trust": {"trust_factor": 0.95}},
                   tenant=T, scope=["group:eng"])
    [m] = fuse([a], [b])
    assert m.identity == "chunk:c1" and len(m.provenance) == 2
    assert m.scope == ["group:eng", "tenant"] and m.confidence == 0.95
    assert m.valid_to.year == 2025  # narrowest window: never more current than a source says


def test_fusion_refuses_to_mix_tenants():
    with pytest.raises(ValueError):
        fuse([from_chunk(CHUNK, tenant="a")], [from_chunk(CHUNK, tenant="b")])


def test_different_surfaces_stay_distinct_items():
    items = fuse([from_chunk(CHUNK, tenant=T)], [from_graph_edge(EDGE, tenant=T)])
    assert {i.context_type for i in items} == {"document_chunk", "graph_fact"}


def test_query_adapter_omits_graph_facts_under_acl():
    local = {"chunks": [CHUNK], "entity_edges": [EDGE]}
    assert len(context_items_from_results(local, None, tenant=T, acl_enforced=False)) == 2
    only = context_items_from_results(local, None, tenant=T, acl_enforced=True)
    assert [i.context_type for i in only] == ["document_chunk"]


def test_confidence_is_bounded():
    with pytest.raises(ValueError):
        ContextItem(identity="x", content="", context_type="memory", source="s", tenant=T, confidence=1.5)
