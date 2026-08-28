from graphrag.retrieval.evidence_bundle import build_evidence_bundle
from graphrag.retrieval.evidence_fusion import apply_evidence_fusion, fusion_weights


def test_relational_weights_favour_graph_and_path_more_than_factoids():
    relational = fusion_weights("How are A and B connected?")
    factoid = fusion_weights("What is A?")
    assert relational.graph + relational.path > factoid.graph + factoid.path


def test_fusion_reranks_a_verified_path_over_weak_text_match():
    chunks = [
        {"chunk_id": "text", "text_score": 0.9, "gnn_score": 0.0, "path_confidence": 0.0},
        {"chunk_id": "path", "text_score": 0.4, "gnn_score": 0.9, "path_confidence": 1.0},
    ]
    result = apply_evidence_fusion(chunks, fusion_weights("How are A and B connected?"))
    assert result[0]["chunk_id"] == "path"
    assert result[0]["fusion_components"]["path"] == 1.0


def test_evidence_bundle_keeps_only_retrieved_identifiers_and_time_bounds():
    bundle = build_evidence_bundle(
        local_results={
            "chunks": [{"chunk_id": "c1", "_doc_name": "doc-a", "path_confidence": 0.8}],
            "referenced_entities": ["A", "A"],
            "entity_edges": [{"src": "A", "relation": "RELATES_TO", "tgt": "B"}],
        },
        global_results={"communities": [{"community_id": "community-1"}]},
        citations=["doc-a", "doc-a"], valid_at="2026-01-01", transaction_at="2026-01-02",
    )
    assert bundle.chunk_ids == ["c1"]
    assert bundle.entity_ids == ["A"]
    assert bundle.path_count == 1
    assert bundle.valid_at == "2026-01-01"
