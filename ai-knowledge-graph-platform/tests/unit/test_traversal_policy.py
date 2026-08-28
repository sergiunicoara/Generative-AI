from graphrag.retrieval.traversal_policy import build_traversal_policy, select_traversal_candidates


def test_legacy_policy_preserves_configured_limits_when_feature_is_off():
    policy = build_traversal_policy(
        "How are A and B related?", configured_max_hops=2, configured_top_k=50, enabled=False,
    )
    assert policy.max_hops == 2
    assert policy.beam_width == 50


def test_adaptive_policy_expands_only_multi_hop_queries():
    policy = build_traversal_policy(
        "Compare the steps across multiple documents", configured_max_hops=2,
        configured_top_k=50, enabled=True,
    )
    assert policy.query_class == "multi_hop"
    assert policy.max_hops == 4
    assert policy.beam_width == 16


def test_adaptive_policy_skips_negative_graph_expansion():
    policy = build_traversal_policy(
        "Is there any evidence of an exception?", configured_max_hops=2,
        configured_top_k=50, enabled=True,
    )
    assert policy.max_hops == 0
    assert select_traversal_candidates([{"chunk_id": "x", "score": 1.0}], policy) == []


def test_candidate_selector_deduplicates_and_stops_after_low_value_tail():
    policy = build_traversal_policy(
        "How are A and B related?", configured_max_hops=2,
        configured_top_k=50, enabled=True,
    )
    selected = select_traversal_candidates([
        {"chunk_id": "a", "score": 0.9},
        {"chunk_id": "a", "score": 0.8},
        {"chunk_id": "b", "score": 0.5},
        {"chunk_id": "c", "score": 0.01},
    ], policy)
    assert [item["chunk_id"] for item in selected] == ["a", "b"]
