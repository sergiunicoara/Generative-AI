"""Unit tests for ContextBuilder.build() chunk ranking and assembly."""

from __future__ import annotations

from graphrag.retrieval.context_builder import ContextBuilder


def _local(
    chunks: list[dict],
    entities: list[dict] | None = None,
    entity_edges: list[dict] | None = None,
) -> dict:
    return {"chunks": chunks, "entities": entities or [], "entity_edges": entity_edges or []}


class TestContextBuilderRanking:
    """build() must rank chunks by final_score (GNN blend), not the raw RRF/path score."""

    def test_ranks_by_final_score_over_raw_score(self):
        """A seed chunk with high final_score but low raw 'score' (RRF) must
        outrank a multi-hop chunk with high raw 'score' but no final_score."""
        chunks = [
            {"chunk_id": "seed", "text": "seed text", "score": 0.03, "rerank_score": 8.0, "final_score": 0.97},
            {"chunk_id": "hop", "text": "hop text", "score": 0.9},
        ]
        _, citations = ContextBuilder().build(_local(chunks), {}, top_k=1)
        assert citations == ["seed"]

    def test_falls_back_to_rerank_score_when_no_final_score(self):
        chunks = [
            {"chunk_id": "a", "text": "a text", "rerank_score": 2.0},
            {"chunk_id": "b", "text": "b text", "rerank_score": 5.0},
        ]
        _, citations = ContextBuilder().build(_local(chunks), {}, top_k=2)
        assert citations == ["b", "a"]

    def test_falls_back_to_score_when_no_rerank_or_final(self):
        chunks = [
            {"chunk_id": "a", "text": "a text", "score": 0.2},
            {"chunk_id": "b", "text": "b text", "score": 0.8},
        ]
        _, citations = ContextBuilder().build(_local(chunks), {}, top_k=2)
        assert citations == ["b", "a"]

    def test_top_k_limits_chunks_and_citations(self):
        chunks = [
            {"chunk_id": f"c{i}", "text": f"text {i}", "final_score": float(i)}
            for i in range(5)
        ]
        context, citations = ContextBuilder().build(_local(chunks), {}, top_k=2)
        assert citations == ["c4", "c3"]
        assert "[Chunk c4]" in context
        assert "[Chunk c2]" not in context

    def test_deduplicates_citations_preserving_order(self):
        chunks = [
            {"chunk_id": "a", "text": "a text", "final_score": 1.0},
            {"chunk_id": "a", "text": "a text", "final_score": 1.0},
        ]
        _, citations = ContextBuilder().build(_local(chunks), {}, top_k=5)
        assert citations == ["a"]


class TestContextBuilderNearDuplicates:
    """A clause repeated verbatim across chunk_ids must not fill every top_k
    slot — distinct chunks from other documents should get the freed slots."""

    def test_near_duplicate_text_does_not_crowd_out_distinct_chunk(self):
        repeated = (
            "Furnizorii clasificati ca CRITICI sunt supusi reevaluarii "
            "SEMESTRIALE conform politicii de calitate a companiei."
        )
        chunks = [
            {"chunk_id": "csr-1", "text": repeated, "final_score": 0.95},
            {"chunk_id": "csr-2", "text": repeated, "final_score": 0.94},
            {"chunk_id": "csr-3", "text": repeated, "final_score": 0.93},
            {"chunk_id": "rfa-1", "text": "Furnizorii CRITICI: reevaluare SEMESTRIALA conform RFA-REG-01.", "final_score": 0.50},
        ]
        _, citations = ContextBuilder().build(_local(chunks), {}, top_k=2)
        assert citations == ["csr-1", "rfa-1"]

    def test_distinct_chunks_all_kept(self):
        chunks = [
            {"chunk_id": "a", "text": "Alpha section about brakes.", "final_score": 0.9},
            {"chunk_id": "b", "text": "Beta section about tires.", "final_score": 0.8},
            {"chunk_id": "c", "text": "Gamma section about engines.", "final_score": 0.7},
        ]
        _, citations = ContextBuilder().build(_local(chunks), {}, top_k=3)
        assert citations == ["a", "b", "c"]


class TestContextBuilderConflicts:
    """An open, unresolved Conflict on a retrieved entity must be surfaced to
    the LLM explicitly — see ContradictionDetector.get_open_conflicts_for_entities."""

    def test_no_conflicts_section_when_none_given(self):
        chunks = [{"chunk_id": "a", "text": "text", "final_score": 1.0}]
        context, _ = ContextBuilder().build(_local(chunks), {}, top_k=1)
        assert "Unresolved conflicts" not in context

    def test_no_conflicts_section_when_empty_list(self):
        chunks = [{"chunk_id": "a", "text": "text", "final_score": 1.0}]
        context, _ = ContextBuilder().build(_local(chunks), {}, top_k=1, conflicts=[])
        assert "Unresolved conflicts" not in context

    def test_conflicts_section_rendered_when_present(self):
        chunks = [{"chunk_id": "a", "text": "text", "final_score": 1.0}]
        conflicts = [{
            "src": "Apple", "tgt": "Orange",
            "relation": "COMPETES_WITH", "conflict_type": "exclusive_state",
        }]
        context, _ = ContextBuilder().build(_local(chunks), {}, top_k=1, conflicts=conflicts)
        assert "Unresolved conflicts" in context
        assert "Apple" in context
        assert "Orange" in context
        assert "exclusive_state" in context

    def test_conflicts_capped_at_five(self):
        chunks = [{"chunk_id": "a", "text": "text", "final_score": 1.0}]
        conflicts = [
            {"src": f"E{i}", "tgt": f"F{i}", "relation": "REL", "conflict_type": "exclusive_state"}
            for i in range(8)
        ]
        context, _ = ContextBuilder().build(_local(chunks), {}, top_k=1, conflicts=conflicts)
        for i in range(5):
            assert f"E{i}" in context
        for i in range(5, 8):
            assert f"E{i}" not in context


class TestContextBuilderGraphRelationships:
    """A graph edge (asserted or transitively inferred) between retrieved
    entities must be surfaced as an explicit fact — chunk text alone often
    states only pairwise facts, never the transitive one. See
    INF-01/CON-02 in evals/golden_set.json."""

    def test_no_section_when_no_edges(self):
        chunks = [{"chunk_id": "a", "text": "text", "final_score": 1.0}]
        context, _ = ContextBuilder().build(_local(chunks), {}, top_k=1)
        assert "Known graph relationships" not in context

    def test_asserted_edge_rendered_without_inferred_label(self):
        chunks = [{"chunk_id": "a", "text": "text", "final_score": 1.0}]
        edges = [{"src": "FAA-AD-2024-01-02", "tgt": "FAA-AD-2022-03-07", "relation": "SUPERSEDES"}]
        context, _ = ContextBuilder().build(_local(chunks, entity_edges=edges), {}, top_k=1)
        assert "Known graph relationships" in context
        assert "FAA-AD-2024-01-02 —SUPERSEDES→ FAA-AD-2022-03-07" in context
        assert "(inferred" not in context

    def test_inferred_edge_labeled_with_rule(self):
        chunks = [{"chunk_id": "a", "text": "text", "final_score": 1.0}]
        edges = [{
            "src": "FAA-AD-2024-01-02", "tgt": "FAA-AD-2020-05-11", "relation": "SUPERSEDES",
            "source_type": "inferred", "inferred_by": "supersedes_transitivity",
        }]
        context, _ = ContextBuilder().build(_local(chunks, entity_edges=edges), {}, top_k=1)
        assert "FAA-AD-2024-01-02 —SUPERSEDES→ FAA-AD-2020-05-11" in context
        assert "(inferred via supersedes_transitivity)" in context

    def test_edge_without_relation_label_skipped(self):
        """No relation string means nothing informative to report — must not
        render a blank/garbled line."""
        chunks = [{"chunk_id": "a", "text": "text", "final_score": 1.0}]
        edges = [{"src": "A", "tgt": "B", "relation": None}]
        context, _ = ContextBuilder().build(_local(chunks, entity_edges=edges), {}, top_k=1)
        assert "Known graph relationships" not in context

    def test_edges_capped_at_ten(self):
        chunks = [{"chunk_id": "a", "text": "text", "final_score": 1.0}]
        edges = [
            {"src": f"E{i}", "tgt": f"F{i}", "relation": "REL"}
            for i in range(15)
        ]
        context, _ = ContextBuilder().build(_local(chunks, entity_edges=edges), {}, top_k=1)
        for i in range(10):
            assert f"E{i}" in context
        for i in range(10, 15):
            assert f"E{i}" not in context

    def test_edge_endpoints_registered_as_citations(self):
        """A fact surfaced only via a graph edge (e.g. a transitive
        supersession chain no single chunk states outright) must still put
        its endpoints in the citations list — otherwise a correct,
        graph-grounded answer reports insufficient citation recall. See
        INF-01 in evals/golden_set.json."""
        chunks = [{"chunk_id": "a", "text": "text", "final_score": 1.0, "source": "FAA-AD-2024-01-02.txt"}]
        edges = [{
            "src": "FAA-AD-2024-01-02", "tgt": "FAA-AD-2020-05-11", "relation": "SUPERSEDES",
            "source_type": "inferred", "inferred_by": "supersedes_transitivity",
        }]
        _, citations = ContextBuilder().build(_local(chunks, entity_edges=edges), {}, top_k=1)
        assert "FAA-AD-2020-05-11" in citations

    def test_edge_endpoint_citations_deduped_against_chunk_citations(self):
        chunks = [{"chunk_id": "a", "text": "text", "final_score": 1.0, "source": "FAA-AD-2024-01-02.txt"}]
        edges = [{"src": "FAA-AD-2024-01-02", "tgt": "FAA-AD-2022-03-07", "relation": "SUPERSEDES"}]
        _, citations = ContextBuilder().build(_local(chunks, entity_edges=edges), {}, top_k=1)
        assert citations.count("FAA-AD-2024-01-02") == 1

    def test_edge_without_relation_label_not_cited(self):
        chunks = [{"chunk_id": "a", "text": "text", "final_score": 1.0}]
        edges = [{"src": "A", "tgt": "B", "relation": None}]
        _, citations = ContextBuilder().build(_local(chunks, entity_edges=edges), {}, top_k=1)
        assert "A" not in citations
        assert "B" not in citations


class TestContextBuilderHopReservedSlots:
    """hop_reserved_slots (Change B, MH-03 design proposal, 2026-08-17):
    additive-only extra slots for hop-only chunks (no rerank_score) that
    GNNScorer's seed-floor dampening always ranks below the weakest seed —
    see gnn_scorer.py's _text_score. Default 0 must be a byte-for-byte no-op
    vs. the pre-2026-08-17 signature; when > 0 it must never evict a chunk
    the ordinary top_k slice already selected."""

    def test_default_zero_is_a_no_op(self):
        """Same call, only the presence of a qualifying hop chunk differs —
        with hop_reserved_slots at its default (0), output must be
        identical either way."""
        seed = {"chunk_id": "seed", "text": "seed text", "final_score": 1.0, "rerank_score": 0.9}
        hop = {"chunk_id": "hop", "text": "hop text", "final_score": 0.05, "gnn_score": 0.9}
        ctx_without_hop, cit_without_hop = ContextBuilder().build(
            _local([seed]), {}, top_k=1,
        )
        ctx_with_hop, cit_with_hop = ContextBuilder().build(
            _local([seed, hop]), {}, top_k=1,
        )
        assert ctx_without_hop == ctx_with_hop
        assert cit_without_hop == cit_with_hop
        assert "hop text" not in ctx_with_hop

    def test_reserved_slot_admits_qualifying_hop_chunk(self):
        seed = {"chunk_id": "seed", "text": "seed text", "final_score": 1.0, "rerank_score": 0.9}
        hop = {"chunk_id": "hop", "text": "hop text", "final_score": 0.05, "gnn_score": 0.9}
        context, citations = ContextBuilder().build(
            _local([seed, hop]), {}, top_k=1, hop_reserved_slots=1,
        )
        assert "hop text" in context
        assert "seed text" in context   # additive -- the seed slot is untouched

    def test_reserved_slot_never_evicts_an_existing_seed(self):
        """Two seeds already fill top_k=2; a hop chunk must be ADDED as a
        6th-ish slot, never swap out either seed."""
        seed_a = {"chunk_id": "sa", "text": "seed A", "final_score": 1.0, "rerank_score": 0.9}
        seed_b = {"chunk_id": "sb", "text": "seed B", "final_score": 0.8, "rerank_score": 0.7}
        hop = {"chunk_id": "hop", "text": "hop text", "final_score": 0.05, "gnn_score": 0.9}
        context, _ = ContextBuilder().build(
            _local([seed_a, seed_b, hop]), {}, top_k=2, hop_reserved_slots=1,
        )
        assert "seed A" in context
        assert "seed B" in context
        assert "hop text" in context

    def test_hop_chunk_below_min_gnn_floor_not_admitted(self):
        seed = {"chunk_id": "seed", "text": "seed text", "final_score": 1.0, "rerank_score": 0.9}
        weak_hop = {"chunk_id": "hop", "text": "weak hop text", "final_score": 0.02, "gnn_score": 0.1}
        context, _ = ContextBuilder().build(
            _local([seed, weak_hop]), {}, top_k=1,
            hop_reserved_slots=1, hop_reserved_min_gnn=0.3,
        )
        assert "weak hop text" not in context

    def test_seed_chunk_never_double_counted_as_a_hop_slot(self):
        """A chunk that already made the top_k cut (has rerank_score) must
        not also be pulled in as a "reserved hop slot" duplicate."""
        seed = {"chunk_id": "seed", "text": "seed text", "final_score": 1.0, "rerank_score": 0.9}
        context, citations = ContextBuilder().build(
            _local([seed]), {}, top_k=1, hop_reserved_slots=2,
        )
        assert context.count("seed text") == 1
        assert citations.count("seed") <= 1

    def test_reserved_slots_capped_even_with_many_qualifying_hops(self):
        seed = {"chunk_id": "seed", "text": "seed text", "final_score": 1.0, "rerank_score": 0.9}
        # Genuinely distinct passages -- near-identical text (e.g. "hop text
        # 0" vs "hop text 1") would correctly trip the existing
        # near-duplicate filter and undercount this test's own assertion.
        passages = [
            "Southwest Airlines operates the affected 737 MAX fleet.",
            "The maintenance crew inspected the AOA sensor bracket.",
            "Boeing issued a service bulletin for the MCAS software.",
            "The FAA emergency order grounded the aircraft type.",
            "CFM International manufactures the LEAP-1B engine.",
        ]
        hops = [
            {"chunk_id": f"hop{i}", "text": passages[i], "final_score": 0.05 - i * 0.001,
             "gnn_score": 0.9}
            for i in range(5)
        ]
        context, _ = ContextBuilder().build(
            _local([seed, *hops]), {}, top_k=1, hop_reserved_slots=2,
        )
        admitted = sum(1 for p in passages if p in context)
        assert admitted == 2
