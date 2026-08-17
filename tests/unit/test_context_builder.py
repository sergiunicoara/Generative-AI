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
