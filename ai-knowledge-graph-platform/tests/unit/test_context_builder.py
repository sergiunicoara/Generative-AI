"""Unit tests for ContextBuilder.build() chunk ranking and assembly."""

from __future__ import annotations

from graphrag.retrieval.context_builder import ContextBuilder


def _local(chunks: list[dict], entities: list[dict] | None = None) -> dict:
    return {"chunks": chunks, "entities": entities or []}


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
