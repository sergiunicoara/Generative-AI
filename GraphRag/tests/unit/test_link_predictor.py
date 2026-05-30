"""Tests for graphrag.graph.link_predictor.LinkPredictor."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag.graph.link_predictor import (
    LinkPredictor,
    _l2_norm,
    _translate,
    _transe_similarity,
)


# ── Vector math unit tests ─────────────────────────────────────────────────────

class TestVectorMath:
    def test_l2_norm_unit_vector(self) -> None:
        v = [3.0, 4.0]
        n = _l2_norm(v)
        assert abs(math.sqrt(sum(x * x for x in n)) - 1.0) < 1e-9

    def test_l2_norm_zero_vector_no_division_error(self) -> None:
        n = _l2_norm([0.0, 0.0, 0.0])
        # Should not raise ZeroDivisionError; result is defined as all-zeros / 1.0
        assert all(x == 0.0 for x in n)

    def test_translate_returns_unit_vector(self) -> None:
        h = [1.0, 0.0]
        r = [0.0, 1.0]
        t = _translate(h, r)
        assert abs(math.sqrt(sum(x * x for x in t)) - 1.0) < 1e-9

    def test_transe_similarity_identical_gives_high_score(self) -> None:
        h = [1.0, 0.0]
        r = [0.0, 0.0]
        t = [1.0, 0.0]   # h + r == t → distance 0
        score = _transe_similarity(h, r, t)
        assert score > 0.99  # 1 / (1 + ~0) ≈ 1

    def test_transe_similarity_opposite_gives_low_score(self) -> None:
        h = [1.0, 0.0]
        r = [0.0, 0.0]
        t = [-1.0, 0.0]  # large distance
        score = _transe_similarity(h, r, t)
        assert score < 0.5


# ── Mock factories ─────────────────────────────────────────────────────────────

def _make_neo4j(entity_emb: list[float], ann_results: list[dict] | None = None):
    """Return a mock Neo4j client."""
    neo4j = AsyncMock()
    entity_row = [{"emb": entity_emb}] if entity_emb else []
    ann_rows   = ann_results or []
    neo4j.run = AsyncMock(side_effect=[entity_row, ann_rows])
    return neo4j


def _make_trainer(rel_emb: dict[str, list[float]]):
    t = MagicMock()
    t._rel_emb = rel_emb
    return t


# ── predict_tail tests ─────────────────────────────────────────────────────────

class TestPredictTail:
    @pytest.mark.asyncio
    async def test_returns_sorted_by_score_desc(self) -> None:
        head_emb = [1.0, 0.0, 0.0]
        rel_emb  = [0.0, 0.0, 0.0]
        ann_rows = [
            {"entity_id": "e1", "name": "E1", "type": "CONCEPT",
             "embedding": [1.0, 0.0, 0.0], "score": 0.9},
            {"entity_id": "e2", "name": "E2", "type": "CONCEPT",
             "embedding": [-1.0, 0.0, 0.0], "score": 0.1},
        ]
        neo4j   = _make_neo4j(head_emb, ann_rows)
        trainer = _make_trainer({"SUPERSEDES": rel_emb})
        pred    = LinkPredictor(neo4j, trainer)

        results = await pred.predict_tail("entity-1", "SUPERSEDES", top_k=5)
        assert len(results) == 2
        assert results[0]["score"] >= results[1]["score"]

    @pytest.mark.asyncio
    async def test_top_k_respected(self) -> None:
        head_emb = [1.0, 0.0]
        rel_emb  = [0.0, 0.0]
        ann_rows = [{"entity_id": f"e{i}", "name": f"E{i}", "type": "CONCEPT",
                     "embedding": [1.0, 0.0], "score": 0.5}
                    for i in range(10)]
        neo4j   = _make_neo4j(head_emb, ann_rows)
        trainer = _make_trainer({"REL": rel_emb})
        pred    = LinkPredictor(neo4j, trainer)

        results = await pred.predict_tail("e0", "REL", top_k=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_missing_entity_raises_value_error(self) -> None:
        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[])  # no entity found
        pred  = LinkPredictor(neo4j, _make_trainer({"REL": [1.0]}))

        with pytest.raises(ValueError, match="Entity not found"):
            await pred.predict_tail("nonexistent-id", "REL")

    @pytest.mark.asyncio
    async def test_no_rel_emb_falls_back_to_cosine(self) -> None:
        head_emb = [1.0, 0.0]
        ann_rows = [{"entity_id": "e1", "name": "E1", "type": "T",
                     "embedding": [1.0, 0.0], "score": 0.8}]
        neo4j   = _make_neo4j(head_emb, ann_rows)
        trainer = _make_trainer({})  # no trained embeddings
        pred    = LinkPredictor(neo4j, trainer)

        results = await pred.predict_tail("head-id", "UNKNOWN_REL")
        # Should return results without raising
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_tenant_passed_to_neo4j(self) -> None:
        head_emb = [1.0, 0.0]
        neo4j    = AsyncMock()
        neo4j.run = AsyncMock(side_effect=[
            [{"emb": head_emb}],
            [],  # ANN returns empty
        ])
        pred = LinkPredictor(neo4j, _make_trainer({"REL": [0.0, 0.0]}))
        await pred.predict_tail("head-id", "REL", tenant="aerospace")

        # First call is entity embedding lookup — tenant must be in kwargs
        first_call_kwargs = neo4j.run.call_args_list[0].kwargs
        assert first_call_kwargs.get("tenant") == "aerospace"

    @pytest.mark.asyncio
    async def test_empty_ann_results_returns_empty_list(self) -> None:
        neo4j = _make_neo4j([1.0, 0.0], [])
        pred  = LinkPredictor(neo4j, _make_trainer({"REL": [0.0, 0.0]}))
        results = await pred.predict_tail("head-id", "REL")
        assert results == []


# ── predict_relation tests ─────────────────────────────────────────────────────

class TestPredictRelation:
    @pytest.mark.asyncio
    async def test_returns_sorted_by_score(self) -> None:
        head_emb = [1.0, 0.0]
        tail_emb = [1.0, 0.0]

        neo4j = AsyncMock()
        neo4j.run = AsyncMock(side_effect=[
            [{"emb": head_emb}],
            [{"emb": tail_emb}],
        ])
        rel_emb = {
            "REL_A": [0.0, 0.0],   # h + r == t → high score
            "REL_B": [5.0, 5.0],   # large distance → low score
        }
        pred    = LinkPredictor(neo4j, _make_trainer(rel_emb))
        results = await pred.predict_relation("h", "t")

        assert len(results) == 2
        assert results[0]["score"] >= results[1]["score"]

    @pytest.mark.asyncio
    async def test_no_trained_relations_returns_empty(self) -> None:
        neo4j = AsyncMock()
        neo4j.run = AsyncMock(side_effect=[
            [{"emb": [1.0, 0.0]}],
            [{"emb": [1.0, 0.0]}],
        ])
        pred    = LinkPredictor(neo4j, _make_trainer({}))
        results = await pred.predict_relation("h", "t")
        assert results == []


# ── find_missing_links tests ───────────────────────────────────────────────────

class TestFindMissingLinks:
    @pytest.mark.asyncio
    async def test_empty_entity_list_returns_empty(self) -> None:
        pred = LinkPredictor(AsyncMock(), _make_trainer({}))
        assert await pred.find_missing_links([]) == []

    @pytest.mark.asyncio
    async def test_filters_by_threshold(self) -> None:
        head_emb = [1.0, 0.0]
        ann_low  = [{"entity_id": "t1", "name": "T1", "type": "T",
                     "embedding": [-1.0, 0.0], "score": 0.1}]
        neo4j = AsyncMock()
        neo4j.run = AsyncMock(side_effect=[
            [],          # existing edges query → none
            [{"emb": head_emb}],   # entity emb for head
            ann_low,               # ANN results → low score candidate
        ])
        pred    = LinkPredictor(neo4j, _make_trainer({"REL": [5.0, 5.0]}))
        results = await pred.find_missing_links(["head-1"], threshold=0.9)
        assert results == []   # low score filtered out
