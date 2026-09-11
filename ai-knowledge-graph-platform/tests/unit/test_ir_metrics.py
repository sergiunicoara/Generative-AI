"""Unit tests for graphrag.evaluation.ir_metrics against hand-computed values."""
from __future__ import annotations

import math

from graphrag.evaluation.ir_metrics import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)

# Fixture: 5-item ranked list, relevant ids at positions 2 and 4 (1-indexed).
RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = {"b", "d"}


class TestPrecisionAtK:
    def test_top3_has_one_of_two_relevant(self):
        # top-3 = [a, b, c] -> 1 relevant ("b") / 3 = 0.333...
        assert precision_at_k(RANKED, RELEVANT, 3) == 1 / 3

    def test_top4_has_both_relevant(self):
        # top-4 = [a, b, c, d] -> 2 relevant / 4 = 0.5
        assert precision_at_k(RANKED, RELEVANT, 4) == 0.5

    def test_empty_ranked_is_zero(self):
        assert precision_at_k([], RELEVANT, 5) == 0.0

    def test_k_zero_is_zero(self):
        assert precision_at_k(RANKED, RELEVANT, 0) == 0.0

    def test_k_larger_than_list_uses_whole_list(self):
        assert precision_at_k(RANKED, RELEVANT, 100) == 2 / 5


class TestRecallAtK:
    def test_top3_finds_one_of_two_relevant(self):
        assert recall_at_k(RANKED, RELEVANT, 3) == 0.5

    def test_top4_finds_both_relevant(self):
        assert recall_at_k(RANKED, RELEVANT, 4) == 1.0

    def test_empty_relevant_is_vacuously_perfect(self):
        assert recall_at_k(RANKED, set(), 3) == 1.0

    def test_empty_ranked_with_nonempty_relevant_is_zero(self):
        assert recall_at_k([], RELEVANT, 5) == 0.0


class TestReciprocalRank:
    def test_first_relevant_at_position_2(self):
        assert reciprocal_rank(RANKED, RELEVANT) == 0.5

    def test_first_relevant_at_position_1(self):
        assert reciprocal_rank(RANKED, {"a"}) == 1.0

    def test_no_relevant_found_is_zero(self):
        assert reciprocal_rank(RANKED, {"z"}) == 0.0

    def test_empty_ranked_is_zero(self):
        assert reciprocal_rank([], RELEVANT) == 0.0


class TestAveragePrecision:
    def test_two_relevant_at_positions_2_and_4(self):
        # AP = (precision@2 + precision@4) / |relevant|
        #    = (1/2 + 2/4) / 2 = (0.5 + 0.5) / 2 = 0.5
        assert average_precision(RANKED, RELEVANT) == 0.5

    def test_relevant_ids_all_at_the_top(self):
        # relevant = {a, b}: AP = (1/1 + 2/2) / 2 = 1.0
        assert average_precision(RANKED, {"a", "b"}) == 1.0

    def test_empty_relevant_is_zero(self):
        assert average_precision(RANKED, set()) == 0.0

    def test_no_hits_is_zero(self):
        assert average_precision(RANKED, {"z"}) == 0.0


class TestNdcgAtK:
    def test_binary_relevance_matches_manual_computation(self):
        relevance = {"b": 1.0, "d": 1.0}
        # DCG@4 = 1/log2(3) [rank2] + 1/log2(5) [rank4]
        # Ideal order for k=4 puts both relevant ids first: ranks 1 and 2
        # (only 2 positive-relevance ids exist, so the ideal top-4 is just
        # those two, and the log2(i+1) terms beyond rank 2 don't apply).
        expected_dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
        expected_ideal = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        assert ndcg_at_k(RANKED, relevance, 4) == expected_dcg / expected_ideal

    def test_perfect_ranking_is_1(self):
        relevance = {"a": 1.0, "b": 1.0}
        assert ndcg_at_k(["a", "b", "c"], relevance, 3) == 1.0

    def test_no_positive_relevance_is_zero(self):
        assert ndcg_at_k(RANKED, {}, 3) == 0.0

    def test_empty_ranked_is_zero(self):
        assert ndcg_at_k([], {"a": 1.0}, 3) == 0.0
