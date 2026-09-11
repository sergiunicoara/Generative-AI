"""Unit tests for graphrag.evaluation.retrieval_quality_eval.

No live services, no Docker, no LLM calls -- `retrieve` is a fake async
callable returning fixed ranked lists, proving the aggregation and
threshold-enforcement logic against known numbers.
"""
from __future__ import annotations

import pytest

from graphrag.evaluation.retrieval_quality_eval import evaluate_retrieval_quality

THRESHOLDS = {"min_context_precision": 0.5, "min_citation_recall": 0.5}

QUESTIONS = [
    {
        "id": "Q1",
        "question": "well-ranked question",
        "expected_citations": ["Doc-A"],
    },
    {
        "id": "Q2",
        "question": "poorly-ranked question",
        "expected_citations": ["Doc-B"],
    },
    {
        "id": "Q3",
        "question": "unanswerable meta-question",
        "expected_citations": [],
    },
]

# Q1: relevant doc at rank 1 of 2 -> precision@2=0.5, recall@2=1.0, MRR=1.0
# Q2: relevant doc missing entirely -> precision@2=0.0, recall@2=0.0, MRR=0.0
_RANKED_BY_QUESTION = {
    "well-ranked question": ["Doc-A", "Doc-Z"],
    "poorly-ranked question": ["Doc-X", "Doc-Y"],
}


async def _fake_retrieve(question: str) -> list[str]:
    return _RANKED_BY_QUESTION[question]


class TestEvaluateRetrievalQuality:
    async def test_scores_only_questions_with_expected_citations(self):
        report = await evaluate_retrieval_quality(QUESTIONS, _fake_retrieve, THRESHOLDS, k=2)
        assert {s.id for s in report.scored} == {"Q1", "Q2"}
        assert report.skipped_ids == ["Q3"]

    async def test_per_question_metrics_are_exact(self):
        report = await evaluate_retrieval_quality(QUESTIONS, _fake_retrieve, THRESHOLDS, k=2)
        by_id = {s.id: s for s in report.scored}

        assert by_id["Q1"].precision_at_k == 0.5
        assert by_id["Q1"].recall_at_k == 1.0
        assert by_id["Q1"].reciprocal_rank == 1.0

        assert by_id["Q2"].precision_at_k == 0.0
        assert by_id["Q2"].recall_at_k == 0.0
        assert by_id["Q2"].reciprocal_rank == 0.0

    async def test_aggregate_means_average_across_scored_questions_only(self):
        report = await evaluate_retrieval_quality(QUESTIONS, _fake_retrieve, THRESHOLDS, k=2)
        # mean over Q1 and Q2 only -- Q3 (no expected_citations) is excluded,
        # not averaged in as a vacuous 1.0 recall.
        assert report.mean_precision == pytest.approx((0.5 + 0.0) / 2)
        assert report.mean_recall == pytest.approx((1.0 + 0.0) / 2)

    async def test_below_threshold_mean_fails_and_names_the_bad_question(self):
        report = await evaluate_retrieval_quality(QUESTIONS, _fake_retrieve, THRESHOLDS, k=2)
        # mean_recall = 0.5 meets the 0.5 threshold exactly, but mean_precision
        # = 0.25 < 0.5 -- the aggregate must fail.
        assert report.passed is False
        assert "Q2" in report.failing_questions
        assert "Q1" not in report.failing_questions

    async def test_all_questions_passing_thresholds_marks_report_passed(self):
        easy_thresholds = {"min_context_precision": 0.0, "min_citation_recall": 0.0}
        report = await evaluate_retrieval_quality(QUESTIONS, _fake_retrieve, easy_thresholds, k=2)
        assert report.passed is True
        assert report.failing_questions == []

    async def test_no_scoreable_questions_never_passes(self):
        all_unanswerable = [{"id": "Q3", "question": "x", "expected_citations": []}]
        report = await evaluate_retrieval_quality(all_unanswerable, _fake_retrieve, THRESHOLDS, k=2)
        assert report.scored == []
        assert report.passed is False
