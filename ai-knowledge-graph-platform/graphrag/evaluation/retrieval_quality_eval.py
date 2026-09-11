"""Score real retrieval rankings against `evals/golden_set.json` and enforce
the thresholds it declares.

`evals/golden_set.json` has carried a ``thresholds`` block
(``min_context_precision``, ``min_faithfulness``, ``min_relevancy``,
``min_citation_recall``, ``pass_rate_min``) since it was introduced, but
``scripts/run_golden_eval.py`` -- the script the eval gate documented in
``docs/CONTRIBUTING.md`` actually runs -- only ever enforces ``pass_rate_min``
plus its own per-question term/citation presence checks. The other three
thresholds are declared and never read anywhere else in the codebase except
by the separate, non-gated ``run_faithfulness_eval.py`` (LLM-judge RAGAS
scores). This module closes that gap for ``min_context_precision`` and
``min_citation_recall`` specifically, using classical IR metrics
(`graphrag.evaluation.ir_metrics`) computed against each question's existing
``expected_citations`` field -- no schema change, no new LLM calls.

Faithfulness/relevancy (LLM-judge metrics) are intentionally out of scope
here; they are already real and already wired via
`graphrag/evaluation/ragas_evaluator.py`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from graphrag.evaluation import ir_metrics
from graphrag.graph.alias_registry import canonical_document_key

# Ranked results are scored against the top-N ids used for precision/recall
# `@k`. golden_set.json questions don't carry an explicit k, and /search's
# own default top_k is 10 (api/routes/search.py: SearchRequest.top_k default)
# -- reused here so this eval measures the same depth a real caller sees.
DEFAULT_K = 10

RetrieveFn = Callable[[str], Awaitable[list[str]]]


@dataclass
class QuestionScore:
    id: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    average_precision: float


@dataclass
class RetrievalQualityReport:
    scored: list[QuestionScore] = field(default_factory=list)
    skipped_ids: list[str] = field(default_factory=list)  # no expected_citations
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_reciprocal_rank: float = 0.0
    mean_average_precision: float = 0.0
    min_context_precision_threshold: float = 0.0
    min_citation_recall_threshold: float = 0.0
    passed: bool = False
    failing_questions: list[str] = field(default_factory=list)


def _relevant_keys(expected_citations: list[str]) -> set[str]:
    return {canonical_document_key(c) for c in expected_citations}


def _ranked_keys(ranked_citations: list[str]) -> list[str]:
    # Preserve order and rank position while normalizing each id, matching
    # the same normalization applied to `expected_citations` so the two
    # sides are comparable regardless of which surface form (filename stem
    # vs. entity-derived name) either side happens to use -- see
    # canonical_document_key's own docstring for the two-naming-system case
    # this addresses.
    return [canonical_document_key(c) for c in ranked_citations]


async def evaluate_retrieval_quality(
    questions: list[dict],
    retrieve: RetrieveFn,
    thresholds: dict,
    k: int = DEFAULT_K,
) -> RetrievalQualityReport:
    """Score `retrieve`'s ranked output for each question against its
    `expected_citations`, then compare the aggregate means against
    `thresholds["min_context_precision"]` / `thresholds["min_citation_recall"]`.

    Questions with no `expected_citations` (e.g. golden_set.json's
    intentionally-unanswerable meta-questions, per its own v2.1 changelog
    entry) are skipped rather than scored -- there is no relevant set to
    rank against, and recall_at_k's vacuous-1.0 convention for an empty
    relevant set would silently inflate the aggregate if included.
    """
    report = RetrievalQualityReport(
        min_context_precision_threshold=thresholds.get("min_context_precision", 0.0),
        min_citation_recall_threshold=thresholds.get("min_citation_recall", 0.0),
    )

    for question in questions:
        expected = question.get("expected_citations") or []
        if not expected:
            report.skipped_ids.append(question["id"])
            continue

        relevant = _relevant_keys(expected)
        ranked = _ranked_keys(await retrieve(question["question"]))

        report.scored.append(
            QuestionScore(
                id=question["id"],
                precision_at_k=ir_metrics.precision_at_k(ranked, relevant, k),
                recall_at_k=ir_metrics.recall_at_k(ranked, relevant, k),
                reciprocal_rank=ir_metrics.reciprocal_rank(ranked, relevant),
                average_precision=ir_metrics.average_precision(ranked, relevant),
            )
        )

    n = len(report.scored)
    if n:
        report.mean_precision = sum(s.precision_at_k for s in report.scored) / n
        report.mean_recall = sum(s.recall_at_k for s in report.scored) / n
        report.mean_reciprocal_rank = sum(s.reciprocal_rank for s in report.scored) / n
        report.mean_average_precision = sum(s.average_precision for s in report.scored) / n

    report.failing_questions = [
        s.id
        for s in report.scored
        if s.precision_at_k < report.min_context_precision_threshold
        or s.recall_at_k < report.min_citation_recall_threshold
    ]
    report.passed = (
        n > 0
        and report.mean_precision >= report.min_context_precision_threshold
        and report.mean_recall >= report.min_citation_recall_threshold
    )

    return report


__all__ = ["evaluate_retrieval_quality", "RetrievalQualityReport", "QuestionScore", "RetrieveFn"]
