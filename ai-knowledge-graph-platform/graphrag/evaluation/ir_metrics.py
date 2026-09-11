"""Classical information-retrieval metrics: precision@k, recall@k, MRR, NDCG.

Pure functions over plain ranked-id lists — no dependency on how the ranking
was produced. This lets the same functions score `/search` output
(`api/routes/search.py`), `LocalSearch.search()` output
(`graphrag/retrieval/local_search.py`), or a hand-written test fixture
identically, and keeps them trivially unit-testable without any retrieval
infrastructure.

Callers are expected to already have normalized ids into one comparable
space before calling these functions (e.g. via
`graphrag.graph.alias_registry.canonical_document_key`) — these functions do
plain equality/membership checks, no normalization of their own.
"""

from __future__ import annotations

import math


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top `k` ranked ids that are relevant.

    Returns 0.0 for an empty ranked list (nothing was returned to be
    precise about) — deliberately distinct from `recall_at_k`'s empty-
    `relevant` case below, which is a different degenerate condition.
    """
    if k <= 0 or not ranked:
        return 0.0
    top_k = ranked[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of all relevant ids captured within the top `k` ranked ids.

    Returns 1.0 when `relevant` is empty — there is nothing to miss, so
    recall is vacuously perfect. Callers with an empty `expected_citations`
    should exclude the question from scoring entirely rather than rely on
    this value (see `retrieval_quality_eval.py`), since a vacuous 1.0
    shouldn't be averaged in as if it were evidence of a well-ranked result.
    """
    if not relevant:
        return 1.0
    if k <= 0 or not ranked:
        return 0.0
    top_k = set(ranked[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1 / (rank of the first relevant id), 1-indexed; 0.0 if none found."""
    for i, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def average_precision(ranked: list[str], relevant: set[str]) -> float:
    """Mean of precision@k evaluated at each rank holding a relevant id.

    0.0 if `relevant` is empty or none of it appears in `ranked` — this is
    the standard AP convention (unlike `recall_at_k`, an empty `relevant`
    here is not vacuously perfect, since AP is defined as an average over
    the relevant set's own hit positions, which don't exist to average).
    """
    if not relevant or not ranked:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, item in enumerate(ranked, start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / i
    if hits == 0:
        return 0.0
    return precision_sum / len(relevant)


def ndcg_at_k(ranked: list[str], relevance: dict[str, float], k: int) -> float:
    """Normalized discounted cumulative gain over the top `k` ranked ids.

    `relevance` maps id -> graded relevance (missing ids default to 0, so a
    binary `relevant` set can be passed as `{id: 1.0 for id in relevant}`).
    Returns 0.0 when the ideal DCG is 0 (no positive relevance anywhere) —
    there is no achievable gain to normalize against.
    """
    if k <= 0 or not ranked:
        return 0.0

    def _dcg(ids: list[str]) -> float:
        total = 0.0
        for i, item in enumerate(ids[:k], start=1):
            gain = relevance.get(item, 0.0)
            if gain:
                # Standard log2(rank + 1) discount.
                total += gain / math.log2(i + 1)
        return total

    ideal_order = sorted(relevance, key=lambda i: relevance[i], reverse=True)
    ideal_dcg = _dcg(ideal_order)
    if ideal_dcg == 0.0:
        return 0.0
    return _dcg(ranked) / ideal_dcg


__all__ = [
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "average_precision",
    "ndcg_at_k",
]
