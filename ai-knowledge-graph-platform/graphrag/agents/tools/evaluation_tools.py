"""ADK FunctionTool wrappers for evaluation and KPI logging."""

from __future__ import annotations

import asyncio

from graphrag.core.models import EvalJob, KPIEvent


def score_answer(
    query_id: str,
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict:
    """Run RAGAS evaluation on a single query turn and return metric scores."""
    from graphrag.evaluation.ragas_evaluator import RagasEvaluator
    evaluator = RagasEvaluator()
    result = asyncio.run(
        evaluator.evaluate_single(
            query_id=query_id,
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )
    )
    return result.model_dump()


def log_kpi(event: dict) -> bool:
    """Persist a KPIEvent to the time-series store."""
    from graphrag.business_matrix.kpi_tracker import KPITracker
    tracker = KPITracker()
    kpi = KPIEvent(**event)
    asyncio.run(tracker.record(kpi))
    return True
