from unittest.mock import AsyncMock

import pytest

from graphrag.core.exceptions import EvaluationError
from graphrag.evaluation.factory import ResilientEvaluator
from graphrag.evaluation.reference_evaluator import ReferenceEvaluator


@pytest.mark.asyncio
async def test_reference_evaluator_reports_its_non_ragas_provenance():
    result = await ReferenceEvaluator().evaluate_single(
        query_id="q1", question="What colour is the sky?", answer="The sky is blue.",
        contexts=["The sky is blue on a clear day."], ground_truth="blue",
    )

    assert result.evaluation_source == "reference"
    assert result.faithfulness > 0.5
    assert result.context_recall == 1.0


@pytest.mark.asyncio
async def test_ragas_failure_falls_back_without_mislabeling_score():
    primary = AsyncMock()
    primary.evaluate_single.side_effect = EvaluationError("upstream unavailable")
    evaluator = ResilientEvaluator(primary, ReferenceEvaluator())

    result = await evaluator.evaluate_single(
        query_id="q1", question="alpha?", answer="alpha", contexts=["alpha"], ground_truth="alpha",
    )

    assert result.evaluation_source == "reference"
