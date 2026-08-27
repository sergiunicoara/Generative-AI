"""Evaluation-backend selection with an explicit, observable RAGAS fallback."""

from __future__ import annotations

import structlog

from graphrag.core.config import get_settings
from graphrag.core.exceptions import EvaluationError
from graphrag.evaluation.ragas_evaluator import RagasEvaluator
from graphrag.evaluation.reference_evaluator import ReferenceEvaluator

log = structlog.get_logger(__name__)


class ResilientEvaluator:
    def __init__(self, primary, fallback=None):
        self.primary = primary
        self.fallback = fallback

    async def evaluate_single(self, **kwargs):
        try:
            return await self.primary.evaluate_single(**kwargs)
        except EvaluationError:
            if self.fallback is None:
                raise
            log.warning("evaluation.primary_failed_using_fallback", primary="ragas", fallback="reference")
            return await self.fallback.evaluate_single(**kwargs)


def build_evaluator():
    cfg = get_settings().evaluation
    backend = str(cfg.get("backend", "ragas"))
    if backend == "reference":
        return ReferenceEvaluator()
    if backend != "ragas":
        raise ValueError("evaluation.backend must be 'ragas' or 'reference'")
    fallback = ReferenceEvaluator() if cfg.get("ragas_fallback_to_reference", True) else None
    return ResilientEvaluator(RagasEvaluator(), fallback)


__all__ = ["ResilientEvaluator", "build_evaluator"]
