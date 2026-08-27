"""Dependency-light, deterministic evaluation backend.

This is not presented as a semantic-judge replacement: it measures explicit
token support by the supplied evidence and reference answer.  It is a safe
fallback when an optional upstream judge cannot be imported or called, and it
makes the fallback visible in every persisted KPI through ``evaluation_source``.
"""

from __future__ import annotations

import re

from graphrag.core.models import EvalResult

_TOKEN = re.compile(r"[\w]+(?:[-'][\w]+)*", re.UNICODE)


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in _TOKEN.findall(value or "")}


def _coverage(needles: set[str], haystack: set[str]) -> float:
    return len(needles & haystack) / len(needles) if needles else 0.0


class ReferenceEvaluator:
    """Compute auditable lexical support scores without network dependencies."""

    async def evaluate_single(
        self,
        query_id: str,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = "",
    ) -> EvalResult:
        answer_tokens = _tokens(answer)
        question_tokens = _tokens(question)
        context_sets = [_tokens(context) for context in contexts if context]
        context_tokens = set().union(*context_sets) if context_sets else set()
        reference_tokens = _tokens(ground_truth)
        supported = _coverage(answer_tokens, context_tokens)
        precision = (
            sum(_coverage(reference_tokens, context) for context in context_sets) / len(context_sets)
            if context_sets and reference_tokens else 0.0
        )
        return EvalResult(
            job_id=query_id,
            query_id=query_id,
            faithfulness=supported,
            answer_relevancy=_coverage(answer_tokens, question_tokens),
            context_precision=precision,
            context_recall=_coverage(reference_tokens, context_tokens),
            evaluation_source="reference",
        )


__all__ = ["ReferenceEvaluator"]
