"""Deterministic, versioned evaluation rubrics.

Rubrics intentionally consume a plain observation mapping so they can be used
by API, workers, and offline replay without coupling the evaluator to a
particular retriever or LLM provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from pydantic import BaseModel, Field


class RubricResult(BaseModel):
    rubric_id: str
    version: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    hard_failed: bool = False
    rubrics: list[RubricResult] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)


def build_observations(
    *,
    answer: str,
    citations: list[str],
    contexts: list[str],
    tenant: str,
    policy_result: str = "",
    policy_reason_code: str = "",
    valid_at: str | None = None,
    transaction_at: str | None = None,
    trajectory: Any = None,
    faithfulness: float = 0.0,
    latency_ms: float = 0.0,
    latency_budget_ms: float | None = None,
    cost_usd: float | None = None,
    cost_budget_usd: float | None = None,
    answer_support_threshold: float = 0.8,
) -> dict[str, Any]:
    """Build auditable deterministic inputs from a completed query turn.

    An empty ``policy_result`` represents a legacy result that predates the
    retrieval authorization field.  It is recorded as *unknown*, rather than
    being reclassified as an authorization success.  Explicit ``escalate`` or
    ``deny`` results are mandatory hard failures.
    """
    context_text = "\n".join(contexts).casefold()
    normalized_citations = [item.strip() for item in citations if item and item.strip()]
    citation_resolved = bool(normalized_citations) and all(
        citation.casefold() in context_text for citation in normalized_citations
    )
    outcomes = [str(getattr(step, "outcome", "completed")).casefold() for step in getattr(trajectory, "steps", [])]
    policy = policy_result.casefold().strip()
    explicitly_denied = policy in {"deny", "denied", "escalate", "blocked"}
    policy_known = policy in {"allow", "allowed", "deny", "denied", "escalate", "blocked"}
    temporal_request = valid_at is not None or transaction_at is not None
    try:
        from graphrag.graph.pii_guard import PIIGuard
        pii_detected = PIIGuard(None).has_pii(answer)
    except Exception:  # pragma: no cover - PII detection must never break evaluation
        pii_detected = True
    return {
        "citation_present": bool(normalized_citations),
        "citation_present_evidence": normalized_citations,
        "citation_resolves_to_source": citation_resolved,
        "citation_resolves_to_source_evidence": normalized_citations if citation_resolved else [],
        "answer_supported": faithfulness >= answer_support_threshold,
        "answer_supported_evidence": [f"faithfulness={faithfulness:.4f}", f"threshold={answer_support_threshold:.4f}"],
        "tool_execution_success": not any(outcome in {"failed", "error", "denied"} for outcome in outcomes),
        "authorized_scope": not explicitly_denied,
        "authorized_scope_evidence": [f"policy_result={policy or 'unknown'}", f"reason={policy_reason_code or 'unknown'}"],
        "tenant_scope_preserved": bool(tenant.strip()),
        "tenant_scope_preserved_evidence": [f"tenant={tenant}"],
        "freshness_verified": not temporal_request or bool(normalized_citations),
        "freshness_verified_evidence": [f"valid_at={valid_at}", f"transaction_at={transaction_at}"],
        "cost_budget_respected": cost_budget_usd is None or (cost_usd is not None and cost_usd <= cost_budget_usd),
        "cost_budget_respected_evidence": [f"cost_usd={cost_usd}", f"budget_usd={cost_budget_usd}"],
        "latency_budget_respected": latency_budget_ms is None or latency_ms <= latency_budget_ms,
        "latency_budget_respected_evidence": [f"latency_ms={latency_ms}", f"budget_ms={latency_budget_ms}"],
        "pii_policy_respected": not pii_detected,
        "pii_policy_respected_evidence": ["pii_scan=clear" if not pii_detected else "pii_scan=detected"],
        "authorization_policy_known": policy_known,
    }


RubricFn = Callable[[Mapping[str, Any]], RubricResult]


@dataclass(frozen=True)
class RubricSpec:
    rubric_id: str
    version: str
    evaluate: RubricFn
    weight: float = 1.0
    hard_failure: bool = False
    depends_on: tuple[str, ...] = ()
    penalty: float = 0.0


def _check(rubric_id: str, obs: Mapping[str, Any], key: str, *, version: str = "1.0") -> RubricResult:
    value = bool(obs.get(key, False))
    return RubricResult(
        rubric_id=rubric_id, version=version, passed=value, score=1.0 if value else 0.0,
        reason=f"{key}={value}", evidence=[str(x) for x in obs.get(f"{key}_evidence", [])],
    )


class RubricRegistry:
    def __init__(self, specs: list[RubricSpec] | None = None) -> None:
        self._specs: dict[str, RubricSpec] = {}
        for spec in specs or default_rubrics():
            self.register(spec)

    def register(self, spec: RubricSpec) -> None:
        if not spec.rubric_id or spec.weight < 0 or spec.penalty < 0:
            raise ValueError("invalid rubric specification")
        self._specs[spec.rubric_id] = spec

    def get(self, rubric_id: str) -> RubricSpec:
        return self._specs[rubric_id]

    def evaluate(self, observations: Mapping[str, Any], rubric_ids: list[str] | None = None) -> EvaluationResult:
        selected = rubric_ids or list(self._specs)
        results: list[RubricResult] = []
        passed_ids: set[str] = set()
        for rubric_id in selected:
            spec = self.get(rubric_id)
            missing = [dep for dep in spec.depends_on if dep not in passed_ids]
            result = RubricResult(rubric_id=spec.rubric_id, version=spec.version, passed=False, score=0.0,
                                  reason=f"dependency failed: {', '.join(missing)}") if missing else spec.evaluate(observations)
            results.append(result)
            if result.passed:
                passed_ids.add(rubric_id)
        total_weight = sum(self.get(r.rubric_id).weight for r in results)
        score = sum(r.score * self.get(r.rubric_id).weight for r in results) / total_weight if total_weight else 0.0
        score -= sum(self.get(r.rubric_id).penalty for r in results if not r.passed)
        hard_failed = any(not r.passed and self.get(r.rubric_id).hard_failure for r in results)
        return EvaluationResult(score=max(0.0, min(1.0, score)), passed=not hard_failed and all(r.passed for r in results),
                                hard_failed=hard_failed, rubrics=results,
                                config={"rubric_versions": {r.rubric_id: r.version for r in results}})


def default_rubrics() -> list[RubricSpec]:
    hard = {"authorized_scope", "tenant_scope_preserved", "pii_policy_respected"}
    ids = ("citation_present", "citation_resolves_to_source", "answer_supported", "tool_execution_success",
           "authorized_scope", "tenant_scope_preserved", "freshness_verified", "cost_budget_respected",
           "latency_budget_respected", "pii_policy_respected")
    return [RubricSpec(rubric_id=i, version="1.0", evaluate=lambda obs, i=i: _check(i, obs, i),
                       hard_failure=i in hard, weight=2.0 if i in hard else 1.0) for i in ids]


__all__ = [
    "EvaluationResult", "RubricRegistry", "RubricResult", "RubricSpec",
    "build_observations", "default_rubrics",
]
