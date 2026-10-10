"""Routing / fallback / sufficiency metrics. Labels are bounded enums only."""
from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram
except ImportError:  # pragma: no cover
    Counter = Histogram = None

_routes = Counter("graphrag_query_routes_total", "Query routes chosen by the deterministic router",
                  ["route", "policy"]) if Counter else None
_fallbacks = Counter("graphrag_fallback_triggers_total", "Agentic (IRCoT) fallback triggers",
                     ["reason"]) if Counter else None
_insufficient = Counter("graphrag_insufficient_context_total", "Answers judged insufficient, by reason",
                        ["reason"]) if Counter else None
_stale = Counter("graphrag_stale_evidence_total",
                 "Retrieved chunks whose evidence is not current (superseded/expired/stale)") if Counter else None
_latency = Histogram("graphrag_retrieval_latency_seconds", "End-to-end retrieval+answer latency by mode",
                     ["mode"], buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)) if Histogram else None
_expansion = Histogram("graphrag_graph_expansion_results", "Chunks added by graph expansion per query",
                       buckets=(0, 1, 2, 5, 10, 20, 50, 100)) if Histogram else None
_denials = Counter("graphrag_retrieval_authorization_denials_total",
                   "Queries whose evidence was entirely removed by authorization") if Counter else None


_coverage = Histogram("graphrag_answer_evidence_coverage",
                      "Share of answer statements supported by the retrieved evidence (lexical grounding)",
                      buckets=(0.0, 0.25, 0.5, 0.75, 0.9, 1.0)) if Histogram else None
_unsupported = Counter("graphrag_unsupported_statements_total",
                       "Answer statements not supported by the retrieved evidence") if Counter else None
_statements = Counter("graphrag_answer_statements_total",
                      "Answer statements checked for evidence support") if Counter else None
_superseded_used = Counter("graphrag_superseded_evidence_used_total",
                           "Evidence items in answers that are superseded or otherwise not current") if Counter else None
_claims_checked = Counter("graphrag_claims_verified_total", "Sentences checked by the LLM claim verifier") if Counter else None
_claims_stripped = Counter("graphrag_claims_stripped_total", "Sentences removed by the LLM claim verifier") if Counter else None


def record_answer_grounding(grounding: dict, *, non_current_evidence: int = 0) -> None:
    """Evidence coverage, unsupported-statement rate and non-current evidence usage for one answer."""
    if not grounding:
        return
    n = int(grounding.get("statements") or 0)
    if _coverage is not None and n:
        _coverage.observe(float(grounding.get("grounding_ratio", 1.0)))
    if _statements is not None and n:
        _statements.inc(n)
    if _unsupported is not None and grounding.get("unsupported_statements"):
        _unsupported.inc(len(grounding["unsupported_statements"]))
    if _superseded_used is not None and non_current_evidence:
        _superseded_used.inc(non_current_evidence)


def record_claim_verification(total: int, removed: int) -> None:
    if _claims_checked is not None and total:
        _claims_checked.inc(total)
    if _claims_stripped is not None and removed:
        _claims_stripped.inc(removed)


def record_route(route: str, policy: str) -> None:
    if _routes is not None:
        _routes.labels(route=route, policy=policy).inc()


def record_fallback(reason: str) -> None:
    if _fallbacks is not None:
        _fallbacks.labels(reason=reason).inc()


def record_insufficient(reason: str) -> None:
    if _insufficient is not None:
        _insufficient.labels(reason=reason).inc()


def record_stale(n: int) -> None:
    if _stale is not None and n:
        _stale.inc(n)


def record_latency(mode: str, seconds: float) -> None:
    if _latency is not None:
        _latency.labels(mode=mode).observe(max(0.0, seconds))


def record_expansion(n: int) -> None:
    if _expansion is not None:
        _expansion.observe(n)


def record_denial() -> None:
    if _denials is not None:
        _denials.inc()
