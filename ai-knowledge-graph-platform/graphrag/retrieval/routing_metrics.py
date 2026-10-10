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
