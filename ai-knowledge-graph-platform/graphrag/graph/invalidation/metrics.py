"""Low-cardinality metrics for targeted invalidation (bounded enum labels only)."""
from __future__ import annotations

try:
    from prometheus_client import Counter
except ImportError:  # pragma: no cover
    Counter = None

_events = Counter("graphrag_invalidation_events_total",
                  "Invalidation events by kind and outcome", ["kind", "outcome"]) if Counter else None
_artifacts = Counter("graphrag_invalidated_artifacts_total",
                     "Derived artifacts marked NEEDS_REVIEW, by kind", ["artifact_kind"]) if Counter else None
_evicted = Counter("graphrag_invalidated_cached_answers_total",
                   "Cached answers evicted by targeted invalidation") if Counter else None
_recomputed = Counter("graphrag_recomputed_artifacts_total",
                      "Recompute outcomes by artifact kind", ["artifact_kind", "outcome"]) if Counter else None


def record_event(kind: str, outcome: str) -> None:
    if _events is not None:
        _events.labels(kind=kind, outcome=outcome).inc()


def record_artifacts(kind: str, n: int) -> None:
    if _artifacts is not None and n:
        _artifacts.labels(artifact_kind=kind).inc(n)


def record_evicted(n: int) -> None:
    if _evicted is not None and n:
        _evicted.inc(n)


def record_recompute(kind: str, outcome: str) -> None:
    if _recomputed is not None:
        _recomputed.labels(artifact_kind=kind, outcome=outcome).inc()
