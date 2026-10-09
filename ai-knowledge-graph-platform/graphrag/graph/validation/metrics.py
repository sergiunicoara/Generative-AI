"""Low-cardinality Prometheus metrics for the publication gate.

Labels are bounded enums (rule id, severity, record kind, outcome). Tenant,
document and entity identifiers go to structured logs, never to labels.
"""
from __future__ import annotations

try:
    from prometheus_client import Counter
except ImportError:  # pragma: no cover - optional at local import time
    Counter = None

from graphrag.graph.validation.report import ValidationReport

_violations = Counter(
    "graphrag_validation_violations_total",
    "Graph-publication validation violations by rule and severity",
    ["rule_id", "severity"],
) if Counter else None
_quarantined = Counter(
    "graphrag_quarantined_records_total",
    "Records quarantined by the publication gate",
    ["record_kind"],
) if Counter else None
_batches = Counter(
    "graphrag_publication_batches_total",
    "Ingestion batches by publication outcome",
    ["outcome"],
) if Counter else None
_retries = Counter(
    "graphrag_quarantine_retries_total",
    "Quarantine retry attempts by outcome",
    ["outcome"],
) if Counter else None


def record_report(report: ValidationReport) -> None:
    if _violations is None:
        return
    for v in report.violations:
        _violations.labels(rule_id=v.rule_id, severity=v.severity.value).inc()


def record_quarantined(record_kind: str, n: int = 1) -> None:
    if _quarantined is not None and n:
        _quarantined.labels(record_kind=record_kind).inc(n)


def record_batch(outcome: str) -> None:
    if _batches is not None:
        _batches.labels(outcome=outcome).inc()


def record_retry(outcome: str) -> None:
    if _retries is not None:
        _retries.labels(outcome=outcome).inc()
