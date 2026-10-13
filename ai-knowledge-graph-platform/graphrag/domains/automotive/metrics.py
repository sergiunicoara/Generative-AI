"""Prometheus gauges for IATF audit state, set whenever the API computes the clause view or verifies the chain.

Labels are tenant + a fixed status vocabulary (low cardinality: only tenants configured under ``iatf_audit`` exist).
Alert examples: ``graphrag_iatf_signoff_chain_valid == 0`` (tamper) or ``graphrag_iatf_clauses{status="missing"} > 0``.
"""
from __future__ import annotations

from typing import Any

import structlog

from graphrag.domains.automotive.audit import STATUSES

log = structlog.get_logger(__name__)

try:
    from prometheus_client import Gauge
except ImportError:  # pragma: no cover - optional dependency
    Gauge = None

MAPPING_STATUSES = ("proposed", "confirmed", "rejected", "changed_since_review")

_clauses = Gauge("graphrag_iatf_clauses", "IATF catalogue clauses by evidence status",
                 ["tenant", "status"]) if Gauge else None
_mapping = Gauge("graphrag_iatf_clause_mappings", "IATF clause mappings by review status",
                 ["tenant", "status"]) if Gauge else None
_conflicted = Gauge("graphrag_iatf_clauses_with_conflicts", "IATF clauses whose current evidence has open "
                    "contradiction tags", ["tenant"]) if Gauge else None
_chain_valid = Gauge("graphrag_iatf_signoff_chain_valid", "1 when the tenant's sign-off hash chain verifies",
                     ["tenant"]) if Gauge else None


def _safe(fn) -> None:
    """Instrumentation must never break the request it observes."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        log.debug("iatf_metrics.update_failed", error=str(exc))


def record_view(tenant: str, view: dict[str, Any]) -> None:
    """Only a full-catalogue view is recorded; a single-clause view would under-count."""
    if _clauses is None:
        return

    def update():
        for status in STATUSES:
            _clauses.labels(tenant=tenant, status=status).set(view["counts"].get(status, 0))
        mapping_counts = {s: 0 for s in MAPPING_STATUSES}
        for clause in view["clauses"]:
            status = (clause.get("mapping") or {}).get("status")
            if status in mapping_counts:
                mapping_counts[status] += 1
        for status, count in mapping_counts.items():
            _mapping.labels(tenant=tenant, status=status).set(count)
        _conflicted.labels(tenant=tenant).set(sum(1 for c in view["clauses"] if c["conflicts"]))

    _safe(update)


def record_chain(tenant: str, verdict: dict[str, Any]) -> None:
    if _chain_valid is None:
        return
    _safe(lambda: _chain_valid.labels(tenant=tenant).set(1 if verdict.get("valid") else 0))


__all__ = ["MAPPING_STATUSES", "record_chain", "record_view"]
