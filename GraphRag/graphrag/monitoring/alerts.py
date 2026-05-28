"""Alert service — threshold-based alerts from GraphHealthSnapshot metrics.

Problem solved
--------------
GraphHealthSnapshot nodes are persisted after every evaluator run but nothing
reads them proactively.  Threshold breaches go unnoticed until someone manually
queries the graph or RAGAS scores start falling (a lagging signal).

Architecture
------------
- AlertService.check(report) — compares metric values in a full_report() dict
  against configured thresholds; returns a list of alert dicts.
- AlertService.fire(alerts) — emits a structlog ERROR for each alert with
  structured fields (alert_type, metric, value, threshold, tenant, fired_at).
  Compatible with any log aggregator (ELK, Datadog, GCP Logging).
- AlertService.check_and_fire(report, tenant) — convenience wrapper.
- A module-level deque (_recent_alerts) accumulates the last ALERT_HISTORY
  alerts so the GET /kg/health/alerts endpoint can return them without a DB query.

Configuration (config/settings.yml → business_matrix.alert_thresholds):
  latency_p95_ms:      3000   # breach if API p95 latency exceeds this (ms)
  faithfulness:        0.7    # breach if RAGAS faithfulness drops below this
  context_recall:      0.6    # breach if RAGAS context_recall drops below this
  contradiction_rate:  0.05   # breach if conflicts_per_1k_edges exceeds 5 %
  orphan_rate:         0.10   # breach if orphan_rate exceeds 10 %
  low_confidence_rate: 0.30   # breach if noise_edge_rate exceeds 30 %

Wiring
------
GraphEvaluator.persist_snapshot() calls get_alert_service().check_and_fire()
after writing the GraphHealthSnapshot node, so every health-check pass also
runs the alert check automatically.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

import structlog

log = structlog.get_logger(__name__)

ALERT_HISTORY = 100   # how many recent alerts to keep in memory


# ── Default thresholds ─────────────────────────────────────────────────────────

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "latency_p95_ms":     3000.0,
    "faithfulness":       0.7,
    "context_recall":     0.6,
    "contradiction_rate": 0.05,   # conflicts_per_1k_edges / 1000 (normalised to 0-1)
    "orphan_rate":        0.10,
    "low_confidence_rate": 0.30,
}

# Module-level accumulator — populated by AlertService.fire()
_recent_alerts: deque[dict] = deque(maxlen=ALERT_HISTORY)


class AlertService:
    """
    Checks graph health metrics against configured thresholds and emits
    structured log events when thresholds are breached.

    Usage::

        svc = get_alert_service()
        alerts = svc.check_and_fire(report, tenant="acme")
        # Each breach fires a structlog ERROR that any log aggregator can route.
    """

    def __init__(self, thresholds: dict | None = None):
        """
        Parameters
        ----------
        thresholds:
            Override or extend the default thresholds.  Keys must match the
            metric names listed in the module docstring.
        """
        self._thresholds = dict(_DEFAULT_THRESHOLDS)
        if thresholds:
            self._thresholds.update(thresholds)

    # ── Threshold access ───────────────────────────────────────────────────────

    @property
    def thresholds(self) -> dict[str, float]:
        """Return a copy of the active thresholds dict."""
        return dict(self._thresholds)

    # ── Core check ────────────────────────────────────────────────────────────

    def check(self, report: dict, tenant: str = "default") -> list[dict]:
        """
        Compare metrics in *report* against configured thresholds.

        Parameters
        ----------
        report:
            Dict as returned by ``GraphEvaluator.full_report()``.
        tenant:
            Tenant label to attach to every alert dict.

        Returns
        -------
        list[dict]
            One dict per breached threshold, empty list when all is healthy.
            Each dict: ``{alert_type, metric, value, threshold, tenant, fired_at}``.
        """
        alerts: list[dict] = []
        now = datetime.now(timezone.utc).isoformat()

        def _alert(metric: str, value: float, threshold: float, direction: str = "above") -> None:
            alerts.append({
                "alert_type": "threshold_breach",
                "metric":     metric,
                "value":      round(value, 6),
                "threshold":  threshold,
                "direction":  direction,   # "above" = bad when value > threshold
                "tenant":     tenant,
                "fired_at":   now,
            })

        # ── Contradiction rate ─────────────────────────────────────────────────
        # full_report key: report["contradiction"]["conflicts_per_1k_edges"]
        # threshold is expressed as rate (0.0–1.0) → normalise 1k-edges value
        contradiction_raw = (
            report.get("contradiction", {}).get("conflicts_per_1k_edges", 0) or 0
        )
        contradiction_rate = contradiction_raw / 1000.0   # convert to 0-1 scale
        thr = self._thresholds.get("contradiction_rate", _DEFAULT_THRESHOLDS["contradiction_rate"])
        if contradiction_rate > thr:
            _alert("contradiction_rate", contradiction_rate, thr)

        # ── Orphan rate ────────────────────────────────────────────────────────
        orphan_rate = report.get("orphan_growth", {}).get("orphan_rate", 0) or 0
        thr = self._thresholds.get("orphan_rate", _DEFAULT_THRESHOLDS["orphan_rate"])
        if orphan_rate > thr:
            _alert("orphan_rate", orphan_rate, thr)

        # ── Low-confidence edge rate ───────────────────────────────────────────
        low_conf_rate = (
            report.get("relation_precision", {}).get("noise_edge_rate", 0) or 0
        )
        thr = self._thresholds.get("low_confidence_rate", _DEFAULT_THRESHOLDS["low_confidence_rate"])
        if low_conf_rate > thr:
            _alert("low_confidence_rate", low_conf_rate, thr)

        # ── RAGAS faithfulness (if present in report) ─────────────────────────
        faithfulness = report.get("faithfulness")
        if faithfulness is not None:
            thr = self._thresholds.get("faithfulness", _DEFAULT_THRESHOLDS["faithfulness"])
            if faithfulness < thr:   # direction: "below" is bad
                _alert("faithfulness", faithfulness, thr, direction="below")

        # ── RAGAS context recall (if present in report) ───────────────────────
        context_recall = report.get("context_recall")
        if context_recall is not None:
            thr = self._thresholds.get("context_recall", _DEFAULT_THRESHOLDS["context_recall"])
            if context_recall < thr:
                _alert("context_recall", context_recall, thr, direction="below")

        # ── API latency p95 (if present in report) ────────────────────────────
        latency = report.get("latency_p95_ms")
        if latency is not None:
            thr = self._thresholds.get("latency_p95_ms", _DEFAULT_THRESHOLDS["latency_p95_ms"])
            if latency > thr:
                _alert("latency_p95_ms", latency, thr)

        return alerts

    # ── Fire ──────────────────────────────────────────────────────────────────

    def fire(self, alerts: list[dict]) -> None:
        """
        Emit a structlog ERROR for each alert and append to the in-memory
        history deque so GET /kg/health/alerts can retrieve them.
        """
        for alert in alerts:
            log.error(
                "alert.threshold_breach",
                metric=alert["metric"],
                value=alert["value"],
                threshold=alert["threshold"],
                direction=alert["direction"],
                tenant=alert["tenant"],
                fired_at=alert["fired_at"],
            )
            _recent_alerts.append(alert)

    # ── Convenience ───────────────────────────────────────────────────────────

    def check_and_fire(self, report: dict, tenant: str = "default") -> list[dict]:
        """Check thresholds, fire any breaches, and return the alerts list."""
        alerts = self.check(report, tenant=tenant)
        if alerts:
            self.fire(alerts)
        return alerts


# ── Module-level helpers ───────────────────────────────────────────────────────

def get_recent_alerts(limit: int = ALERT_HISTORY) -> list[dict]:
    """Return the most recently fired alerts (newest first)."""
    items = list(_recent_alerts)
    items.reverse()
    return items[:limit]


_svc: AlertService | None = None


def get_alert_service() -> AlertService:
    """Return the process-level AlertService singleton (created lazily).

    Reads ``business_matrix.alert_thresholds`` from ``config/settings.yml``
    and merges them over the defaults.
    """
    global _svc
    if _svc is None:
        try:
            from graphrag.core.config import get_settings
            cfg = get_settings()
            bm_thresholds = (
                getattr(cfg, "business_matrix", None) or {}
            )
            thresholds = bm_thresholds.get("alert_thresholds", {}) if isinstance(bm_thresholds, dict) else {}
        except Exception:
            thresholds = {}
        _svc = AlertService(thresholds=thresholds or None)
    return _svc
