"""Unit tests for graphrag.monitoring.alerts — AlertService threshold checks."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from graphrag.monitoring.alerts import (
    AlertService,
    _DEFAULT_THRESHOLDS,
    _recent_alerts,
    get_recent_alerts,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _report(**overrides) -> dict:
    """Build a minimal full_report()-style dict with safe defaults."""
    base = {
        "contradiction": {"conflicts_per_1k_edges": 0},
        "orphan_growth":  {"orphan_rate": 0.0},
        "relation_precision": {"noise_edge_rate": 0.0},
    }
    base.update(overrides)
    return base


# ── tests ──────────────────────────────────────────────────────────────────────

def test_no_alerts_when_all_metrics_healthy():
    svc = AlertService()
    alerts = svc.check(_report())
    assert alerts == []


def test_contradiction_rate_breach():
    """conflicts_per_1k_edges > threshold*1000 triggers contradiction_rate alert."""
    svc = AlertService(thresholds={"contradiction_rate": 0.05})
    # 0.05 rate = 50 per 1k edges. 51 > 50 → breach
    report = _report(contradiction={"conflicts_per_1k_edges": 51})
    alerts = svc.check(report)
    assert len(alerts) == 1
    assert alerts[0]["metric"] == "contradiction_rate"
    assert alerts[0]["direction"] == "above"


def test_orphan_rate_breach():
    svc = AlertService(thresholds={"orphan_rate": 0.10})
    report = _report(orphan_growth={"orphan_rate": 0.15})
    alerts = svc.check(report)
    assert any(a["metric"] == "orphan_rate" for a in alerts)


def test_low_confidence_rate_breach():
    svc = AlertService(thresholds={"low_confidence_rate": 0.30})
    report = _report(relation_precision={"noise_edge_rate": 0.35})
    alerts = svc.check(report)
    assert any(a["metric"] == "low_confidence_rate" for a in alerts)


def test_faithfulness_breach_below_threshold():
    """faithfulness < threshold fires a 'below' direction alert."""
    svc = AlertService(thresholds={"faithfulness": 0.8})
    report = _report(faithfulness=0.5)
    alerts = svc.check(report)
    faith_alerts = [a for a in alerts if a["metric"] == "faithfulness"]
    assert len(faith_alerts) == 1
    assert faith_alerts[0]["direction"] == "below"
    assert faith_alerts[0]["value"] == pytest.approx(0.5)
    assert faith_alerts[0]["threshold"] == pytest.approx(0.8)


def test_faithfulness_absent_from_report_skipped():
    """When faithfulness is not in report, no alert is fired."""
    svc = AlertService()
    report = _report()   # no faithfulness key
    alerts = svc.check(report)
    assert not any(a["metric"] == "faithfulness" for a in alerts)


def test_multiple_breaches_returned():
    svc = AlertService(thresholds={
        "orphan_rate": 0.05,
        "low_confidence_rate": 0.10,
    })
    report = _report(
        orphan_growth={"orphan_rate": 0.20},
        relation_precision={"noise_edge_rate": 0.40},
    )
    alerts = svc.check(report)
    assert len(alerts) == 2


def test_custom_threshold_overrides_default():
    """Constructor threshold dict overrides the class defaults."""
    svc = AlertService(thresholds={"orphan_rate": 0.50})
    # Default is 0.10; custom is 0.50 → 0.30 should NOT breach
    report = _report(orphan_growth={"orphan_rate": 0.30})
    assert svc.check(report) == []


def test_fire_logs_error_for_each_alert():
    svc = AlertService()
    import graphrag.monitoring.alerts as alerts_module
    logged = []

    with patch.object(alerts_module.log, "error", side_effect=lambda *a, **kw: logged.append(kw)):
        # Mock Redis so fire() falls back to in-process deque
        with patch.object(alerts_module, "_push_to_redis", return_value=False):
            svc.fire([
                {"metric": "orphan_rate", "value": 0.2, "threshold": 0.1,
                 "direction": "above", "tenant": "t1",
                 "fired_at": datetime.now(timezone.utc).isoformat(),
                 "alert_type": "threshold_breach"},
            ])
    assert len(logged) == 1
    assert logged[0]["metric"] == "orphan_rate"


def test_check_and_fire_returns_alerts_and_fires():
    svc = AlertService(thresholds={"orphan_rate": 0.05})
    import graphrag.monitoring.alerts as alerts_module
    fired = []

    with patch.object(alerts_module.log, "error", side_effect=lambda *a, **kw: fired.append(kw)):
        with patch.object(alerts_module, "_push_to_redis", return_value=False):
            alerts = svc.check_and_fire(
                _report(orphan_growth={"orphan_rate": 0.20}), tenant="acme"
            )

    assert len(alerts) == 1
    assert alerts[0]["tenant"] == "acme"
    assert len(fired) == 1


def test_alert_dict_has_required_fields():
    svc = AlertService(thresholds={"orphan_rate": 0.05})
    alerts = svc.check(_report(orphan_growth={"orphan_rate": 0.20}))
    a = alerts[0]
    required = {"alert_type", "metric", "value", "threshold", "direction", "tenant", "fired_at"}
    assert required.issubset(a.keys())


def test_threshold_property_returns_copy():
    """Mutating the returned thresholds dict does not affect the service."""
    svc = AlertService()
    t = svc.thresholds
    t["orphan_rate"] = 999.0
    assert svc.thresholds["orphan_rate"] == _DEFAULT_THRESHOLDS["orphan_rate"]
