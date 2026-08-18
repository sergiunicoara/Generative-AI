"""KPI timeseries metric allowlist — identifier-column exfiltration guard.

`KPITracker.get_timeseries` selects its ORM column via `getattr(KPIEventRow,
metric)`, where `metric` is a raw query-string parameter. `KPIEventRow` also
carries `query_id`, and `GET /kpis/timeseries` had neither a scope requirement
nor a tenant filter — `KPIEventRow` has no tenant column to filter on.

So `?metric=query_id` returned every tenant's query IDs, which
`GET /query/{query_id}` would then redeem for the full stored answer. These
tests pin the allowlist that removes the enumeration half of that chain.

See docs/context_graph_gap_plan.md F10.
"""

from __future__ import annotations

import pytest

from graphrag.business_matrix.kpi_tracker import (
    _ALLOWED_TIMESERIES_METRICS,
    KPITracker,
)


class TestTimeseriesMetricAllowlist:
    @pytest.mark.parametrize("identifier_column", ["query_id", "event_id", "model_version"])
    async def test_identifier_columns_are_rejected(self, identifier_column):
        """The exfiltration vector itself: a non-measurement column must not be
        selectable, even though it is a real attribute of KPIEventRow."""
        with pytest.raises(ValueError, match="unsupported metric"):
            await KPITracker().get_timeseries(metric=identifier_column)

    async def test_unknown_column_is_rejected_not_silently_defaulted(self):
        """The old code fell back to latency_ms for anything unrecognized, so an
        identifier request looked like a successful latency query."""
        with pytest.raises(ValueError, match="unsupported metric"):
            await KPITracker().get_timeseries(metric="no_such_column")

    async def test_dunder_attribute_is_rejected(self):
        """getattr() would otherwise happily return ORM/py internals."""
        with pytest.raises(ValueError, match="unsupported metric"):
            await KPITracker().get_timeseries(metric="__class__")

    def test_allowlist_contains_only_numeric_measurements(self):
        """Guard against someone widening the set to include an identifier
        while chasing a dashboard feature."""
        assert _ALLOWED_TIMESERIES_METRICS == frozenset({
            "latency_ms", "faithfulness", "answer_relevancy",
            "context_precision", "context_recall", "cost_usd",
        })
        assert "query_id" not in _ALLOWED_TIMESERIES_METRICS
        assert "event_id" not in _ALLOWED_TIMESERIES_METRICS
