"""Tenant-aware cost aggregation (docs/IMPLEMENTATION_AUDIT.md item #5).

api/quota.py's record_tenant_usage() existed with zero callers, and every
CostEvent emitted from genai_telemetry._finish() -- the only place with a
real, non-zero per-call cost -- hardcoded tenant="". These tests guard the
fix: a tenant_context() published around retrieval must reach both the
CostEvent's tenant field and a scheduled record_tenant_usage() call.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from graphrag.observability.correlation import current_tenant, tenant_context
from graphrag.observability import genai_telemetry


def test_tenant_context_round_trips_and_resets():
    assert current_tenant() == ""
    with tenant_context("acme"):
        assert current_tenant() == "acme"
    assert current_tenant() == ""


def test_tenant_context_resets_on_exception():
    with pytest.raises(ValueError):
        with tenant_context("acme"):
            assert current_tenant() == "acme"
            raise ValueError("boom")
    assert current_tenant() == ""


def test_tenant_context_nesting_restores_outer_value():
    with tenant_context("outer"):
        with tenant_context("inner"):
            assert current_tenant() == "inner"
        assert current_tenant() == "outer"
    assert current_tenant() == ""


class TestFinishTenantAttribution:
    def test_cost_event_carries_active_tenant_and_schedules_usage(self):
        response = {"response_model": "gpt-test", "input_tokens": 10, "output_tokens": 5}
        with (
            patch("graphrag.observability.pricing.estimated_cost_usd", return_value=0.02),
            patch("graphrag.observability.cost_attribution.record_cost_event") as mock_record,
            patch.object(genai_telemetry, "_schedule_tenant_usage_record") as mock_schedule,
        ):
            with tenant_context("acme"):
                genai_telemetry._finish("openai", "chat", "success", 0.5, "openai", response)

        assert mock_record.call_count == 1
        event = mock_record.call_args.args[0]
        assert event.tenant == "acme"
        assert event.cost_usd == 0.02
        mock_schedule.assert_called_once_with("acme", 0.02)

    def test_cost_event_without_active_tenant_is_empty_and_not_billed(self):
        response = {"response_model": "gpt-test", "input_tokens": 10, "output_tokens": 5}
        with (
            patch("graphrag.observability.pricing.estimated_cost_usd", return_value=0.02),
            patch("graphrag.observability.cost_attribution.record_cost_event") as mock_record,
            patch.object(genai_telemetry, "_schedule_tenant_usage_record") as mock_schedule,
        ):
            genai_telemetry._finish("openai", "chat", "success", 0.5, "openai", response)

        event = mock_record.call_args.args[0]
        assert event.tenant == ""
        # No tenant to bill -- must not schedule a quota update for nobody.
        mock_schedule.assert_not_called()

    def test_zero_cost_does_not_schedule_a_quota_update(self):
        response = {"response_model": "gpt-test", "input_tokens": 0, "output_tokens": 0}
        with (
            patch("graphrag.observability.pricing.estimated_cost_usd", return_value=0.0),
            patch("graphrag.observability.cost_attribution.record_cost_event"),
            patch.object(genai_telemetry, "_schedule_tenant_usage_record") as mock_schedule,
        ):
            with tenant_context("acme"):
                genai_telemetry._finish("openai", "chat", "success", 0.5, "openai", response)

        mock_schedule.assert_not_called()


class TestScheduleTenantUsageRecord:
    def test_invokes_record_tenant_usage(self):
        async def run():
            with patch("api.quota.record_tenant_usage") as mock_usage:
                genai_telemetry._schedule_tenant_usage_record("acme", 0.42)
                await asyncio.sleep(0)  # let the scheduled task run
            return mock_usage

        mock_usage = asyncio.run(run())
        mock_usage.assert_called_once_with("acme", cost_usd=0.42)

    def test_tolerates_no_running_event_loop(self):
        # _finish() runs synchronously; if this were ever reached with no
        # running loop, it must not raise out of telemetry code.
        genai_telemetry._schedule_tenant_usage_record("acme", 0.42)
