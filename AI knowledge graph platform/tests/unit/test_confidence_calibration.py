"""Unit tests for CalibrationService — Brier score, calibration curve, apply_calibration."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graphrag.graph.confidence_calibration import CALIBRATION_BINS, CalibrationService


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture
def neo4j_mock():
    return AsyncMock()


@pytest.fixture
def svc(neo4j_mock):
    return CalibrationService(neo4j_client=neo4j_mock)


# ── add_sample ─────────────────────────────────────────────────────────────────

class TestAddSample:
    async def test_returns_string_id(self, svc, neo4j_mock):
        neo4j_mock.run = AsyncMock(return_value=[])
        sid = await svc.add_sample(predicted_confidence=0.8, actual_outcome=1.0)
        assert isinstance(sid, str)
        assert len(sid) == 36   # UUID4

    async def test_clamps_predicted_to_0_1(self, svc, neo4j_mock):
        """Values outside [0,1] must be clamped before persisting."""
        neo4j_mock.run = AsyncMock(return_value=[])
        await svc.add_sample(predicted_confidence=1.5, actual_outcome=-0.5)
        call_kwargs = neo4j_mock.run.call_args[1]
        assert call_kwargs["predicted"] == pytest.approx(1.0)
        assert call_kwargs["actual"] == pytest.approx(0.0)

    async def test_error_sq_computed_correctly(self, svc, neo4j_mock):
        neo4j_mock.run = AsyncMock(return_value=[])
        await svc.add_sample(predicted_confidence=0.7, actual_outcome=1.0)
        call_kwargs = neo4j_mock.run.call_args[1]
        expected = (0.7 - 1.0) ** 2
        assert call_kwargs["error_sq"] == pytest.approx(expected)

    async def test_tenant_passed_to_neo4j(self, svc, neo4j_mock):
        neo4j_mock.run = AsyncMock(return_value=[])
        await svc.add_sample(0.5, 1.0, tenant="finance")
        assert neo4j_mock.run.call_args[1]["tenant"] == "finance"


# ── add_batch ──────────────────────────────────────────────────────────────────

class TestAddBatch:
    async def test_returns_correct_number_of_ids(self, svc, neo4j_mock):
        neo4j_mock.run = AsyncMock(return_value=[])
        samples = [
            {"predicted_confidence": 0.9, "actual_outcome": 1.0},
            {"predicted_confidence": 0.3, "actual_outcome": 0.0},
            {"predicted_confidence": 0.6, "actual_outcome": 1.0},
        ]
        ids = await svc.add_batch(samples, tenant="acme")
        assert len(ids) == 3
        assert all(isinstance(i, str) for i in ids)
        assert len(set(ids)) == 3   # all unique

    async def test_optional_fields_default_correctly(self, svc, neo4j_mock):
        neo4j_mock.run = AsyncMock(return_value=[])
        await svc.add_batch([{"predicted_confidence": 0.5, "actual_outcome": 0.0}])
        # Should not raise; optional keys should have defaults


# ── brier_score ────────────────────────────────────────────────────────────────

class TestBrierScore:
    async def test_no_samples_returns_negative_one(self, svc, neo4j_mock):
        neo4j_mock.run = AsyncMock(return_value=[{"brier": None, "n": 0}])
        score = await svc.brier_score(tenant="acme")
        assert score == pytest.approx(-1.0)

    async def test_empty_rows_returns_negative_one(self, svc, neo4j_mock):
        neo4j_mock.run = AsyncMock(return_value=[])
        score = await svc.brier_score()
        assert score == pytest.approx(-1.0)

    async def test_perfect_predictions_returns_zero(self, svc, neo4j_mock):
        """Perfect calibration (predicted == actual) → Brier score = 0."""
        neo4j_mock.run = AsyncMock(return_value=[{"brier": 0.0, "n": 10}])
        score = await svc.brier_score()
        assert score == pytest.approx(0.0)

    async def test_uniform_50_percent_returns_quarter(self, svc, neo4j_mock):
        """Uniform 0.5 predictions → Brier = 0.25 (uninformative baseline)."""
        neo4j_mock.run = AsyncMock(return_value=[{"brier": 0.25, "n": 100}])
        score = await svc.brier_score()
        assert score == pytest.approx(0.25)

    async def test_score_is_rounded_to_6_places(self, svc, neo4j_mock):
        neo4j_mock.run = AsyncMock(return_value=[{"brier": 0.123456789, "n": 5}])
        score = await svc.brier_score()
        assert score == pytest.approx(round(0.123456789, 6))


# ── calibration_curve ─────────────────────────────────────────────────────────

class TestCalibrationCurve:
    async def test_returns_correct_number_of_bins(self, svc, neo4j_mock):
        neo4j_mock.run = AsyncMock(return_value=[])
        curve = await svc.calibration_curve()
        assert len(curve) == CALIBRATION_BINS

    async def test_bin_bounds_span_zero_to_one(self, svc, neo4j_mock):
        neo4j_mock.run = AsyncMock(return_value=[])
        curve = await svc.calibration_curve()
        assert curve[0]["bin_start"] == pytest.approx(0.0)
        assert curve[-1]["bin_end"] == pytest.approx(1.0)

    async def test_empty_bin_has_zero_count(self, svc, neo4j_mock):
        neo4j_mock.run = AsyncMock(return_value=[])
        curve = await svc.calibration_curve()
        assert all(b["n"] == 0 for b in curve)

    async def test_samples_placed_in_correct_bin(self, svc, neo4j_mock):
        # One sample at predicted=0.75 → should land in bin [0.7, 0.8] (index 7)
        neo4j_mock.run = AsyncMock(return_value=[
            {"predicted": 0.75, "actual": 1.0}
        ])
        curve = await svc.calibration_curve(bins=10)
        bin7 = next(b for b in curve if b["bin_start"] == pytest.approx(0.7))
        assert bin7["n"] == 1
        assert bin7["mean_predicted"] == pytest.approx(0.75)
        assert bin7["mean_actual"] == pytest.approx(1.0)

    async def test_calibration_gap_is_predicted_minus_actual(self, svc, neo4j_mock):
        neo4j_mock.run = AsyncMock(return_value=[
            {"predicted": 0.85, "actual": 0.5}
        ])
        curve = await svc.calibration_curve(bins=10)
        populated = [b for b in curve if b["n"] > 0]
        assert len(populated) == 1
        gap = populated[0]["calibration_gap"]
        assert gap == pytest.approx(0.85 - 0.5, abs=0.001)

    async def test_multiple_samples_averaged_per_bin(self, svc, neo4j_mock):
        # Two samples in [0.2, 0.3] bin
        neo4j_mock.run = AsyncMock(return_value=[
            {"predicted": 0.21, "actual": 0.0},
            {"predicted": 0.29, "actual": 1.0},
        ])
        curve = await svc.calibration_curve(bins=10)
        bin2 = next(b for b in curve if b["bin_start"] == pytest.approx(0.2))
        assert bin2["n"] == 2
        assert bin2["mean_actual"] == pytest.approx(0.5)


# ── apply_calibration ──────────────────────────────────────────────────────────

class TestApplyCalibration:
    async def test_returns_mean_actual_for_populated_bin(self, svc, neo4j_mock):
        """Raw confidence should be replaced by the mean_actual of its bin."""
        neo4j_mock.run = AsyncMock(return_value=[
            {"predicted": 0.82, "actual": 0.6}
        ])
        calibrated = await svc.apply_calibration(0.82, tenant="acme")
        assert calibrated == pytest.approx(0.6, abs=0.01)

    async def test_returns_raw_confidence_for_empty_bin(self, svc, neo4j_mock):
        """If the bin has no samples, raw confidence should be returned unchanged."""
        neo4j_mock.run = AsyncMock(return_value=[])  # no samples at all
        calibrated = await svc.apply_calibration(0.55)
        assert calibrated == pytest.approx(0.55)

    async def test_result_is_clamped_to_0_1_range(self, svc, neo4j_mock):
        """Calibrated values should stay in [0, 1]."""
        neo4j_mock.run = AsyncMock(return_value=[
            {"predicted": 0.5, "actual": 0.7}
        ])
        calibrated = await svc.apply_calibration(0.5)
        assert 0.0 <= calibrated <= 1.0


# ── calibration_summary ────────────────────────────────────────────────────────

class TestCalibrationSummary:
    async def test_required_fields_present(self, svc, neo4j_mock):
        neo4j_mock.run = AsyncMock(return_value=[{"brier": 0.08, "n": 50}])
        summary = await svc.calibration_summary()
        for field in ("brier_score", "sample_count", "max_calibration_gap",
                      "over_confident_bins", "under_confident_bins",
                      "calibration_curve", "verdict"):
            assert field in summary, f"Missing field: {field}"

    async def test_well_calibrated_verdict(self, svc, neo4j_mock):
        # brier < 0.10 → "well-calibrated"
        neo4j_mock.run = AsyncMock(return_value=[{"brier": 0.05, "n": 100}])
        summary = await svc.calibration_summary()
        assert summary["verdict"] == "well-calibrated"

    async def test_no_samples_verdict(self, svc, neo4j_mock):
        # No samples: brier = -1.0 → falls into "needs-review"
        neo4j_mock.run = AsyncMock(return_value=[{"brier": None, "n": 0}])
        summary = await svc.calibration_summary()
        # Just assert it doesn't crash and returns a string verdict
        assert isinstance(summary["verdict"], str)
