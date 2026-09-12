"""Energy evaluation report: measured local evidence and claim boundaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import build_energy_evaluation_report as report_mod  # noqa: E402


def test_report_scores_the_fixed_energy_cases_and_required_dimensions():
    report = report_mod.build_report(latency_iterations=1, ingestion_iterations=1)

    assert report["report_schema_version"] == "energy-evaluation/v1"
    assert report["synthetic"] is True
    assert report["answer_correctness"]["passed"] == 5
    assert report["answer_correctness"]["total"] == 5
    assert report["evidence_accuracy"]["passed"] is True
    assert report["abstention"]["passed"] is True
    assert report["tenant_isolation"]["passed"] is True
    assert report["freshness"]["quarantined_count"] == 0
    assert report["latency"]["sample_count"] == 5
    assert report["ingestion_throughput"]["candidate_records"] > 0
    assert "not an enterprise ingestion-capacity benchmark" in report["ingestion_throughput"]["claim_policy"]


def test_latency_failure_is_explicit_and_does_not_hide_the_other_evidence():
    with patch.object(report_mod, "measure_answer_latency", side_effect=RuntimeError("clock unavailable")):
        report = report_mod.build_report(latency_iterations=1, ingestion_iterations=1)

    assert report["latency"]["available"] is False
    assert "clock unavailable" in report["latency"]["reason"]
    assert report["answer_correctness"]["passed"] == 5


def test_main_writes_the_versioned_json_artifact(tmp_path):
    output = tmp_path / "energy-evaluation.json"
    expected = {"report_schema_version": "energy-evaluation/v1"}
    with patch.object(report_mod, "build_report", return_value=expected), patch.object(
        sys,
        "argv",
        [
            "build_energy_evaluation_report.py", "--output", str(output),
            "--latency-iterations", "1", "--ingestion-iterations", "1",
        ],
    ):
        report_mod.main()

    assert json.loads(output.read_text(encoding="utf-8")) == expected
