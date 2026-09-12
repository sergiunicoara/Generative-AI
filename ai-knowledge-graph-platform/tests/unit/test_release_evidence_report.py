"""scripts/build_release_evidence_report.py: shape, commit stamping, and
graceful degradation when a source artifact is missing.

Mocks git/pip-audit/file reads rather than depending on this checkout's own
git state or a real pip-audit install, so this test is deterministic
regardless of what environment it runs in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import build_release_evidence_report as report_mod  # noqa: E402


class TestCommitEvidence:
    def test_reports_sha_branch_and_dirty_state_from_git(self):
        def fake_run(cmd):
            if cmd[1:] == ["rev-parse", "HEAD"]:
                return ("abc123def456" * 3 + "abcd", "", 0)  # 40 hex-ish chars
            if cmd[1:] == ["rev-parse", "--abbrev-ref", "HEAD"]:
                return ("main", "", 0)
            if cmd[1:] == ["status", "--porcelain"]:
                return ("M some/file.py", "", 0)
            if cmd[1:] == ["rev-parse", "--is-inside-work-tree"]:
                return ("true", "", 0)
            raise AssertionError(f"unexpected git invocation: {cmd}")

        with patch.object(report_mod, "_run", side_effect=fake_run):
            commit = report_mod.commit_evidence()

        assert commit["branch"] == "main"
        assert commit["dirty"] is True
        assert commit["short_sha"] == commit["sha"][:12]

    def test_falls_back_to_github_sha_env_when_git_is_unavailable(self, monkeypatch):
        monkeypatch.setenv("GITHUB_SHA", "envfallbacksha1234567890")
        monkeypatch.setenv("GITHUB_REF_NAME", "feature/x")
        with patch.object(report_mod, "_run", side_effect=FileNotFoundError()):
            commit = report_mod.commit_evidence()
        assert commit["sha"] == "envfallbacksha1234567890"
        assert commit["branch"] == "feature/x"
        assert commit["dirty"] is None  # cannot be determined without git

    def test_no_git_and_no_env_var_is_an_explicit_reason_not_a_crash(self, monkeypatch):
        monkeypatch.delenv("GITHUB_SHA", raising=False)
        with patch.object(report_mod, "_run", side_effect=FileNotFoundError()):
            commit = report_mod.commit_evidence()
        assert commit["sha"] is None
        assert "reason" in commit


class TestDependencyEvidence:
    def test_pip_audit_not_installed_is_an_explicit_reason(self):
        with patch.object(report_mod, "_run", side_effect=FileNotFoundError()):
            result = report_mod.dependency_evidence()
        assert result["scanned"] is False
        assert "pip-audit" in result["reason"]

    def test_pip_audit_finding_vulnerabilities_is_still_a_successful_scan(self):
        """pip-audit exits non-zero when it FINDS something -- that is a
        result to report, not a tool failure."""
        payload = json.dumps({
            "dependencies": [
                {"name": "bad-pkg", "version": "1.0", "vulns": [{"id": "CVE-2024-0001"}]},
                {"name": "good-pkg", "version": "2.0", "vulns": []},
            ],
        })
        with patch.object(report_mod, "_run", return_value=(payload, "", 1)):
            result = report_mod.dependency_evidence()
        assert result["scanned"] is True
        assert result["dependencies_scanned"] == 2
        assert result["vulnerable_dependencies"] == 1
        assert result["vulnerabilities"][0]["name"] == "bad-pkg"

    def test_unparseable_output_is_reported_not_raised(self):
        with patch.object(report_mod, "_run", return_value=("not json", "", 0)):
            result = report_mod.dependency_evidence()
        assert result["scanned"] is False
        assert "unparseable" in result["reason"]


class TestGracefulDegradationWhenAnArtifactIsMissing:
    def test_missing_benchmark_artifact_is_explicit_not_silent(self, tmp_path):
        missing = tmp_path / "does-not-exist.json"
        with patch.object(report_mod, "_BENCHMARK_PATH", missing):
            result = report_mod.benchmark_evidence()
        assert result["available"] is False
        assert "reason" in result

    def test_present_artifact_is_read_and_labelled_with_its_source(self, tmp_path):
        artifact = tmp_path / "recovery-exercise.json"
        artifact.write_text(json.dumps({"match": True}), encoding="utf-8")
        with patch.object(report_mod, "_RECOVERY_PATH", artifact), \
             patch.object(report_mod, "REPO_ROOT", tmp_path):
            result = report_mod.recovery_evidence()
        assert result["available"] is True
        assert result["match"] is True

    def test_malformed_json_artifact_is_reported_not_raised(self, tmp_path):
        artifact = tmp_path / "graphrag-benchmark-report.json"
        artifact.write_text("{not valid json", encoding="utf-8")
        with patch.object(report_mod, "_BENCHMARK_PATH", artifact):
            result = report_mod.benchmark_evidence()
        assert result["available"] is False
        assert "not valid JSON" in result["reason"]

    def test_digest_only_artifact_is_not_presented_as_database_recovery(self, tmp_path):
        artifact = tmp_path / "recovery-exercise.json"
        artifact.write_text(json.dumps({
            "database_recovery_proof": False, "match": True,
        }), encoding="utf-8")
        with patch.object(report_mod, "_RECOVERY_PATH", artifact):
            result = report_mod.recovery_evidence()
        assert result["available"] is False
        assert "file-integrity" in result["reason"]


class TestBuildReportShape:
    def test_report_carries_schema_version_and_claim_policy(self):
        with patch.object(report_mod, "commit_evidence", return_value={"sha": "x"}), \
             patch.object(report_mod, "dependency_evidence", return_value={"scanned": False}), \
             patch.object(report_mod, "live_test_evidence", return_value={"available": False}), \
             patch.object(report_mod, "benchmark_evidence", return_value={"available": False}), \
             patch.object(report_mod, "recovery_evidence", return_value={"available": False}):
            report = report_mod.build_report()

        assert report["report_schema_version"] == "release-evidence/v1"
        assert "claim_policy" in report
        assert "generated_at" in report
        assert set(report) >= {"commit", "dependencies", "live_test_results",
                                "benchmark_results", "recovery_results"}

    def test_main_writes_json_and_optional_markdown(self, tmp_path):
        out_json = tmp_path / "release-evidence.json"
        out_md = tmp_path / "release-evidence.md"
        with patch.object(report_mod, "build_report", return_value={
            "report_schema_version": "release-evidence/v1",
            "generated_at": "2026-01-01T00:00:00+00:00",
            "claim_policy": "test policy",
            "commit": {"short_sha": "abc123", "branch": "main", "dirty": False},
            "dependencies": {"scanned": False, "reason": "n/a"},
            "live_test_results": {"available": False, "reason": "n/a"},
            "benchmark_results": {"available": False, "reason": "n/a"},
            "recovery_results": {"available": False, "reason": "n/a"},
        }), patch.object(
            sys, "argv",
            ["build_release_evidence_report.py", "--output", str(out_json), "--markdown", str(out_md)],
        ):
            report_mod.main()

        assert out_json.exists()
        assert json.loads(out_json.read_text(encoding="utf-8"))["report_schema_version"] == "release-evidence/v1"
        assert out_md.exists()
        assert "abc123" in out_md.read_text(encoding="utf-8")
