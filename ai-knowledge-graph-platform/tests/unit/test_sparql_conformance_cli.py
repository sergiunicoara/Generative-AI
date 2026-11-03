"""scripts/sparql_conformance.py: exit codes, report output and credential handling, all offline."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("sparql_conformance_cli", ROOT / "scripts" / "sparql_conformance.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def test_local_baseline_exits_zero_and_prints_the_fingerprint(capsys):
    code = cli.main([])
    out = capsys.readouterr().out
    assert (code, "dataset fingerprint (sha256):" in out, "10/10 passed" in out) == (0, True, True)


def test_json_report_is_written_when_requested(tmp_path):
    target = tmp_path / "report.json"
    cli.main(["--json", str(target)])
    assert json.loads(target.read_text(encoding="utf-8"))["failed"] == 0


def test_a_mismatch_exits_non_zero(monkeypatch):
    suite = cli.load_suite()
    suite["cases"][0]["expected"] = ["Nothing"]
    monkeypatch.setattr(cli, "load_suite", lambda: suite)
    assert cli.main([]) == 1


def test_a_remote_target_without_domain_and_level_is_a_usage_error(capsys):
    code = cli.main(["--endpoint", "https://sparql.example.com/kg/query"])
    assert (code, "--domain and --level" in capsys.readouterr().err) == (2, True)


def test_endpoint_and_vendor_together_is_a_usage_error():
    args = cli._parse_args(["--endpoint", "https://a.example/q", "--vendor", "stardog"])
    with pytest.raises(cli.UsageError):
        cli._build_target(args)


def test_vendor_without_base_url_and_db_is_a_usage_error():
    args = cli._parse_args(["--vendor", "stardog", "--domain", "energy", "--level", "schema"])
    with pytest.raises(cli.UsageError):
        cli._build_target(args)


def test_missing_auth_env_var_is_a_usage_error(monkeypatch):
    monkeypatch.delenv("SPARQL_CONF_TEST_AUTH", raising=False)
    args = cli._parse_args([
        "--endpoint", "https://a.example/q", "--domain", "energy", "--level", "schema",
        "--auth-env", "SPARQL_CONF_TEST_AUTH",
    ])
    with pytest.raises(cli.UsageError, match="SPARQL_CONF_TEST_AUTH"):
        cli._build_target(args)


def test_the_target_label_never_contains_the_credentials(monkeypatch):
    monkeypatch.setenv("SPARQL_CONF_TEST_AUTH", "admin:s3cret")
    args = cli._parse_args([
        "--vendor", "stardog", "--base-url", "https://sd.example.com:5820", "--db", "kg",
        "--domain", "automotive", "--level", "instance", "--auth-env", "SPARQL_CONF_TEST_AUTH",
    ])
    source, label = cli._build_target(args)
    assert (label, source.query_url) == ("stardog:kg", "https://sd.example.com:5820/kg/query")


def test_endpoint_label_drops_userinfo_and_query_string():
    assert cli._endpoint_label("https://u:p@sd.example.com:5820/kg/query?token=abc") == "endpoint:sd.example.com/kg/query"
