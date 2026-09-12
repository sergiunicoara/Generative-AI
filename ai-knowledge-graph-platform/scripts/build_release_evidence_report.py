"""Aggregate commit, dependency, live-test, benchmark, and recovery evidence
into one release-evidence report.

Closes a gap a follow-up platform review named explicitly: a report linking
commit info, dependency info, live-test results, benchmark results, and
recovery/restore results. This repo already has a mature "evidence
artifact" convention (`report_schema_version` + `claim_policy` + a JSON
dump -- see graphrag/evidence/reports.py) and
scripts/build_public_local_evaluation_report.py is the direct precedent for
stitching several such JSON artifacts into one report. This script follows
the same shape and adds the two things nothing in the repo did before it:
commit-SHA stamping and a dependency/vulnerability scan (via `pip-audit`).

This aggregates existing evidence; it does not invent it. Every section is
independently optional -- a missing input becomes an explicit `null` with a
`reason` string, never a silently omitted key, matching
scripts/export_operational_evidence.py's convention. Run this after (not
instead of) the scripts that actually produce each artifact; see
docs/local-evidence-runbook.md.

Usage:
    python scripts/build_release_evidence_report.py --output artifacts/release-evidence.json
    python scripts/build_release_evidence_report.py --output artifacts/release-evidence.json --markdown docs/release-evidence.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Conventional artifact locations this script looks for -- see each
# producing script's own docstring/CLI for how to (re)generate them.
_RETRIEVAL_QUALITY_PATH = REPO_ROOT / "evals" / "retrieval_quality_last_run.json"
_GRAPH_FACT_GOLDEN_EVAL_PATH = REPO_ROOT / "artifacts" / "graph-fact-golden-eval.json"
_BENCHMARK_PATH = REPO_ROOT / "artifacts" / "graphrag-benchmark-report.json"
_RECOVERY_PATH = REPO_ROOT / "artifacts" / "recovery-exercise.json"


def _run(cmd: list[str]) -> tuple[str, str, int]:
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell, no untrusted input
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def _git(*args: str) -> str | None:
    try:
        stdout, _stderr, code = _run(["git", *args])
        return stdout if code == 0 and stdout else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def commit_evidence() -> dict[str, Any]:
    """Commit SHA, branch, and dirty-working-tree state.

    Prefers real `git` output; falls back to CI-provided env vars
    (`GITHUB_SHA`/`GITHUB_REF_NAME`) when this runs from an archive/tarball
    checkout with no `.git` directory (a real CI shape, not hypothetical --
    some artifact-upload/download steps do exactly this).
    """
    import os

    sha = _git("rev-parse", "HEAD") or os.getenv("GITHUB_SHA")
    if not sha:
        return {"sha": None, "short_sha": None, "branch": None, "dirty": None,
                "reason": "not a git checkout and no GITHUB_SHA in the environment"}
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or os.getenv("GITHUB_REF_NAME")
    status = _git("status", "--porcelain")
    return {
        "sha": sha,
        "short_sha": sha[:12],
        "branch": branch,
        # `status` is None both when the tree is clean AND when `git status`
        # itself failed (e.g. no .git dir, GITHUB_SHA fallback path) -- only
        # report `dirty` when we actually ran the command successfully.
        "dirty": bool(status) if _git("rev-parse", "--is-inside-work-tree") else None,
    }


def dependency_evidence() -> dict[str, Any]:
    """`pip-audit --format json` output, or an explicit reason it's absent.

    `pip-audit` exits non-zero when it *finds* vulnerabilities -- that is a
    result, not a failure, so only a missing binary or unparseable output is
    treated as "unavailable".
    """
    try:
        stdout, stderr, _code = _run(["pip-audit", "--format", "json"])
    except FileNotFoundError:
        return {"scanned": False, "reason": "pip-audit is not installed"}
    except subprocess.TimeoutExpired:
        return {"scanned": False, "reason": "pip-audit timed out"}
    try:
        payload = json.loads(stdout)
    except ValueError:
        return {"scanned": False, "reason": f"pip-audit produced unparseable output: {stderr[:500]}"}
    dependencies = payload if isinstance(payload, list) else payload.get("dependencies", [])
    vulnerable = [dep for dep in dependencies if dep.get("vulns")]
    return {
        "scanned": True,
        "tool": "pip-audit",
        "dependencies_scanned": len(dependencies),
        "vulnerable_dependencies": len(vulnerable),
        "vulnerabilities": [
            {"name": dep.get("name"), "version": dep.get("version"),
             "vuln_ids": [v.get("id") for v in dep.get("vulns", [])]}
            for dep in vulnerable
        ],
    }


def _display_path(path: Path) -> str:
    """`path` relative to REPO_ROOT when possible, else its absolute form.

    A plain `relative_to()` raises when `path` isn't actually under
    REPO_ROOT (true for any test-injected path, and plausible in a real
    deployment that configures an artifact location outside the repo) --
    this is meant only for a human-readable message, so it must never be
    the thing that turns "artifact not found" into an unhandled crash.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_json_if_exists(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "reason": f"{label} not found at {_display_path(path)}"}
    try:
        return {"available": True, "source": _display_path(path),
                **json.loads(path.read_text(encoding="utf-8"))}
    except ValueError as exc:
        return {"available": False, "reason": f"{label} at {path} is not valid JSON: {exc}"}


def live_test_evidence() -> dict[str, Any]:
    if _RETRIEVAL_QUALITY_PATH.exists():
        return _read_json_if_exists(_RETRIEVAL_QUALITY_PATH, label="retrieval-quality live-test result")
    return _read_json_if_exists(_GRAPH_FACT_GOLDEN_EVAL_PATH, label="graph-fact golden-eval live-test result")


def benchmark_evidence() -> dict[str, Any]:
    return _read_json_if_exists(_BENCHMARK_PATH, label="GraphRAG benchmark result")


def recovery_evidence() -> dict[str, Any]:
    # A legacy artifact at this path can only be a file-digest comparison; it
    # must not be mistaken for database recovery evidence. Live recovery is
    # exercised by the Docker-backed GraphDB/Neo4j e2e tests documented in
    # docs/local-evidence-runbook.md.
    result = _read_json_if_exists(_RECOVERY_PATH, label="database recovery result")
    if result.get("available") and result.get("database_recovery_proof") is False:
        return {
            "available": False,
            "reason": "artifact is a file-integrity check, not database recovery evidence",
        }
    return result


def build_report() -> dict[str, Any]:
    return {
        "report_schema_version": "release-evidence/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_policy": (
            "This report aggregates evidence already produced by other scripts in this "
            "repo; it does not itself run or re-verify any of them. A `null`/`available: "
            "false` section means that evidence was not found at its conventional path, "
            "not that the underlying check failed. See docs/local-evidence-runbook.md for "
            "how to produce each section's source artifact."
        ),
        "commit": commit_evidence(),
        "dependencies": dependency_evidence(),
        "live_test_results": live_test_evidence(),
        "benchmark_results": benchmark_evidence(),
        "recovery_results": recovery_evidence(),
    }


def _markdown_summary(report: dict[str, Any]) -> str:
    commit = report["commit"]
    lines = [
        "# Release Evidence Report", "",
        f"Generated: {report['generated_at']}", "",
        f"- Commit: `{commit.get('short_sha') or 'unknown'}`"
        f" ({commit.get('branch') or 'unknown branch'}"
        f"{', dirty working tree' if commit.get('dirty') else ''})",
        "",
        "## Sections", "",
        "| Section | Available |",
        "|---|---|",
    ]
    for key in ("dependencies", "live_test_results", "benchmark_results", "recovery_results"):
        section = report[key]
        available = section.get("scanned", section.get("available"))
        lines.append(f"| {key.replace('_', ' ')} | {'yes' if available else 'no — ' + section.get('reason', 'not found')} |")
    lines += ["", "## Claim policy", "", report["claim_policy"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown_summary(report), encoding="utf-8")
        print(args.markdown)


if __name__ == "__main__":
    main()
