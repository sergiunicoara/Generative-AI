"""Run the Energy POC workflow and save its real command transcript for video rendering."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/presentation/energy_demo_real_run.json"

WORKFLOW = [
    ("create_source", [sys.executable, "scripts/create_energy_demo_sqlite.py"]),
    (
        "validate_r2rml",
        [
            sys.executable, "scripts/ingest_r2rml.py",
            "--mapping", "ontology/mappings/energy-assets.r2rml.ttl",
            "--sqlite", "artifacts/energy-demo-sap.sqlite",
            "--tenant", "energy-demo",
            "--source-id", "synthetic-sap",
            "--validate-only",
        ],
    ),
    ("run_demo", [sys.executable, "scripts/run_energy_demo.py", "--export-turtle", "artifacts/energy-demo.ttl"]),
    ("run_historical", [sys.executable, "scripts/run_energy_demo.py", "--as-of", "2026-05-01T00:00:00Z"]),
    ("evaluate", [sys.executable, "scripts/evaluate_energy_demo.py"]),
    ("unit_tests", [sys.executable, "-m", "pytest", "tests/unit/test_energy_demo.py", "-q"]),
]


def main() -> None:
    commands: dict[str, dict[str, object]] = {}
    for name, command in WORKFLOW:
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        commands[name] = {
            "command": " ".join(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    trace = {
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repository": str(ROOT),
        "commands": commands,
        "all_succeeded": all(item["returncode"] == 0 for item in commands.values()),
    }
    OUT.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(OUT)
    raise SystemExit(0 if trace["all_succeeded"] else 1)


if __name__ == "__main__":
    main()
