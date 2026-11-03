"""Run every use-case charter in evals/charters and report against its thresholds.

    python scripts/run_charters.py

Exit code 1 when any charter misses a threshold or is malformed (for example
missing a label owner).  No network, database or model.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.context_graph.charters import CharterError, load_charter, run_charter  # noqa: E402


def main() -> int:
    failed = False
    for path in sorted((ROOT / "evals" / "charters").glob("*.yml")):
        try:
            result = run_charter(load_charter(path), root=ROOT)
        except CharterError as exc:
            print(f"REJECTED {path.name}: {exc}")
            failed = True
            continue
        failed |= not result["passed"]
        print(f"{'PASS' if result['passed'] else 'FAIL'} {result['charter']} ({result['split']}): "
              + json.dumps({k: (v["value"], v["denominator"]) for k, v in result["metrics"].items()}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
