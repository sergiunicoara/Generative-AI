#!/usr/bin/env python
"""Evaluate the labelled synthetic Energy Asset & Maintenance Intelligence cases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.domains.energy.demo import EnergyDemoService, TENANT  # noqa: E402


def main() -> None:
    cases = json.loads((ROOT / "evals/energy_demo/questions.json").read_text(encoding="utf-8"))["questions"]
    service = EnergyDemoService()
    results = []
    for case in cases:
        response = service.answer(case["id"], tenant=TENANT, as_of=case.get("as_of"))
        response_text = str(response)
        expected_values = [
            item
            for key, value in case.items() if key.startswith("expected_")
            for item in (value if isinstance(value, list) else [value])
        ]
        passed = all(str(value) in response_text for value in expected_values)
        results.append({"id": case["id"], "passed": passed, "status": response["status"]})
    report = {"dataset": "energy-demo/v1", "cases": results, "passed": sum(item["passed"] for item in results), "total": len(results)}
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
