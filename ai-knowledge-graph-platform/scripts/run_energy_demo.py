#!/usr/bin/env python
"""Run the deterministic Energy Asset & Maintenance Intelligence POC.

Usage:
  python scripts/run_energy_demo.py
  python scripts/run_energy_demo.py --as-of 2026-05-01T00:00:00Z
  python scripts/run_energy_demo.py --export-turtle artifacts/energy-demo.ttl

The output is a synthetic, advisory demonstration.  It never controls assets
or sends maintenance actions to external systems.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.domains.energy.demo import EnergyDemoService, TENANT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default=None, help="Historical ISO-8601 instant")
    parser.add_argument("--export-turtle", type=Path, help="Write generated RDF Turtle to this path")
    parser.add_argument("--source-sqlite", type=Path, help="Materialize the R2RML-mapped SQLite source into the RDF graph")
    args = parser.parse_args()
    service = EnergyDemoService(source_db=args.source_sqlite)
    if args.export_turtle:
        args.export_turtle.parent.mkdir(parents=True, exist_ok=True)
        args.export_turtle.write_text(service.export_turtle(), encoding="utf-8")
        print(f"Wrote RDF Turtle: {args.export_turtle}")
    for question_id, question in service.questions.items():
        result = service.answer(question_id, tenant=TENANT, as_of=args.as_of)
        print(f"\n{question}\n{json.dumps(result, indent=2)}")
    print(f"\nInvalid-batch validation: {json.dumps(service.validate_candidate())}")


if __name__ == "__main__":
    main()
