#!/usr/bin/env python
"""Create the synthetic SAP-shaped SQLite export used by the energy R2RML demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Thin CLI wrapper -- the real fixture-building logic lives in
# graphrag/domains/energy/fixtures.py so graphrag/domains/energy/demo.py
# (library code) can build the identical fixture without importing from
# scripts/. Re-exported under the same name so every existing
# `from scripts.create_energy_demo_sqlite import create` call site keeps
# working unchanged.
from graphrag.domains.energy.fixtures import create_sap_fixture_sqlite as create  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/energy-demo-sap.sqlite"))
    args = parser.parse_args()
    create(args.output)
    print(f"Created synthetic SAP-shaped SQLite export: {args.output}")


if __name__ == "__main__":
    main()
