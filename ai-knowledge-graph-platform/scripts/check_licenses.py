"""Fail if a configured model has no allowed-licence entry in config/licenses.yml. Offline; no cost.

    python scripts/check_licenses.py [--strict]     # --strict also fails on entries still `needs_review`
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphrag.core.license_register import check, find_model_ids, load_register  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = check(load_register(), find_model_ids())
    print(json.dumps(result, indent=2))
    if not result["ok"] or (args.strict and result["needs_review"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
