#!/usr/bin/env python
"""IATF 16949 clause evidence from the automotive corpus. Offline: no LLM, no API, no Neo4j, no cost.

    python scripts/iatf_audit.py coverage [--as-of 2026-06-13T10:00:00]
    python scripts/iatf_audit.py freshness [--fail-on missing,stale,only_superseded,conflict]
    python scripts/iatf_audit.py export --format csv --out audit.csv [--as-of ...]

``freshness`` exits 1 when any clause has a problem of a kind listed in --fail-on, so a scheduler (cron, CI, a k8s
CronJob) can alert on it. Sign-offs live in Neo4j and are included by the API export (GET /audit/iatf/export), not
by this offline command.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.domains.automotive.audit import (  # noqa: E402
    audit_view,
    export_rows,
    freshness_problems,
    to_csv,
)

DEFAULT_FAIL_ON = "missing,stale,only_superseded"


def _emit(text: str, out: str | None) -> None:
    if out:
        Path(out).write_text(text, encoding="utf-8")
    sys.stdout.buffer.write((text if text.endswith("\n") else text + "\n").encode("utf-8"))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["coverage", "freshness", "export"])
    ap.add_argument("--as-of", help="ISO timestamp; documents recorded later are ignored")
    ap.add_argument("--fail-on", default=DEFAULT_FAIL_ON,
                    help=f"freshness: comma list of missing,stale,only_superseded,conflict (default {DEFAULT_FAIL_ON})")
    ap.add_argument("--format", choices=["json", "csv"], default="json")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    view = audit_view(as_of=args.as_of)

    if args.command == "coverage":
        summary = {k: view[k] for k in ("standard", "mapping_status", "as_of", "documents_considered", "counts",
                                        "coverage", "disclaimer")}
        summary["clauses"] = [{"clause": c["clause"], "status": c["status"],
                               "evidence": [e["doc_id"] for e in c["evidence"]],
                               "conflict_tags": [x["tag"] for x in c["conflicts"]]} for c in view["clauses"]]
        _emit(json.dumps(summary, indent=2, ensure_ascii=False), args.out)
        return 0

    if args.command == "freshness":
        fail_on = {s.strip() for s in args.fail_on.split(",") if s.strip()}
        problems = freshness_problems(view)
        failing = [p for p in problems if p["problem"] in fail_on]
        _emit(json.dumps({"problems": problems, "failing": failing, "fail_on": sorted(fail_on)}, indent=2,
                         ensure_ascii=False), args.out)
        return 1 if failing else 0

    rows = export_rows(view)
    _emit(to_csv(rows) if args.format == "csv" else json.dumps({"view": view, "rows": rows}, indent=2,
                                                                ensure_ascii=False), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
