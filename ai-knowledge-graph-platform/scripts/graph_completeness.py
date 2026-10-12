#!/usr/bin/env python
"""Offline, read-only graph data-completeness report for one tenant (no LLM, no cost).

    python scripts/graph_completeness.py --tenant aerospace [--source-id S] [--out report.json]

Shares of documents ingested / chunked / linked to entities, chunks embedded, entities described / embedded / typed /
connected, relations sourced / dated / scored. See graphrag/graph/completeness.py. Needs a reachable Neo4j.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.graph.completeness import graph_completeness_report  # noqa: E402
from graphrag.graph.neo4j_client import get_neo4j  # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--source-id")
    ap.add_argument("--out", help="also write the JSON report to this path")
    args = ap.parse_args(argv)
    report = asyncio.run(graph_completeness_report(get_neo4j(), args.tenant, args.source_id))
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
