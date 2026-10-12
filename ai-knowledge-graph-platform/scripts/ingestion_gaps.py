#!/usr/bin/env python
"""Read-only ingestion-gap report: what a source has that the graph is missing, holds stale, or never finished.

    python scripts/ingestion_gaps.py --tenant acme --source-id sharepoint --inventory inventory.json

``inventory.json`` is either ``{"external_id": "version", ...}`` or a bare list of external ids: the listing a
connector already produces for ``reconcile`` (ids and versions only, no content), so this adds no load on the source.
Nothing is written. ``repair_ids`` in the output is what a connector should re-fetch. Needs a reachable Neo4j.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.enterprise.gaps import find_ingestion_gaps  # noqa: E402
from graphrag.graph.neo4j_client import get_neo4j  # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--inventory", required=True, help="JSON file: {id: version} or [id, ...]")
    ap.add_argument("--out", help="also write the JSON report to this path")
    args = ap.parse_args(argv)
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    report = asyncio.run(find_ingestion_gaps(
        get_neo4j(), tenant=args.tenant, source_id=args.source_id, inventory=inventory))
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
