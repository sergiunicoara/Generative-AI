"""Recompute derived artifacts waiting in NEEDS_REVIEW, and sweep expired evidence.

  python scripts/recompute_stale.py --tenant acme               # recompute up to --limit artifacts
  python scripts/recompute_stale.py --tenant acme --sweep-expiry # first turn passed valid_to into events

Safe to run on a schedule: both steps are idempotent (docs/invalidation.md).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from graphrag.graph.invalidation.expiry import sweep_expired
from graphrag.graph.invalidation.recompute import RecomputeWorker
from graphrag.graph.neo4j_client import get_neo4j


async def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--sweep-expiry", action="store_true")
    args = ap.parse_args(argv)
    neo4j = get_neo4j()
    out = {}
    if args.sweep_expiry:
        out["expiry"] = await sweep_expired(neo4j, args.tenant)
    out["recompute"] = await RecomputeWorker(neo4j).run_once(args.tenant, limit=args.limit)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
