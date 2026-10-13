#!/usr/bin/env python
"""Write or verify a signed checkpoint of a tenant's IATF sign-off chain head. No LLM, no cost; needs Neo4j.

    IATF_CHECKPOINT_KEY=... python scripts/iatf_checkpoint.py write  --tenant automotive --path /mnt/worm/iatf.jsonl
    IATF_CHECKPOINT_KEY=... python scripts/iatf_checkpoint.py verify --tenant automotive --path /mnt/worm/iatf.jsonl

``write`` refuses to checkpoint a chain that does not verify. ``verify`` exits 1 when the chain or any checkpoint
fails. Schedule ``write`` (for example daily) and ``verify`` (for example hourly). See docs/iatf-audit.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.domains.automotive.checkpoints import (  # noqa: E402
    append_checkpoint,
    make_checkpoint,
    read_checkpoints,
    verify_against_checkpoints,
)
from graphrag.domains.automotive.signoff import SignoffStore  # noqa: E402
from graphrag.graph.neo4j_client import get_neo4j  # noqa: E402


async def run(command: str, tenant: str, path: Path, neo4j=None) -> tuple[int, dict]:
    store = SignoffStore(neo4j or get_neo4j())
    entries = await store.entries(tenant)
    chain = await store.verify(tenant)
    if command == "write":
        if not chain["valid"]:
            return 1, {"written": False, "reason": "chain does not verify", "chain": chain}
        record = make_checkpoint(tenant=tenant, seq=chain["entries"], head=chain["head"])
        append_checkpoint(path, record)
        return 0, {"written": True, "checkpoint": record}
    anchored = verify_against_checkpoints(entries, read_checkpoints(path, tenant))
    ok = chain["valid"] and anchored["valid"]
    return (0 if ok else 1), {"chain": chain, "checkpoints": anchored}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["write", "verify"])
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--path", required=True, type=Path)
    args = ap.parse_args(argv)
    code, report = asyncio.run(run(args.command, args.tenant, args.path))
    print(json.dumps(report, indent=2, default=str))
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
