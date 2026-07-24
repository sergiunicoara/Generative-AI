"""Backfill Chunk.document_id from the existing PART_OF edge — REQUIRED before
the natural-key ingestion fix (see tasks/lessons.md A136) can run.

Background
----------
Chunk nodes have never carried a document_id property (measured: 0 of 102 on
the aerospace tenant despite schema.cypher declaring an index on it and
counterfactual.py querying it — that query has always matched nothing).

merge_chunk / merge_chunks_batch now MERGE on (document_id, chunk_index)
instead of a fresh uuid4() id, so any chunk without document_id cannot match
the new key and would be duplicated exactly once more on the next re-ingest —
the opposite of what the fix is for. This script must run once, before the
new ingestion code processes any document, on every existing deployment.

What it does
------------
    MATCH (c:Chunk)-[:PART_OF]->(d:Document) SET c.document_id = d.id

Idempotent — chunks that already carry the correct document_id are a no-op on
re-run (SET reassigns the same value).

Usage
-----
    python scripts/backfill_chunk_document_id.py                 # dry-run, all tenants
    python scripts/backfill_chunk_document_id.py --apply
    python scripts/backfill_chunk_document_id.py --tenant aerospace --apply

Exit codes
----------
  0  — nothing to backfill, or applied successfully
  1  — failed
  2  — dry-run found chunks that would be backfilled
"""

from __future__ import annotations

import argparse
import asyncio

import structlog

log = structlog.get_logger("backfill_chunk_document_id")


async def count_missing(neo4j, tenant: str | None) -> list[dict]:
    tenant_filter = "AND c.tenant = $tenant" if tenant else ""
    params = {"tenant": tenant} if tenant else {}
    return await neo4j.run(
        f"""
        MATCH (c:Chunk)-[:PART_OF]->(d:Document)
        WHERE (c.document_id IS NULL OR c.document_id <> d.id) {tenant_filter}
        RETURN coalesce(c.tenant, '') AS tenant, count(c) AS n
        ORDER BY tenant
        """,
        **params,
    )


async def backfill(neo4j, tenant: str | None) -> int:
    tenant_filter = "AND c.tenant = $tenant" if tenant else ""
    params = {"tenant": tenant} if tenant else {}
    rows = await neo4j.run(
        f"""
        MATCH (c:Chunk)-[:PART_OF]->(d:Document)
        WHERE (c.document_id IS NULL OR c.document_id <> d.id) {tenant_filter}
        SET c.document_id = d.id
        RETURN count(c) AS n
        """,
        **params,
    )
    return int(rows[0]["n"]) if rows else 0


async def main(args) -> int:
    from graphrag.graph.neo4j_client import get_neo4j

    neo4j = get_neo4j()
    tenant = args.tenant or None

    try:
        counts = await count_missing(neo4j, tenant)
    except Exception as exc:
        log.error("backfill_chunk_document_id.count_failed", error=str(exc))
        return 1

    total = sum(int(r["n"]) for r in counts)
    if not total:
        print("No chunks missing document_id — nothing to backfill.")
        return 0

    for row in counts:
        print(f"  {row['tenant'] or '(no tenant)':<16} {row['n']:>5} chunks missing document_id")
    print(f"  {'TOTAL':<16} {total:>5}")

    if not args.apply:
        print("\n[dry-run] Re-run with --apply to backfill document_id from PART_OF.")
        return 2

    try:
        updated = await backfill(neo4j, tenant)
    except Exception as exc:
        log.error("backfill_chunk_document_id.apply_failed", error=str(exc))
        return 1

    print(f"\nBackfilled document_id on {updated} chunks.")
    return 0


if __name__ == "__main__":
    import io
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Backfill Chunk.document_id from the PART_OF edge (required "
                    "before the natural-key ingestion fix)."
    )
    parser.add_argument("--tenant", default="", help="Limit to one tenant (default: all)")
    parser.add_argument("--apply", action="store_true",
                        help="Actually write document_id (default: dry-run)")
    sys.exit(asyncio.run(main(parser.parse_args())))
