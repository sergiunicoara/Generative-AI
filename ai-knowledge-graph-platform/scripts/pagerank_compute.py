"""Standalone PageRank compute script — runs GDS PageRank and prints top entities.

Usage
-----
    # Compute for one tenant:
    python scripts/pagerank_compute.py --tenant automotive

    # Compute for all tenants that have entities:
    python scripts/pagerank_compute.py --tenant all

Exit codes
----------
  0  — completed successfully
  1  — failed (exception)
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import structlog

# Windows console defaults to cp1252, which can't encode Romanian diacritics
# (ț, ș, etc.) present in entity names — reconfigure stdout to UTF-8.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

log = structlog.get_logger("pagerank_compute")


async def get_all_tenants(neo4j_client) -> list[str]:
    rows = await neo4j_client.run(
        "MATCH (e:Entity) RETURN DISTINCT e.tenant AS tenant ORDER BY tenant"
    )
    return [r["tenant"] for r in rows if r.get("tenant")]


async def compute_tenant(tenant: str) -> dict:
    from graphrag.graph.pagerank import PageRankComputer
    from graphrag.graph.neo4j_client import get_neo4j

    neo4j_client = get_neo4j()
    report = await PageRankComputer(tenant=tenant).compute_and_persist()

    top_rows = await neo4j_client.get_top_entities_by_pagerank(tenant=tenant, top_k=10)
    print(f"\n=== PageRank top entities — tenant: {tenant} ===")
    if not top_rows:
        print("  (no entities scored)")
    for i, row in enumerate(top_rows, start=1):
        print(f"  {i:2}. {row['name']:<45} [{row['type']}]  score={row['score']:.5f}")

    return report


async def main(args: argparse.Namespace) -> int:
    from graphrag.graph.neo4j_client import get_neo4j

    neo4j_client = get_neo4j()

    if args.tenant == "all":
        tenants = await get_all_tenants(neo4j_client)
        if not tenants:
            tenants = ["default"]
    else:
        tenants = [args.tenant]

    log.info("pagerank_compute.run_start", tenants=tenants)

    exit_code = 0
    try:
        for tenant in tenants:
            try:
                await compute_tenant(tenant)
            except Exception as exc:
                log.error("pagerank_compute.tenant_failed", tenant=tenant, error=str(exc))
                print(f"[error] tenant={tenant}: {exc}")
                exit_code = 1
    finally:
        await neo4j_client.close()

    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute PageRank centrality for one or all tenants."
    )
    parser.add_argument(
        "--tenant",
        default="default",
        help="Tenant to compute (default: 'default'; use 'all' for every tenant with entities)",
    )
    parsed = parser.parse_args()

    # Suppress neo4j driver deprecation notifications (GDS Cypher projection API)
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import logging
    logging.getLogger("neo4j").setLevel(logging.ERROR)

    loop = asyncio.new_event_loop()
    try:
        exit_code = loop.run_until_complete(main(parsed))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
    sys.exit(exit_code)
