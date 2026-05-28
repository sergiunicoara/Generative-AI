"""Standalone community rebuild script — decoupled from the ingestion path.

Problem solved
--------------
Previously community detection (Leiden algorithm) ran inline inside the
ingestion worker, blocking document processing and consuming GNN/embedding
resources at write time.  Large graphs made ingestion unpredictably slow.

This script runs as a separate process (cron job, Celery task, or manual
invocation) and rebuilds communities only when the staleness score exceeds
the configured threshold.

Usage
-----
    # Rebuild all tenants (default):
    python scripts/community_rebuild.py

    # Rebuild a specific tenant only:
    python scripts/community_rebuild.py --tenant aerospace

    # Force rebuild even if graph is not stale:
    python scripts/community_rebuild.py --force

    # Dry-run: check staleness without rebuilding:
    python scripts/community_rebuild.py --dry-run

Exit codes
----------
  0  — completed successfully (rebuilt or not stale)
  1  — rebuild failed (exception)
  2  — dry-run found stale graph (rebuild needed, none performed)
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import structlog

log = structlog.get_logger("community_rebuild")


async def rebuild_tenant(
    tenant: str,
    neo4j_client,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Check staleness and conditionally rebuild communities for one tenant.

    Returns a report dict with keys: tenant, stale, rebuilt, community_count.
    """
    from graphrag.graph.community_builder import CommunityBuilder
    from graphrag.graph.community_manager import CommunityManager
    from graphrag.graph.community_summarizer import CommunitySummarizer
    from graphrag.graph.neo4j_client import Neo4jClient

    manager = CommunityManager(neo4j_client)
    stale_report = await manager.check_staleness(tenant=tenant)

    is_stale = stale_report.get("should_rebuild", False)
    staleness = stale_report.get("staleness_score", 0.0)

    log.info(
        "community_rebuild.staleness_check",
        tenant=tenant,
        staleness=round(staleness, 4),
        should_rebuild=is_stale,
    )

    if dry_run:
        return {
            "tenant": tenant,
            "stale": is_stale,
            "staleness": staleness,
            "rebuilt": False,
            "community_count": 0,
            "dry_run": True,
        }

    if not is_stale and not force:
        log.info(
            "community_rebuild.skipped",
            tenant=tenant,
            reason="graph not stale",
        )
        return {
            "tenant": tenant,
            "stale": False,
            "staleness": staleness,
            "rebuilt": False,
            "community_count": 0,
        }

    log.info(
        "community_rebuild.starting",
        tenant=tenant,
        forced=force,
        staleness=round(staleness, 4),
    )

    builder    = CommunityBuilder(tenant=tenant)
    summarizer = CommunitySummarizer()

    communities = await builder.build()
    if communities:
        communities = await summarizer.summarize_all(communities)
        for community in communities:
            await neo4j_client.merge_community(community)

    await manager.mark_rebuilt(tenant=tenant)

    log.info(
        "community_rebuild.done",
        tenant=tenant,
        community_count=len(communities),
    )
    return {
        "tenant": tenant,
        "stale": True,
        "staleness": staleness,
        "rebuilt": True,
        "community_count": len(communities),
    }


async def get_all_tenants(neo4j_client) -> list[str]:
    """Return all distinct tenants that have at least one Entity node."""
    rows = await neo4j_client.run(
        "MATCH (e:Entity) RETURN DISTINCT e.tenant AS tenant ORDER BY tenant"
    )
    return [r["tenant"] for r in rows if r.get("tenant")]


async def main(args: argparse.Namespace) -> int:
    from graphrag.graph.neo4j_client import get_neo4j

    neo4j_client = get_neo4j()

    # Determine tenants to process
    if args.tenant:
        tenants = [args.tenant]
    else:
        tenants = await get_all_tenants(neo4j_client)
        if not tenants:
            tenants = ["default"]

    log.info("community_rebuild.run_start", tenants=tenants, force=args.force, dry_run=args.dry_run)

    reports = []
    exit_code = 0

    for tenant in tenants:
        try:
            report = await rebuild_tenant(
                tenant=tenant,
                neo4j_client=neo4j_client,
                force=args.force,
                dry_run=args.dry_run,
            )
            reports.append(report)
            if args.dry_run and report["stale"]:
                exit_code = 2   # signal that rebuild is needed
        except Exception as exc:
            log.error("community_rebuild.tenant_failed", tenant=tenant, error=str(exc))
            reports.append({"tenant": tenant, "error": str(exc)})
            exit_code = 1

    # Summary
    rebuilt  = sum(1 for r in reports if r.get("rebuilt"))
    skipped  = sum(1 for r in reports if not r.get("rebuilt") and not r.get("error"))
    failed   = sum(1 for r in reports if r.get("error"))
    total_communities = sum(r.get("community_count", 0) for r in reports)

    log.info(
        "community_rebuild.summary",
        tenants_processed=len(tenants),
        rebuilt=rebuilt,
        skipped=skipped,
        failed=failed,
        total_communities=total_communities,
    )

    if args.dry_run:
        stale_tenants = [r["tenant"] for r in reports if r.get("stale")]
        if stale_tenants:
            print(f"[dry-run] Stale tenants needing rebuild: {stale_tenants}")
        else:
            print("[dry-run] All tenants up to date.")

    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild GraphRAG community summaries for one or all tenants."
    )
    parser.add_argument(
        "--tenant",
        default="",
        help="Tenant to rebuild (default: all tenants)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the graph is not stale",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check staleness but do not rebuild",
    )
    parsed = parser.parse_args()
    sys.exit(asyncio.run(main(parsed)))
