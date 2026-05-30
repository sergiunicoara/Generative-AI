"""Graph maintenance â€” long-term decay cleanup and health monitoring.

Problems solved
---------------
1. Long-term graph decay â€” stale nodes, orphaned relationships, and
   low-confidence edges accumulate silently over months. Without
   scheduled cleanup, graph quality degrades and retrieval worsens.

2. Emergent graph complexity â€” the graph structure grows in unexpected
   ways as more documents are ingested. This script monitors structural
   health metrics so degradation is visible before it becomes a problem.

Run schedule (recommended):
    Daily:   python scripts/maintenance.py --mode stale
    Weekly:  python scripts/maintenance.py --mode full
    Monthly: python scripts/maintenance.py --mode report

Or add to cron:
    0 2 * * * cd /app && python scripts/maintenance.py --mode stale
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import structlog

from graphrag.graph.neo4j_client import get_neo4j
from graphrag.graph.cycle_detector import CycleDetector
from graphrag.graph.ingestion_validator import IngestionValidator
from graphrag.graph.propagation import PropagationService

log = structlog.get_logger(__name__)

# Thresholds for maintenance actions
STALE_EDGE_DAYS        = 365   # edges older than this with conf < 0.3 are pruned
LOW_CONF_PRUNE_THRESH  = 0.2   # edges below this confidence are candidates for removal
ORPHAN_AGE_DAYS        = 30    # orphan nodes older than this are safe to flag


async def run_stale_cleanup(neo4j) -> dict:
    """
    Remove edges that are:
    - Older than STALE_EDGE_DAYS days
    - AND have confidence below LOW_CONF_PRUNE_THRESH
    - AND are superseded (source doc has been superseded)
    """
    result = await neo4j.run(
        """
        MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
        WHERE r.confidence < $conf_threshold
          AND r.extracted_at IS NOT NULL
          AND duration.between(datetime(r.extracted_at), datetime()).days > $age_days
        OPTIONAL MATCH (d:Document {id: r.source_doc_id})
        WHERE d.superseded_by IS NOT NULL
        WITH r, count(d) AS superseded_count
        WHERE superseded_count > 0
        DELETE r
        RETURN count(r) AS removed
        """,
        conf_threshold=LOW_CONF_PRUNE_THRESH,
        age_days=STALE_EDGE_DAYS,
    )
    removed = result[0]["removed"] if result else 0
    log.info("maintenance.stale_edges_removed", count=removed)
    return {"stale_edges_removed": removed}


async def run_orphan_cleanup(neo4j) -> dict:
    """
    Flag orphan entities (no chunk link, older than ORPHAN_AGE_DAYS).
    Does NOT delete â€” flags for human review.
    """
    result = await neo4j.run(
        """
        MATCH (e:Entity)
        WHERE NOT (e)<-[:MENTIONS]-(:Chunk)
        SET e.orphan_flagged = true,
            e.orphan_flagged_at = datetime()
        RETURN count(e) AS flagged
        """
    )
    flagged = result[0]["flagged"] if result else 0
    log.info("maintenance.orphans_flagged", count=flagged)
    return {"orphans_flagged": flagged}


async def run_dirty_recompute(neo4j) -> dict:
    """Recompute all dirty materialized aggregates."""
    svc = PropagationService(neo4j)
    count = await svc.batch_recompute_dirty(limit=500)
    return {"aggregates_recomputed": count}


async def run_cycle_check(neo4j) -> dict:
    """Detect and flag cyclic dependencies."""
    detector = CycleDetector(neo4j)
    cycles = await detector.run()
    return {"cycles_detected": len(cycles)}


async def run_health_report(neo4j) -> dict:
    """Generate a full structural health report."""
    rows = await neo4j.run(
        """
        MATCH (e:Entity) WITH count(e) AS entities
        MATCH (c:Chunk)  WITH entities, count(c) AS chunks
        MATCH ()-[r:RELATES_TO]->() WITH entities, chunks, count(r) AS relations
        OPTIONAL MATCH (e:Entity) WHERE e.orphan_flagged = true
        WITH entities, chunks, relations, count(e) AS orphans
        OPTIONAL MATCH (e:Entity) WHERE e.status_dirty = true
        WITH entities, chunks, relations, orphans, count(e) AS dirty_nodes
        OPTIONAL MATCH (d:Document) WHERE d.superseded_by IS NOT NULL
        RETURN entities, chunks, relations, orphans, dirty_nodes,
               count(d) AS superseded_docs
        """
    )
    report = dict(rows[0]) if rows else {}

    # Confidence distribution
    conf_rows = await neo4j.run(
        """
        MATCH ()-[r:RELATES_TO]->()
        RETURN avg(r.confidence) AS avg_confidence,
               min(r.confidence) AS min_confidence,
               count(CASE WHEN r.confidence < 0.5 THEN 1 END) AS low_conf_count
        """
    )
    if conf_rows:
        report.update(
            {
                "avg_edge_confidence": round(conf_rows[0]["avg_confidence"] or 0, 3),
                "min_edge_confidence": round(conf_rows[0]["min_confidence"] or 0, 3),
                "low_confidence_edges": conf_rows[0]["low_conf_count"],
            }
        )

    log.info("maintenance.health_report", **report)
    return report


async def main():
    parser = argparse.ArgumentParser(description="Knowledge graph maintenance")
    parser.add_argument(
        "--mode",
        choices=["stale", "orphans", "dirty", "cycles", "report", "full"],
        default="report",
        help="Maintenance mode to run",
    )
    args = parser.parse_args()

    neo4j = get_neo4j()
    results = {}

    if args.mode in ("stale", "full"):
        results.update(await run_stale_cleanup(neo4j))

    if args.mode in ("orphans", "full"):
        results.update(await run_orphan_cleanup(neo4j))

    if args.mode in ("dirty", "full"):
        results.update(await run_dirty_recompute(neo4j))

    if args.mode in ("cycles", "full"):
        results.update(await run_cycle_check(neo4j))

    if args.mode in ("report", "full"):
        results.update(await run_health_report(neo4j))

    print("\nâ”€â”€ Maintenance Report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    for key, value in results.items():
        print(f"  {key:<35} {value}")
    print()

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())

