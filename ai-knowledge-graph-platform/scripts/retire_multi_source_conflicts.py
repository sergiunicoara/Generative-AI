"""Retire `multi_source` Conflict nodes — they were never contradictions.

Background
----------
`ContradictionDetector` used to run a `multi_source` strategy that fired on
"same (src, rel, tgt) asserted by two non-superseding documents". An edge *is* a
single triple, so both documents necessarily assert the **same fact** — that is
corroboration, and `merge_relations_batch` already treats it as such by raising
the edge's confidence via noisy-OR. The detector was flagging the same evidence
as a defect.

Measured before this change: 94 of aerospace's 95 and 61 of automotive's 63 open
conflicts were `multi_source`, so the conflicts dashboard and the
`contradiction_rate` alert were reading a number that meant almost nothing, and
the four strategies that detect real disagreement were invisible underneath it.

The strategy is now `_record_corroboration()` and writes
`independent_source_count` to the edge. This script retires the Conflict nodes
it already created.

Why status and not DELETE
-------------------------
`false_positive` is an already-supported status (`ContradictionDetector.resolve`,
`api/routes/corrections.py`, the dashboard's resolution dropdown), and there is
precedent for exactly this operation in tasks/lessons.md. Setting status keeps
the audit trail, drops the nodes out of every `status: 'open'` metric, and is
trivially reversible — deletion is none of those things.

Usage
-----
    python scripts/retire_multi_source_conflicts.py                 # dry-run, all tenants
    python scripts/retire_multi_source_conflicts.py --apply
    python scripts/retire_multi_source_conflicts.py --tenant aerospace --apply

Exit codes
----------
  0  — nothing to retire, or applied successfully
  1  — failed
  2  — dry-run found conflicts that would be retired
"""

from __future__ import annotations

import argparse
import asyncio

import structlog

log = structlog.get_logger("retire_multi_source_conflicts")

_RESOLVED_BY = "system:multi_source_retired"


async def count_multi_source(neo4j, tenant: str | None) -> list[dict]:
    tenant_filter = "AND c.tenant = $tenant" if tenant else ""
    params = {"tenant": tenant} if tenant else {}
    return await neo4j.run(
        f"""
        MATCH (c:Conflict {{conflict_type: 'multi_source', status: 'open'}})
        WHERE true {tenant_filter}
        RETURN coalesce(c.tenant, '') AS tenant, count(c) AS n
        ORDER BY tenant
        """,
        **params,
    )


async def retire(neo4j, tenant: str | None) -> int:
    """Mark open multi_source conflicts as false positives. Idempotent — a second
    run matches nothing, because the status filter no longer holds."""
    tenant_filter = "AND c.tenant = $tenant" if tenant else ""
    params = {"tenant": tenant} if tenant else {}
    rows = await neo4j.run(
        f"""
        MATCH (c:Conflict {{conflict_type: 'multi_source', status: 'open'}})
        WHERE true {tenant_filter}
        SET c.status      = 'false_positive',
            c.resolved_at = datetime(),
            c.resolved_by = $resolved_by,
            c.resolution_note =
                'multi_source detected corroboration, not contradiction: an edge is '
              + 'one triple, so its source documents assert the same fact. '
              + 'Superseded by independent_source_count on the edge.'
        RETURN count(c) AS n
        """,
        resolved_by=_RESOLVED_BY,
        **params,
    )
    return int(rows[0]["n"]) if rows else 0


async def main(args) -> int:
    from graphrag.graph.neo4j_client import get_neo4j

    neo4j = get_neo4j()
    tenant = args.tenant or None

    try:
        counts = await count_multi_source(neo4j, tenant)
    except Exception as exc:
        log.error("retire_multi_source.count_failed", error=str(exc))
        return 1

    total = sum(int(r["n"]) for r in counts)
    if not total:
        print("No open multi_source conflicts found — nothing to retire.")
        return 0

    for row in counts:
        print(f"  {row['tenant'] or '(no tenant)':<16} {row['n']:>5} open multi_source conflicts")
    print(f"  {'TOTAL':<16} {total:>5}")

    if not args.apply:
        print("\n[dry-run] Re-run with --apply to retire these as 'false_positive'.")
        return 2

    try:
        retired = await retire(neo4j, tenant)
    except Exception as exc:
        log.error("retire_multi_source.apply_failed", error=str(exc))
        return 1

    print(f"\nRetired {retired} conflicts to status='false_positive' "
          f"(resolved_by={_RESOLVED_BY}).")
    return 0


if __name__ == "__main__":
    import io
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Retire multi_source Conflict nodes as false positives."
    )
    parser.add_argument("--tenant", default="", help="Limit to one tenant (default: all)")
    parser.add_argument("--apply", action="store_true",
                        help="Actually write the status change (default: dry-run)")
    sys.exit(asyncio.run(main(parser.parse_args())))
