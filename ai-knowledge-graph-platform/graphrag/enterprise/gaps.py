"""Ingestion-gap detection: what the source has that the graph never (fully) received.

``ContentSyncService.reconcile`` handles one direction (items that vanished from the source are tombstoned). This
module covers the other: a missed webhook or a failed ingest leaves an item the source has but the graph lacks,
holds an older revision of, or holds without chunks.

No load on the source: the caller supplies an *inventory* (external id -> source version) that the connector already
produces for ``reconcile`` (a listing, no content fetch). The graph side is one read-only Cypher query. Nothing is
written; ``repair_ids`` is the list a connector should re-fetch.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from graphrag.graph.validation.graph_checks import run_read_only

GRAPH_DOCUMENTS = """
MATCH (d:Document {tenant: $tenant, source_id: $source_id})
WHERE coalesce(d.external_id, '') <> ''
OPTIONAL MATCH (c:Chunk {tenant: $tenant})-[:PART_OF]->(d)
WITH d, count(c) AS chunks
RETURN d.external_id AS external_id, coalesce(d.source_version, '') AS source_version,
       coalesce(d.is_deleted, false) AS is_deleted, coalesce(d.status, '') AS status, chunks
"""

BUCKETS = ("missing", "stale", "restored_in_source", "not_ingested", "duplicate_in_graph",
           "unverifiable_version", "would_tombstone")
_REPAIR = ("missing", "stale", "restored_in_source", "not_ingested")


def normalise_inventory(inventory: Mapping[str, str] | Iterable[str]) -> dict[str, str]:
    """Accept ``{external_id: version}`` or a bare list of ids (versions then unknown, so staleness is not judged)."""
    if isinstance(inventory, Mapping):
        return {str(k): str(v or "") for k, v in inventory.items() if str(k)}
    return {str(k): "" for k in inventory if str(k)}


def diff_inventory(inventory: Mapping[str, str] | Iterable[str], graph_rows: list[dict[str, Any]],
                   *, sample: int = 25) -> dict[str, Any]:
    """Pure comparison of a source inventory with the graph's documents for one source."""
    inv = normalise_inventory(inventory)
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in graph_rows:
        by_id.setdefault(str(row.get("external_id", "")), []).append(row)

    found: dict[str, list[str]] = {name: [] for name in BUCKETS}
    ok = 0
    for ext_id, version in inv.items():
        rows = by_id.get(ext_id)
        if not rows:
            found["missing"].append(ext_id)
            continue
        if len(rows) > 1:
            found["duplicate_in_graph"].append(ext_id)
        live = [r for r in rows if not r.get("is_deleted")]
        if not live:
            found["restored_in_source"].append(ext_id)
            continue
        row = live[0]
        if row.get("status") != "done" or int(row.get("chunks") or 0) == 0:
            found["not_ingested"].append(ext_id)
        elif version and not row.get("source_version"):
            found["unverifiable_version"].append(ext_id)
            ok += 1
        elif version and version != row.get("source_version"):
            found["stale"].append(ext_id)
        else:
            ok += 1
    for ext_id, rows in by_id.items():
        if ext_id not in inv and any(not r.get("is_deleted") for r in rows):
            found["would_tombstone"].append(ext_id)

    total = len(inv)
    repair = sorted({i for name in _REPAIR for i in found[name]})
    return {
        "inventory_items": total,
        "graph_documents": len(graph_rows),
        "in_sync": ok,
        # share of source items present, current (when versions are known) and fully ingested
        "coverage": round(ok / total, 4) if total else None,
        "counts": {name: len(ids) for name, ids in found.items()},
        "samples": {name: sorted(ids)[:sample] for name, ids in found.items() if ids},
        "repair_ids": repair,
        "note": "would_tombstone is what reconcile() would soft-delete; unverifiable_version means the graph holds no "
                "source_version so a changed revision cannot be detected. Read-only; nothing was changed.",
    }


async def find_ingestion_gaps(neo4j, *, tenant: str, source_id: str,
                              inventory: Mapping[str, str] | Iterable[str]) -> dict[str, Any]:
    rows = await run_read_only(neo4j, GRAPH_DOCUMENTS, tenant=tenant, source_id=source_id, as_of=None)
    report = diff_inventory(inventory, [dict(r) for r in rows])
    report.update(tenant=tenant, source_id=source_id)
    return report


__all__ = ["BUCKETS", "GRAPH_DOCUMENTS", "diff_inventory", "find_ingestion_gaps", "normalise_inventory"]
