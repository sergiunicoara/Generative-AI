#!/usr/bin/env python
"""
Ingest the 12-document aerospace seed corpus into Neo4j via the full pipeline.

Runs the real production path:
  document_loader -> chunker -> embedder -> extractor (LLM) ->
  graph_writer (alias resolution, contradiction detection) ->
  forward-chaining inference -> graph snapshot

After completion, prints real entity/edge/conflict counts queried from Neo4j
and records them in a GraphHealthSnapshot so the dashboard reflects real data.

Usage:
    py -3.11 scripts/ingest_corpus.py                     # dry-run check only
    py -3.11 scripts/ingest_corpus.py --commit            # full ingestion
    py -3.11 scripts/ingest_corpus.py --commit --wipe     # wipe tenant first
    py -3.11 scripts/ingest_corpus.py --commit --doc FAA-AD-2024-01-02.txt
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import structlog
log = structlog.get_logger("ingest_corpus")

CORPUS_DIR = ROOT / "data" / "sample_docs"
TENANT = "aerospace"

# Authority levels by filename prefix (lower = higher authority)
# AuthorityLevel enum: REGULATORY=1, MANUFACTURER_SPEC=2, INTERNAL_PROCEDURE=3, INFORMAL=4
_AUTHORITY_MAP = {
    "FAA-AD":            1,   # regulatory
    "EASA-AD":           1,   # regulatory
    "14CFR":             1,   # regulatory
    "Boeing_MCAS":       2,   # manufacturer spec
    "737MAX_CMM":        2,   # manufacturer spec
    "Airbus_A320neo":    2,   # manufacturer spec
    "G-ABCD":            3,   # internal procedure
    "SWA_fleet":         3,   # internal procedure
    "Boeing_company":    4,   # informal
}


def _authority_level(filename: str) -> int:
    for prefix, level in _AUTHORITY_MAP.items():
        if filename.startswith(prefix):
            return level
    return 4


async def ingest_all(
    doc_filter: str | None,
    commit: bool,
    wipe: bool,
) -> int:
    from graphrag.core.models import Document, IngestMessage
    from graphrag.graph.neo4j_client import get_neo4j
    from graphrag.graph.inference_engine import ForwardChainingEngine
    from graphrag.graph.graph_snapshots import GraphSnapshotService
    from graphrag.ingestion.document_loader import load_document

    paths = sorted(CORPUS_DIR.glob("*.txt"))
    if doc_filter:
        paths = [p for p in paths if doc_filter.lower() in p.name.lower()]
    if not paths:
        print(f"No documents found in {CORPUS_DIR}")
        return 1

    print(f"\n{'='*60}")
    print(f"  Corpus ingestion  —  {len(paths)} document(s)")
    print(f"  Tenant: {TENANT}   commit={commit}   wipe={wipe}")
    print(f"{'='*60}\n")

    neo4j = get_neo4j()

    if wipe and commit:
        log.info("ingest_corpus.wipe_tenant", tenant=TENANT)
        await neo4j.run(
            "MATCH (n) WHERE n.tenant = $tenant DETACH DELETE n",
            tenant=TENANT,
        )
        print(f"  [wipe] Cleared all nodes for tenant '{TENANT}'\n")

    if not commit:
        print("  DRY RUN — documents that would be ingested:\n")
        for p in paths:
            level = _authority_level(p.name)
            print(f"    {p.name:50s}  authority={level}")
        print(f"\n  Pass --commit to write to Neo4j.\n")
        await neo4j.close()
        return 0

    # ── Full ingestion loop ────────────────────────────────────────────────────
    from graphrag.agents.ingestion_agent import IngestionAgent
    agent = IngestionAgent()

    total_chunks    = 0
    total_entities  = 0
    total_relations = 0
    total_conflicts = 0
    results: list[dict] = []

    for i, path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {path.name}")

        # Load + configure document
        doc = load_document(path)
        doc.tenant          = TENANT
        doc.authority_level = _authority_level(path.name)

        msg = IngestMessage(
            job_id=str(uuid.uuid4()),
            document=doc,
        )

        try:
            result = await agent.run(msg)
            total_chunks    += result["chunks"]
            total_entities  += result["entities"]
            total_relations += result["relations"]
            total_conflicts += result["maintenance"]["new_conflicts"]
            results.append(result)
            print(
                f"       chunks={result['chunks']}  "
                f"entities={result['entities']}  "
                f"relations={result['relations']}  "
                f"conflicts={result['maintenance']['new_conflicts']}"
            )
        except Exception as exc:
            log.error("ingest_corpus.doc_failed", doc=path.name, error=str(exc))
            print(f"       ERROR: {exc}")

    # ── Forward-chaining inference ────────────────────────────────────────────
    print(f"\n[*] Running forward-chaining inference on '{TENANT}' tenant...")
    try:
        engine = ForwardChainingEngine(neo4j)
        fc_report = await engine.run(tenant=TENANT)
        inferred_edges = fc_report.get("total_derived", 0)
        print(f"       Derived edges written: {inferred_edges}")
    except Exception as exc:
        log.warning("ingest_corpus.inference_failed", error=str(exc))
        inferred_edges = 0
        print(f"       WARNING: inference failed — {exc}")

    # ── Real counts from Neo4j ────────────────────────────────────────────────
    print(f"\n[*] Querying real graph counts from Neo4j...")
    try:
        rows = await neo4j.run(
            """
            MATCH (e:Entity {tenant: $tenant})
            WITH count(e) AS entity_count
            OPTIONAL MATCH (:Entity {tenant: $tenant})-[r:RELATES_TO {tenant: $tenant}]->(:Entity)
            WITH entity_count, count(r) AS edge_count
            OPTIONAL MATCH (c:Conflict {tenant: $tenant}) WHERE c.status = 'open'
            RETURN entity_count, edge_count, count(c) AS conflict_count
            """,
            tenant=TENANT,
        )
        row = rows[0] if rows else {}
        neo4j_entities  = row.get("entity_count", 0)
        neo4j_edges     = row.get("edge_count", 0)
        neo4j_conflicts = row.get("conflict_count", 0)
    except Exception as exc:
        log.warning("ingest_corpus.count_query_failed", error=str(exc))
        neo4j_entities  = total_entities
        neo4j_edges     = total_relations
        neo4j_conflicts = total_conflicts

    # Contradiction rate per 1k edges (same convention as MEMORY.md)
    contradiction_rate = (
        round(neo4j_conflicts / neo4j_edges * 1000, 2)
        if neo4j_edges > 0 else 0.0
    )

    # ── Snapshot ──────────────────────────────────────────────────────────────
    print(f"\n[*] Creating graph snapshot...")
    try:
        snap_svc = GraphSnapshotService(neo4j)
        snap_id = await snap_svc.create_snapshot(
            label="corpus-ingest-v1",
            tenant=TENANT,
            include_health=True,
        )
        print(f"       Snapshot: {snap_id}")
    except Exception as exc:
        log.warning("ingest_corpus.snapshot_failed", error=str(exc))
        print(f"       WARNING: snapshot failed — {exc}")

    await neo4j.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  INGESTION COMPLETE  —  tenant: {TENANT}")
    print(f"{'='*60}")
    print(f"  Documents ingested  : {len(results)} / {len(paths)}")
    print(f"  Chunks processed    : {total_chunks}")
    print(f"  Entities (pipeline) : {total_entities}")
    print(f"  Relations (pipeline): {total_relations}")
    print(f"  ---")
    print(f"  Entities in Neo4j   : {neo4j_entities}  (after alias dedup)")
    print(f"  Edges in Neo4j      : {neo4j_edges}  (asserted + inferred)")
    print(f"  Open conflicts      : {neo4j_conflicts}")
    print(f"  Contradiction rate  : {contradiction_rate:.2f} / 1k edges")
    print(f"  Inferred edges      : {inferred_edges}")
    print(f"{'='*60}\n")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--commit", action="store_true",
                        help="Write to Neo4j (default: dry-run)")
    parser.add_argument("--wipe",   action="store_true",
                        help="Delete all aerospace tenant nodes before ingesting")
    parser.add_argument("--doc",    default=None,
                        help="Filter to a single document filename substring")
    args = parser.parse_args()

    raise SystemExit(asyncio.run(ingest_all(
        doc_filter=args.doc,
        commit=args.commit,
        wipe=args.wipe,
    )))


if __name__ == "__main__":
    main()
