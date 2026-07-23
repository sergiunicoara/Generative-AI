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
    python scripts/ingest_corpus.py                     # dry-run check only
    python scripts/ingest_corpus.py --commit            # full ingestion
    python scripts/ingest_corpus.py --commit --wipe     # wipe tenant first
    python scripts/ingest_corpus.py --commit --doc FAA-AD-2024-01-02.txt
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
import uuid
from pathlib import Path

# On Windows, stdout defaults to the ANSI codepage (cp1252) when redirected to
# a file, which raises UnicodeEncodeError on Romanian diacritics (ț/ş) logged
# during extraction/contradiction-detection — silently failing every document
# inside ingest's per-doc except-handler. Force UTF-8 regardless of platform.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import structlog
log = structlog.get_logger("ingest_corpus")

# Per-tenant corpus configuration. Predecessors must sort before successors
# (alphabetically, by full path) so they are always ingested first and their
# doc IDs are known by the time the superseding document is written.
# graph_writer registers supersession via DocumentAuthorityService
# (SUPERSEDES edges + superseded_by property), which retrieval's authority
# weighting reads.
# AuthorityLevel enum: REGULATORY/CLIENT_REQUIREMENT=1, MANUFACTURER_SPEC/QUALITY_MANUAL=2,
#                      INTERNAL_PROCEDURE=3, INFORMAL/FORM_RECORD=4
_CORPUS_CONFIGS = {
    "aerospace": {
        "corpus_dir": ROOT / "data" / "sample_docs",
        "recursive": False,
        "authority_map": {
            "FAA-AD":            1,   # regulatory
            "EASA-AD":           1,   # regulatory
            "14CFR":             1,   # regulatory
            "Boeing_MCAS":       2,   # manufacturer spec
            "737MAX_CMM":        2,   # manufacturer spec
            "Airbus_A320neo":    2,   # manufacturer spec
            "G-ABCD":            3,   # internal procedure
            "SWA_fleet":         3,   # internal procedure
            "Boeing_company":    4,   # informal
        },
        "supersession_map": {
            "FAA-AD-2022-03-07.txt": ["FAA-AD-2020-05-11.txt"],
            "FAA-AD-2024-01-02.txt": ["FAA-AD-2022-03-07.txt"],
        },
    },
}


def _automotive_corpus_config() -> dict:
    """
    Derive the automotive corpus config from its domain ontology.

    config/ontologies/automotive_iatf.yml's `document_prefixes` and
    `supersession_chains` sections are the source of truth, so the ingestion
    script and the ontology never drift apart.
    """
    from graphrag.graph.domain_ontology import (
        get_ontology_path_for_tenant,
        load_domain_ontology,
    )

    ontology_path = get_ontology_path_for_tenant("automotive")
    ontology = load_domain_ontology(ontology_path) if ontology_path else {}

    prefixes = ontology.get("document_prefixes", {}) or {}
    # Longest prefix first so e.g. "PCAL" matches before "PC".
    authority_map = {
        prefix: spec["authority"]
        for prefix, spec in sorted(prefixes.items(), key=lambda kv: -len(kv[0]))
    }

    supersession_map: dict[str, list[str]] = {}
    for chain in ontology.get("supersession_chains", []) or []:
        supersession_map.setdefault(chain["successor"], []).append(chain["predecessor"])

    return {
        "corpus_dir": ROOT / "data" / "automotive",
        "recursive": True,
        "authority_map": authority_map,
        "supersession_map": supersession_map,
    }


def _marketing_corpus_config() -> dict:
    """
    Derive the marketing (WPP demo) corpus config from its domain ontology.

    config/ontologies/marketing_adtech.yml's `document_prefixes` and
    `supersession_chains` sections are the source of truth, so the ingestion
    script and the ontology never drift apart.
    """
    from graphrag.graph.domain_ontology import (
        get_ontology_path_for_tenant,
        load_domain_ontology,
    )

    ontology_path = get_ontology_path_for_tenant("marketing")
    ontology = load_domain_ontology(ontology_path) if ontology_path else {}

    prefixes = ontology.get("document_prefixes", {}) or {}
    # Longest prefix first so e.g. "BrandGuideline" matches before any shorter alias.
    authority_map = {
        prefix: spec["authority"]
        for prefix, spec in sorted(prefixes.items(), key=lambda kv: -len(kv[0]))
    }

    supersession_map: dict[str, list[str]] = {}
    for chain in ontology.get("supersession_chains", []) or []:
        supersession_map.setdefault(chain["successor"], []).append(chain["predecessor"])

    return {
        "corpus_dir": ROOT / "data" / "wpp_demo",
        "recursive": False,
        "authority_map": authority_map,
        "supersession_map": supersession_map,
    }


def _telecom_corpus_config() -> dict:
    """
    Derive the telecom (OSS/BSS) corpus config from its domain ontology.

    config/ontologies/telecom_oss.yml's `document_prefixes` and
    `supersession_chains` sections are the source of truth, so the ingestion
    script and the ontology never drift apart.
    """
    from graphrag.graph.domain_ontology import (
        get_ontology_path_for_tenant,
        load_domain_ontology,
    )

    ontology_path = get_ontology_path_for_tenant("telecom")
    ontology = load_domain_ontology(ontology_path) if ontology_path else {}

    prefixes = ontology.get("document_prefixes", {}) or {}
    # Longest prefix first so e.g. "ProvisioningActivationProcedure" matches
    # before any shorter alias.
    authority_map = {
        prefix: spec["authority"]
        for prefix, spec in sorted(prefixes.items(), key=lambda kv: -len(kv[0]))
    }

    supersession_map: dict[str, list[str]] = {}
    for chain in ontology.get("supersession_chains", []) or []:
        supersession_map.setdefault(chain["successor"], []).append(chain["predecessor"])

    return {
        "corpus_dir": ROOT / "data" / "telecom",
        "recursive": False,
        "authority_map": authority_map,
        "supersession_map": supersession_map,
    }


def _corpus_config(tenant: str) -> dict:
    if tenant == "automotive":
        return _automotive_corpus_config()
    if tenant == "marketing":
        return _marketing_corpus_config()
    if tenant == "telecom":
        return _telecom_corpus_config()
    return _CORPUS_CONFIGS.get(tenant, _CORPUS_CONFIGS["aerospace"])


def _authority_level(filename: str, authority_map: dict[str, int]) -> int:
    for prefix, level in authority_map.items():
        if filename.startswith(prefix):
            return level
    return 4


async def reconcile_supersession(neo4j, tenant: str, supersession_map: dict[str, list[str]]) -> int:
    """
    Ensure every supersession_map entry has a SUPERSEDES edge + superseded_by
    property in Neo4j, resolving successor/predecessor filenames to their
    already-written Document node IDs.

    Why this exists: write_document() (graphrag/ingestion/graph_writer.py)
    calls DocumentAuthorityService.register_supersession() per document as
    it's written, but that Cypher MATCHes *both* documents by id —
    ``MATCH (old:Document {id: $old_id})`` included. ingest_all()'s writer
    drains documents in extraction-*completion* order (concurrent extraction,
    single serial writer), not the sorted predecessor-before-successor order
    ``doc.supersedes`` was built from — so if a successor's write races ahead
    of its predecessor's, register_supersession's MATCH silently matches zero
    rows and the edge is dropped with no error, no warning, nothing. This
    reconciliation re-asserts every configured pair once the whole batch has
    finished writing, when every Document node is guaranteed to exist.
    ``register_supersession`` is MERGE-based, so re-running it for pairs that
    already succeeded is a safe no-op.

    Also callable standalone (see --reconcile-supersession) to repair a tenant
    that was already ingested before this reconciliation step existed, with no
    re-ingestion / LLM calls needed — it's a pure graph patch over existing
    Document nodes.

    Returns the number of successor->predecessor pairs applied.
    """
    if not supersession_map:
        return 0
    from graphrag.graph.document_authority import DocumentAuthorityService

    rows = await neo4j.run(
        "MATCH (d:Document {tenant: $tenant}) RETURN d.filename AS filename, d.id AS id",
        tenant=tenant,
    )
    doc_id_by_filename = {r["filename"]: r["id"] for r in rows}

    svc = DocumentAuthorityService(neo4j)
    applied = 0
    for successor_name, predecessor_names in supersession_map.items():
        new_id = doc_id_by_filename.get(successor_name)
        if not new_id:
            continue
        old_ids = [
            doc_id_by_filename[f] for f in predecessor_names if f in doc_id_by_filename
        ]
        if not old_ids:
            continue
        await svc.register_supersession(new_id, old_ids)
        applied += len(old_ids)
    return applied


async def ingest_all(
    doc_filter: str | None,
    commit: bool,
    wipe: bool,
    tenant: str = "aerospace",
) -> int:
    from graphrag.core.models import Document, IngestMessage
    from graphrag.graph.neo4j_client import get_neo4j
    from graphrag.graph.inference_engine import ForwardChainingEngine
    from graphrag.graph.graph_snapshots import GraphSnapshotService
    from graphrag.ingestion.document_loader import load_document

    # Groq/OpenAI SDK calls run synchronously inside loop.run_in_executor(),
    # sharing the loop's default ThreadPoolExecutor (sized min(32, cpu_count+4)
    # — only ~12 workers on an 8-core box). Concurrent doc-level extraction
    # (below) multiplies concurrent blocking calls past that default size,
    # causing severe queueing that looks like a hang. These are network-I/O
    # waits, not CPU work, so a larger pool is safe.
    import concurrent.futures
    asyncio.get_running_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=64)
    )

    config = _corpus_config(tenant)
    corpus_dir       = config["corpus_dir"]
    authority_map    = config["authority_map"]
    supersession_map = config["supersession_map"]

    paths = sorted(
        corpus_dir.rglob("*.txt") if config["recursive"] else corpus_dir.glob("*.txt")
    )
    if doc_filter:
        filters = [f.strip().lower() for f in doc_filter.split(",") if f.strip()]
        paths = [p for p in paths if any(f in p.name.lower() for f in filters)]
    if not paths:
        print(f"No documents found in {corpus_dir}")
        return 1

    print(f"\n{'='*60}")
    print(f"  Corpus ingestion  —  {len(paths)} document(s)")
    print(f"  Tenant: {tenant}   commit={commit}   wipe={wipe}")
    print(f"{'='*60}\n")

    neo4j = get_neo4j()

    if commit:
        # Ensure schema (constraints + vector/fulltext indexes) before any
        # writes. All statements are IF NOT EXISTS, so this is idempotent —
        # and it guards against a fresh/recreated Neo4j volume where e.g.
        # chunk_embeddings / chunk_fulltext are missing, which silently breaks
        # vector + BM25 retrieval after an otherwise-successful ingestion.
        schema_path = Path(__file__).parents[1] / "graphrag" / "graph" / "schema.cypher"
        applied = 0
        for stmt in schema_path.read_text(encoding="utf-8").split(";"):
            stmt = "\n".join(
                l for l in stmt.splitlines() if not l.strip().startswith("--")
            ).strip()
            if not stmt:
                continue
            try:
                await neo4j.run(stmt)
                applied += 1
            except Exception as exc:  # index may exist with other options — non-fatal
                log.warning("ingest_corpus.schema_stmt_failed", error=str(exc)[:120])
        print(f"  [schema] {applied} constraint/index statements ensured\n")

    if wipe and commit:
        log.info("ingest_corpus.wipe_tenant", tenant=tenant)
        await neo4j.run(
            "MATCH (n) WHERE n.tenant = $tenant DETACH DELETE n",
            tenant=tenant,
        )
        print(f"  [wipe] Cleared all nodes for tenant '{tenant}'\n")

    if not commit:
        print("  DRY RUN — documents that would be ingested:\n")
        for p in paths:
            level = _authority_level(p.name, authority_map)
            print(f"    {p.relative_to(corpus_dir)!s:50s}  authority={level}")
        print(f"\n  Pass --commit to write to Neo4j.\n")
        await neo4j.close()
        return 0

    # ── Checkpoint — skip documents already fully ingested ─────────────────────
    # (only meaningful when not --wipe, since --wipe just cleared everything).
    # A document is "complete" once write() finished without raising — marked
    # by ingest_complete=true on its Document node (set further below).
    if not wipe:
        done_rows = await neo4j.run(
            "MATCH (d:Document {tenant: $tenant, ingest_complete: true}) "
            "RETURN d.filename AS filename",
            tenant=tenant,
        )
        already_done = {r["filename"] for r in done_rows}
        if already_done:
            skipped = [p for p in paths if p.name in already_done]
            paths = [p for p in paths if p.name not in already_done]
            print(f"  [checkpoint] Skipping {len(skipped)} already-ingested document(s)\n")
        if not paths:
            print("  Nothing left to ingest — all documents already complete.\n")
            await neo4j.close()
            return 0

    # ── Streaming ingestion ──────────────────────────────────────────────────────
    from graphrag.agents.ingestion_agent import IngestionAgent
    from graphrag.core.config import get_settings as _get_settings_for_rebuild_toggle

    # Disable per-document community rebuild during bulk ingestion — it's an
    # O(graph size) Leiden run, and re-running it after every single document
    # (the default, meant for one-off/low-volume ingestion) turns an O(n)
    # ingestion job into O(n^2) wall-clock time. Rebuild once at the end
    # instead (matches settings.yml's own comment: "disable only if you run
    # scripts/community_rebuild.py externally" — which is what we do below).
    _graph_cfg = _get_settings_for_rebuild_toggle().graph
    _prior_auto_rebuild = _graph_cfg.get("auto_rebuild_communities", True)
    _graph_cfg["auto_rebuild_communities"] = False

    # Same rationale, two more per-document checks that scan the whole graph
    # regardless of how many documents actually changed: cycle detection
    # (no doc-scoping at all) and contradiction scanning (doc_id narrows
    # intent but not the underlying all-pairs query cost). Disable during
    # the bulk loop, run each once at the end.
    _ingestion_cfg = _get_settings_for_rebuild_toggle().ingestion
    _prior_detect_cycles = _ingestion_cfg.get("detect_cycles_after_ingestion", True)
    _prior_scan_contradictions = _ingestion_cfg.get("scan_contradictions_after_ingestion", True)
    _ingestion_cfg["detect_cycles_after_ingestion"] = False
    _ingestion_cfg["scan_contradictions_after_ingestion"] = False

    total_chunks    = 0
    total_entities  = 0
    total_relations = 0
    total_conflicts = 0
    results: list[dict] = []
    doc_ids_by_filename: dict[str, str] = {}

    # Extraction (chunking, embedding, LLM entity/relation extraction) touches
    # only each document's own data, never the shared Entity/alias graph, so
    # it's safe to run concurrently across documents (this is also the slow,
    # LLM-bound part). Writes race on that shared graph (EntityNotFound errors
    # observed when parallelized — see lesson A129), so a single writer
    # consumes a queue and writes one document at a time, as soon as its
    # extraction is ready — instead of holding every document's extraction
    # result in memory until the whole corpus finishes extracting (the prior
    # design's memory footprint scaled with corpus size; this caps it at
    # roughly extract_concurrency documents in flight).
    docs_by_path: dict = {}
    for path in paths:
        doc = load_document(path)
        doc.tenant          = tenant
        doc.authority_level = _authority_level(path.name, authority_map)
        doc.supersedes = [
            doc_ids_by_filename[f]
            for f in supersession_map.get(path.name, [])
            if f in doc_ids_by_filename
        ]
        doc_ids_by_filename[path.name] = doc.id
        docs_by_path[path] = doc

    from graphrag.core.config import get_settings
    extract_concurrency = get_settings().ingestion.get("doc_extract_concurrency", 4)
    sem = asyncio.Semaphore(extract_concurrency)
    agent = IngestionAgent()
    write_queue: asyncio.Queue = asyncio.Queue()

    async def _extract_one(i: int, path) -> None:
        async with sem:
            print(f"[extract {i}/{len(paths)}] {path.name}")
            msg = IngestMessage(job_id=str(uuid.uuid4()), document=docs_by_path[path])
            try:
                payload = await agent.extract(msg)
            except Exception as exc:
                payload = exc
            await write_queue.put((i, path, payload))

    async def _writer() -> None:
        for _ in range(len(paths)):
            i, path, payload = await write_queue.get()
            print(f"[write {i}/{len(paths)}] {path.name}")
            if isinstance(payload, Exception):
                log.error("ingest_corpus.doc_extract_failed", doc=path.name, error=str(payload))
                print(f"       ERROR (extract): {payload}")
                continue
            try:
                result = await agent.write(payload)
                await neo4j.run(
                    "MATCH (d:Document {id: $doc_id, tenant: $tenant}) "
                    "SET d.ingest_complete = true",
                    doc_id=payload["doc"].id, tenant=tenant,
                )
                results.append(result)
                print(
                    f"       chunks={result['chunks']}  "
                    f"entities={result['entities']}  "
                    f"relations={result['relations']}  "
                    f"conflicts={result['maintenance']['new_conflicts']}"
                )
            except Exception as exc:
                log.error("ingest_corpus.doc_write_failed", doc=path.name, error=str(exc))
                print(f"       ERROR (write): {exc}")

    await asyncio.gather(
        _writer(),
        *(_extract_one(i, p) for i, p in enumerate(paths, 1)),
    )

    _graph_cfg["auto_rebuild_communities"] = _prior_auto_rebuild
    _ingestion_cfg["detect_cycles_after_ingestion"] = _prior_detect_cycles
    _ingestion_cfg["scan_contradictions_after_ingestion"] = _prior_scan_contradictions

    for r in results:
        total_chunks    += r["chunks"]
        total_entities  += r["entities"]
        total_relations += r["relations"]
        total_conflicts += r["maintenance"]["new_conflicts"]

    # ── Cycle detection (once, for the whole batch) ─────────────────────────────
    print(f"\n[*] Checking for cycles on '{tenant}' tenant...")
    try:
        from graphrag.graph.cycle_detector import CycleDetector
        _cycles = await CycleDetector(neo4j).run()
        print(f"       Cycles found: {len(_cycles)}")
    except Exception as exc:
        log.warning("ingest_corpus.cycle_check_failed", error=str(exc))
        print(f"       WARNING: cycle check failed — {exc}")

    # ── Supersession reconciliation (once, for the whole batch) ─────────────────
    # See reconcile_supersession() docstring: the per-document
    # register_supersession() call in write_document() can silently no-op if
    # the predecessor doc's write hasn't landed yet (concurrent extraction,
    # completion-order write queue). Re-assert every configured pair now that
    # every Document node in this batch is guaranteed to exist.
    print(f"\n[*] Reconciling supersession chains on '{tenant}' tenant...")
    try:
        _applied = await reconcile_supersession(neo4j, tenant, supersession_map)
        print(f"       Supersession pairs applied: {_applied}")
    except Exception as exc:
        log.warning("ingest_corpus.supersession_reconcile_failed", error=str(exc))
        print(f"       WARNING: supersession reconciliation failed — {exc}")

    # ── Contradiction scan (once, for the whole batch) ──────────────────────────
    print(f"\n[*] Scanning contradictions on '{tenant}' tenant...")
    try:
        from graphrag.graph.contradiction_detector import ContradictionDetector
        _conflicts = await ContradictionDetector(neo4j).scan(doc_id=None, tenant=tenant)
        total_conflicts = len(_conflicts)
        print(f"       New conflicts: {total_conflicts}")
    except Exception as exc:
        log.warning("ingest_corpus.contradiction_scan_failed", error=str(exc))
        print(f"       WARNING: contradiction scan failed — {exc}")

    # ── Community rebuild (once, for the whole batch) ───────────────────────────
    print(f"\n[*] Rebuilding communities on '{tenant}' tenant...")
    try:
        from graphrag.graph.community_builder import CommunityBuilder
        from graphrag.graph.community_manager import CommunityManager
        from graphrag.graph.community_summarizer import CommunitySummarizer
        _manager = CommunityManager(neo4j)
        _builder = CommunityBuilder(tenant=tenant)
        _communities = await _builder.build()
        if _communities:
            _summarizer = CommunitySummarizer()
            _communities = await _summarizer.summarize_all(_communities)
            for _community in _communities:
                await neo4j.merge_community(_community)
        await _manager.mark_rebuilt(tenant=tenant)
        print(f"       Communities rebuilt: {len(_communities)}")
    except Exception as exc:
        log.warning("ingest_corpus.community_rebuild_failed", error=str(exc))
        print(f"       WARNING: community rebuild failed — {exc}")

    # ── Forward-chaining inference ────────────────────────────────────────────
    print(f"\n[*] Running forward-chaining inference on '{tenant}' tenant...")
    try:
        from graphrag.graph.domain_ontology import (
            get_ontology_path_for_tenant,
            load_domain_ontology,
        )
        from graphrag.graph.inference_engine import InferenceRule
        _ontology_path = get_ontology_path_for_tenant(tenant)
        _ontology = load_domain_ontology(_ontology_path) if _ontology_path else {}
        engine = ForwardChainingEngine(neo4j)
        for rule_cfg in _ontology.get("inference_rules", []):
            engine.add_rule(InferenceRule(
                name=rule_cfg["name"],
                rule_type=rule_cfg["rule_type"],
                relation=rule_cfg["relation"],
                derived_relation=rule_cfg.get("derived_relation", ""),
                body_relation_2=rule_cfg.get("body_relation_2", ""),
                max_depth=rule_cfg.get("max_depth", 3),
                confidence_decay=rule_cfg.get("confidence_decay", 0.9),
            ))
        fc_report = await engine.run(tenant=tenant)
        inferred_edges = fc_report.get("total_inferred", 0)
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
            tenant=tenant,
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
            tenant=tenant,
            include_health=True,
        )
        print(f"       Snapshot: {snap_id}")
    except Exception as exc:
        log.warning("ingest_corpus.snapshot_failed", error=str(exc))
        print(f"       WARNING: snapshot failed — {exc}")

    await neo4j.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  INGESTION COMPLETE  —  tenant: {tenant}")
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


async def _run_reconcile_supersession_only(tenant: str) -> int:
    """Standalone repair path: patch SUPERSEDES edges on an already-ingested
    tenant, no re-ingestion / LLM calls. See reconcile_supersession()."""
    from graphrag.graph.neo4j_client import get_neo4j

    config = _corpus_config(tenant)
    neo4j = get_neo4j()
    print(f"\n[*] Reconciling supersession chains on '{tenant}' tenant (standalone)...")
    applied = await reconcile_supersession(neo4j, tenant, config["supersession_map"])
    print(f"       Supersession pairs applied: {applied}")
    await neo4j.close()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--commit", action="store_true",
                        help="Write to Neo4j (default: dry-run)")
    parser.add_argument("--wipe",   action="store_true",
                        help="Delete all nodes for the selected tenant before ingesting")
    parser.add_argument("--doc",    default=None,
                        help="Filter to document filename substring(s), comma-separated")
    parser.add_argument("--tenant", default="aerospace",
                        choices=sorted(set(_CORPUS_CONFIGS) | {"automotive", "marketing"}),
                        help="Tenant corpus to ingest (default: aerospace)")
    parser.add_argument("--reconcile-supersession", action="store_true",
                        help="Patch SUPERSEDES edges on an already-ingested tenant from "
                             "its supersession_map, then exit. No re-ingestion / LLM calls "
                             "— repairs the race described in reconcile_supersession().")
    args = parser.parse_args()

    if args.reconcile_supersession:
        raise SystemExit(asyncio.run(_run_reconcile_supersession_only(args.tenant)))

    raise SystemExit(asyncio.run(ingest_all(
        doc_filter=args.doc,
        commit=args.commit,
        wipe=args.wipe,
        tenant=args.tenant,
    )))


if __name__ == "__main__":
    main()
