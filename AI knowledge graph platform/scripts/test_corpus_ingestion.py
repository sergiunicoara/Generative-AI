"""
Test the ingestion pipeline against the 12-document aerospace seed corpus.

No Neo4j required — runs chunk → LLM extract in-memory and reports:
  - entities extracted per document
  - relations extracted
  - contradiction candidates (same entity, opposing relations)
  - supersession chains found

Usage:
    python scripts/test_corpus_ingestion.py
    python scripts/test_corpus_ingestion.py --doc FAA-AD-2024-01-02.txt  # single file
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORPUS_DIR = ROOT / "data" / "sample_docs"

ENTITY_TYPES = [
    "REGULATOR", "ORG", "AIRCRAFT_TYPE", "AIRCRAFT_COMPONENT",
    "AIRWORTHINESS_DIRECTIVE", "REGULATION", "MAINTENANCE_PROCEDURE",
    "PRODUCT", "PERSON", "LOCATION",
]

# Relation pairs that indicate potential contradictions
CONTRADICTION_PAIRS = {
    ("IS_AIRWORTHY",          "IS_UNAIRWORTHY"),
    ("IS_COMPLIANT_WITH",     "IS_NON_COMPLIANT_WITH"),
    ("APPROVED",              "REJECTED"),
    ("ACTIVE",                "SUPERSEDED"),
}


async def extract_doc(path: Path, extractor) -> tuple[list, list, dict]:
    """Returns (entities, relations, id_to_name_map)."""
    from graphrag.core.models import Chunk, Document
    from graphrag.ingestion.chunker import chunk_document

    text = path.read_text(encoding="utf-8")
    doc = Document(filename=path.name, source_path=str(path),
                   raw_text=text, tenant="aerospace")
    chunks = chunk_document(doc)
    print(f"  -> {path.name}: {len(chunks)} chunks", flush=True)

    all_entities, all_relations = [], []
    id_to_name: dict[str, str] = {}
    for chunk in chunks:
        try:
            entities, relations = await extractor.extract(chunk)
            all_entities.extend(entities)
            all_relations.extend(relations)
            for e in entities:
                id_to_name[e.id] = e.name
        except Exception as exc:
            print(f"    WARN chunk {chunk.chunk_index} failed: {exc}", flush=True)
    return all_entities, all_relations, id_to_name


def find_contradictions(all_relations: list, id_to_name: dict) -> list[dict]:
    """Find same (src, tgt) entity pairs with opposing relation types."""
    by_pair: dict[tuple, list] = defaultdict(list)
    for r in all_relations:
        src_name = id_to_name.get(r.source_entity_id, r.source_entity_id)
        tgt_name = id_to_name.get(r.target_entity_id, r.target_entity_id)
        key = (src_name.lower(), tgt_name.lower())
        by_pair[key].append((r.relation.upper(), src_name, tgt_name))

    hits = []
    for _key, entries in by_pair.items():
        rel_set = {e[0] for e in entries}
        src_name = entries[0][1]
        tgt_name = entries[0][2]
        for a, b in CONTRADICTION_PAIRS:
            if a in rel_set and b in rel_set:
                hits.append({"src": src_name, "tgt": tgt_name, "relations": sorted(rel_set)})
    return hits


def find_supersessions(all_relations: list, id_to_name: dict) -> list[dict]:
    results = []
    for r in all_relations:
        if r.relation.upper() in ("SUPERSEDES", "SUPERSEDED_BY", "REPLACES"):
            src = id_to_name.get(r.source_entity_id, r.source_entity_id)
            tgt = id_to_name.get(r.target_entity_id, r.target_entity_id)
            results.append({"src": src, "tgt": tgt, "relation": r.relation})
    return results


async def main(doc_filter: str | None = None) -> int:
    from graphrag.ingestion.extractor import Extractor

    paths = sorted(CORPUS_DIR.glob("*.txt"))
    if doc_filter:
        paths = [p for p in paths if doc_filter.lower() in p.name.lower()]
    if not paths:
        print(f"No documents found in {CORPUS_DIR}")
        return 1

    print(f"\n{'='*60}")
    print(f"  Corpus ingestion test — {len(paths)} documents")
    print(f"{'='*60}\n")

    extractor = Extractor()
    all_entities, all_relations = [], []
    all_id_to_name: dict[str, str] = {}
    entity_counts: dict[str, int] = {}

    for path in paths:
        print(f"Processing: {path.name}")
        entities, relations, id_to_name = await extract_doc(path, extractor)
        all_entities.extend(entities)
        all_relations.extend(relations)
        all_id_to_name.update(id_to_name)
        entity_counts[path.name] = len(entities)
        print(f"  OK {len(entities)} entities, {len(relations)} relations extracted\n")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Documents processed : {len(paths)}")
    print(f"  Total entities      : {len(all_entities)}")
    print(f"  Total relations     : {len(all_relations)}")

    # Entity types breakdown
    type_counts: dict[str, int] = defaultdict(int)
    for e in all_entities:
        type_counts[e.type] += 1
    print("\n  Entity types:")
    for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:35s} {n}")

    # Top entities by name frequency
    name_counts: dict[str, int] = defaultdict(int)
    for e in all_entities:
        name_counts[e.name] += 1
    print("\n  Most mentioned entities:")
    for name, cnt in sorted(name_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {name:40s} {cnt}x")

    # ── Supersession chains ────────────────────────────────────────────────────
    supersessions = find_supersessions(all_relations, all_id_to_name)
    print(f"\n  Supersession relations ({len(supersessions)} found):")
    for s in supersessions:
        print(f"    {s['src']:35s} SUPERSEDES -> {s['tgt']}")

    # ── Contradiction candidates ───────────────────────────────────────────────
    contradictions = find_contradictions(all_relations, all_id_to_name)
    print(f"\n  Contradiction candidates ({len(contradictions)} found):")
    if contradictions:
        for c in contradictions:
            print(f"    !! {c['src']} <-> {c['tgt']}")
            print(f"       relations: {', '.join(c['relations'])}")
    else:
        print("    (none detected — may need lower confidence threshold)")

    # ── Per-doc breakdown ──────────────────────────────────────────────────────
    print(f"\n  Per-document entity counts:")
    for fname, cnt in sorted(entity_counts.items(), key=lambda x: -x[1]):
        print(f"    {fname:50s} {cnt}")

    print(f"\n{'='*60}")
    print("  Test complete.")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", help="Filter to a specific document filename")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.doc)))
