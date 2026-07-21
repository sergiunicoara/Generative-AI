#!/usr/bin/env python
"""
Link an external RDF/OWL ontology's entities to this tenant's knowledge graph
and (optionally) merge both into a single Turtle file queryable via SPARQL.

Reuses the existing intra-corpus alias-resolution pipeline (exact ->
normalized -> fuzzy -> embedding -> human review) against a customer's own
Turtle export instead of a second document, and emits owl:sameAs bridges for
confident matches. See graphrag/graph/cross_ontology_linker.py for the
full strategy writeup.

Usage:
    python scripts/link_external_ontology.py --external customer.ttl --tenant telecom
    python scripts/link_external_ontology.py --external customer.ttl --tenant telecom --embed
    python scripts/link_external_ontology.py --external customer.ttl --tenant telecom \
        --merge-with exports/graph_export.ttl --output exports/merged.ttl
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
from pathlib import Path

# Match ingest_corpus.py's Windows console-encoding guard.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import structlog
log = structlog.get_logger("link_external_ontology")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--external", required=True, help="Path to the external Turtle (.ttl) file")
    parser.add_argument("--tenant", default="default", help="Tenant to resolve entities against")
    parser.add_argument("--embed", action="store_true",
                         help="Also attempt embedding-similarity matching for labels with no exact/fuzzy candidate")
    parser.add_argument("--merge-with", help="Path to this platform's own Turtle export to merge with")
    parser.add_argument("--output", help="Where to write the merged Turtle graph (used with --merge-with; default exports/merged.ttl)")
    args = parser.parse_args()

    from graphrag.graph.cross_ontology_linker import CrossOntologyLinker
    from graphrag.graph.neo4j_client import get_neo4j

    embedder = None
    if args.embed:
        from graphrag.ingestion.embedder import Embedder
        embedder = Embedder()

    neo4j = get_neo4j()
    linker = CrossOntologyLinker(neo4j, tenant=args.tenant, embedder=embedder)

    print(f"\n{'='*60}")
    print(f"  Cross-ontology linking  —  tenant: {args.tenant}")
    print(f"  External: {args.external}")
    print(f"{'='*60}\n")

    result = await linker.link(args.external)
    print(result.summary())

    if result.auto_linked:
        print("\nAuto-linked (owl:sameAs):")
        for m in result.auto_linked:
            score_note = f", score={m['score']:.2f}" if "score" in m else ""
            print(f"  {m['label']!r} -> {m['internal_name']} ({m['internal_type']}) [{m['match_type']}{score_note}]")

    if result.queued_for_review:
        print("\nQueued for human review:")
        for m in result.queued_for_review:
            print(f"  {m['label']!r} ~ {m['candidate_name']} (score={m['score']:.1f}, item_id={m['item_id']})")

    if result.unmatched:
        print(f"\nUnmatched: {len(result.unmatched)}")
        for m in result.unmatched[:10]:
            print(f"  {m['label']!r} ({m['reason']})")
        if len(result.unmatched) > 10:
            print(f"  ... and {len(result.unmatched) - 10} more")

    if args.merge_with:
        merged = CrossOntologyLinker.merge_graphs(
            args.external, args.merge_with, extra=result.same_as_graph,
        )
        output = args.output or "exports/merged.ttl"
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        merged.serialize(output, format="turtle")
        print(f"\nMerged graph written to {output} ({len(merged)} triples)")
        print("Query it with: SPARQLBridge.from_turtle(output).query(...)")

    await neo4j.close()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
