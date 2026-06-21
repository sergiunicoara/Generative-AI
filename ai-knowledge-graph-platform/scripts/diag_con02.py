#!/usr/bin/env python3
"""Diagnostic: dump full pipeline trace for CON-02 question to see where FP-INJ-03 and REG-EVF-01 land."""
from __future__ import annotations

import asyncio
import io
import os
import sys
from pathlib import Path

# Windows console cp1252 can't encode Romanian diacritics
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parents[1]))

# Override Neo4j creds to localhost (Docker exposes 7687)
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "graphrag_dev")

import structlog
log = structlog.get_logger("diag_con02")


async def main():
    from graphrag.retrieval.local_search import LocalSearch

    question = "Este obligatoriu ca furnizorii noi să fie auditați la sediul lor înainte de aprobare?"

    ls = LocalSearch()
    results = await ls.search(question, session_id="diag-con02", tenant="automotive")

    chunks = results.get("chunks", [])

    print(f"\n{'='*70}\nCON-02 retrieval trace — total chunks: {len(chunks)}\n{'='*70}\n")

    target_filenames = {"FP-INJ-03", "REG-EVF-01", "CSR-CLIENT-2023"}

    for i, c in enumerate(chunks):
        filename = c.get("_doc_name", "?")
        marker = "  >> " if filename in target_filenames else "     "
        score = c.get("final_score", c.get("rerank_score", c.get("score", 0)))
        chunk_id = c.get("chunk_id", "?")
        text = c.get("text", "")[:80].replace("\n", " ")
        print(f"{marker}{i:3d} | {filename:30s} | score={score:.4f} | {chunk_id[:8]}")
        print(f"       text: {text}...")

    print(f"\n{'='*70}\nSummary by document:\n{'='*70}")
    counts: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        f = c.get("_doc_name", "?")
        counts.setdefault(f, []).append(i)

    for doc in ["CSR-CLIENT-2023", "FP-INJ-03", "REG-EVF-01"]:
        positions = counts.get(doc, [])
        print(f"  {doc:25s}: {len(positions)} chunks at positions {positions}")
        if doc == "FP-INJ-03" and positions:
            j = positions[0]
            c = chunks[j]
            print(f"    >>> chunk @ {j}: {c.get('text','')[:500]}")
    print()


if __name__ == "__main__":
    asyncio.run(main())