"""Print the ranked local-search chunks reaching the LLM for aerospace golden questions.

Used to test whether AUT-01 / MH-06 fail because the answering chunk is absent
from the context, or present-but-unsynthesized -- and to measure how many of the
scarce top-k slots are consumed by duplicate copies of the same document
(aerospace has 4 files ingested twice; 52 of its 138 chunks are duplicate text).

    python scripts/dev/diagnose_aerospace_retrieval.py AUT-01 MH-06
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


async def main() -> None:
    ids = sys.argv[1:] or ["AUT-01", "MH-06"]

    from graphrag.graph.neo4j_client import get_neo4j
    from graphrag.retrieval.local_search import LocalSearch

    golden = json.loads(
        (ROOT / "evals" / "golden_set.json").read_text(encoding="utf-8")
    )
    neo4j = get_neo4j()

    # chunk_id -> (filename, document node id) so duplicate copies are visible
    rows = await neo4j.run(
        """
        MATCH (c:Chunk)-[:PART_OF]->(d:Document {tenant: $tenant})
        RETURN c.id AS chunk_id, d.filename AS filename, d.id AS doc_id
        """,
        tenant="aerospace",
    )
    meta = {r["chunk_id"]: (r["filename"], r["doc_id"]) for r in rows}

    # which filenames exist as more than one Document node
    doc_copies: dict[str, set] = {}
    for fn, did in meta.values():
        doc_copies.setdefault(fn, set()).add(did)
    duplicated = {fn for fn, ids_ in doc_copies.items() if len(ids_) > 1}
    print(f"Duplicated files in aerospace: {sorted(duplicated)}\n")

    searcher = LocalSearch()

    for qid in ids:
        q = next((x for x in golden["questions"] if x["id"] == qid), None)
        if not q:
            print(f"!! {qid} not found")
            continue

        print("=" * 78)
        print(f"{qid}  ({q['type']})  {q['question']}")
        print(f"  expected citations: {q.get('expected_citations')}")
        print("=" * 78)

        result = await searcher.search(q["question"], tenant="aerospace")
        chunks = result.get("chunks", result) if isinstance(result, dict) else result

        seen_docids: Counter = Counter()
        for i, c in enumerate(chunks, 1):
            cid = c.get("chunk_id")
            fn, did = meta.get(cid, ("?", "?"))
            seen_docids[(fn, did)] += 1
            dup_mark = "  <DUP-FILE>" if fn in duplicated else ""
            score = c.get("score") or c.get("rerank_score") or 0.0
            text = (c.get("text") or "").replace("\n", " ")[:95]
            print(f"  {i:2d}. {fn:36s} score={float(score):7.4f}{dup_mark}")
            print(f"      {text}")

        # how many slots went to duplicate copies of the same file
        by_file: Counter = Counter(fn for (fn, _did) in seen_docids)
        wasted = sum(n - 1 for n in by_file.values() if n > 1)
        print(f"\n  slots: {len(chunks)}   distinct files: {len(by_file)}   "
              f"slots lost to same-file repeats: {wasted}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
