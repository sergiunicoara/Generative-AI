"""Integration test for the full 6-step local search pipeline.

Requires:
    docker-compose up -d neo4j
    python scripts/init_neo4j.py        # (once, creates indexes)
    python workers/ingestion_worker.py  # or ingest docs directly
    GOOGLE_API_KEY set in .env

Run:
    python test_hybrid.py
"""
import os
os.environ["PYTHONUTF8"] = "1"

import asyncio
from graphrag.retrieval.local_search import LocalSearch

async def run_query(label: str, question: str):
    ls = LocalSearch()
    result = await ls.search(question)

    chunks   = result["chunks"]
    entities = result["entities"]

    print(f"\n{'='*60}")
    print(f"QUERY : {question}")
    print(f"{'='*60}")
    print(f"  Chunks  : {len(chunks)}   Entities: {len(entities)}")
    print()

    for i, c in enumerate(chunks[:6], 1):
        gnn    = c.get("gnn_score",   None)
        final  = c.get("final_score", None)
        rerank = c.get("rerank_score", None)
        score  = c.get("score", 0.0)
        via    = c.get("via_entity", "")

        score_str = ""
        if rerank is not None:
            score_str += f"rerank={rerank:.2f}  "
        else:
            score_str += f"score={score:.4f}  "
        if gnn is not None:
            score_str += f"gnn={gnn:.4f}  "
        if final is not None:
            score_str += f"final={final:.4f}"

        via_str = f"  via_entity={via}" if via else ""
        print(f"  [{i}] {score_str}{via_str}")
        print(f"      {c['text'][:100]}")
        print()


async def main():
    # Test 1: cross-document query (needs entity graph)
    await run_query(
        "cross-doc",
        "What rockets did Elon Musk's company launch?",
    )

    # Test 2: single-doc factual
    await run_query(
        "factual",
        "Who founded SpaceX?",
    )

    # Test 3: forces GAT attention — query about connected entities
    await run_query(
        "multi-hop",
        "What companies did Elon Musk found and what did they achieve?",
    )


if __name__ == "__main__":
    asyncio.run(main())
