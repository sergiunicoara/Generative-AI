"""Revert orphan_rate back to 0.0 in GraphSnapshot nodes (0.0 is correct)."""
import asyncio

async def main():
    from graphrag.graph.neo4j_client import get_neo4j
    neo4j = get_neo4j()
    T = "aerospace"

    rows = await neo4j.run(
        "MATCH (s:GraphSnapshot {tenant:$t}) "
        "RETURN s.id AS id, s.label AS lbl, s.orphan_rate AS orphan_rate",
        t=T
    )
    print(f"GraphSnapshot nodes: {len(rows)}")
    for r in rows:
        print(f"  {r['id']}  {r['lbl']}  orphan_rate={r['orphan_rate']}")

    # Revert to 0.0 — the correct value (orphan = entity not sourced from a Chunk)
    await neo4j.run(
        "MATCH (s:GraphSnapshot {tenant:$t}) SET s.orphan_rate=0.0, s.orphan_count=0",
        t=T
    )
    print("\nReverted orphan_rate to 0.0 in all GraphSnapshot nodes.")
    await neo4j.close()

asyncio.run(main())
