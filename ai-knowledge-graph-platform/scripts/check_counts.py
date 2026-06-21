"""Quick Neo4j count check."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", default="aerospace")
    args = parser.parse_args()
    t = args.tenant

    from graphrag.graph.neo4j_client import get_neo4j
    neo4j = get_neo4j()

    r1 = await neo4j.run(
        "MATCH (e:Entity {tenant: $t}) RETURN count(e) AS n", t=t
    )
    print(f"Entities ({t}):", r1[0]["n"])

    r2 = await neo4j.run(
        "MATCH (:Entity {tenant: $t})-[r {tenant: $t}]->(:Entity) RETURN count(r) AS n", t=t
    )
    print(f"Edges    ({t}):", r2[0]["n"])

    r3 = await neo4j.run(
        "MATCH (c:Conflict {tenant: $t}) WHERE c.status = 'open' RETURN count(c) AS n", t=t
    )
    print(f"Conflicts({t}):", r3[0]["n"])

    await neo4j.close()


asyncio.run(main())
