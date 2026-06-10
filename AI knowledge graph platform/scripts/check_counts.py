"""Quick Neo4j count check."""
import asyncio

async def main():
    from graphrag.graph.neo4j_client import get_neo4j
    neo4j = get_neo4j()

    r1 = await neo4j.run(
        "MATCH (c:Conflict {tenant: $t}) WHERE c.status = $s RETURN count(c) AS n",
        t="aerospace", s="open"
    )
    print("Open conflicts:", r1[0]["n"])

    r2 = await neo4j.run(
        "MATCH (e:Entity {tenant: $t}) RETURN count(e) AS n", t="aerospace"
    )
    print("Entities:", r2[0]["n"])

    r3 = await neo4j.run(
        "MATCH (:Entity {tenant: $t})-[r:RELATES_TO {tenant: $t}]->(:Entity) RETURN count(r) AS n",
        t="aerospace"
    )
    print("Edges:", r3[0]["n"])

    await neo4j.close()

asyncio.run(main())
