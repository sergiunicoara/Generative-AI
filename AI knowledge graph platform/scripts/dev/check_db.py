import asyncio
from graphrag.graph.neo4j_client import get_neo4j

async def main():
    n = get_neo4j()
    chunks    = await n.run("MATCH (c:Chunk)  RETURN count(c) AS n")
    entities  = await n.run("MATCH (e:Entity) RETURN count(e) AS n")
    rels      = await n.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS n")
    emb_ents  = await n.run("MATCH (e:Entity) WHERE e.embedding IS NOT NULL AND size(e.embedding) > 0 RETURN count(e) AS n")
    print(f"Chunks          : {chunks[0]['n']}")
    print(f"Entities        : {entities[0]['n']}")
    print(f"  with embedding: {emb_ents[0]['n']}")
    print(f"Relations       : {rels[0]['n']}")
    await n.close()

asyncio.run(main())
