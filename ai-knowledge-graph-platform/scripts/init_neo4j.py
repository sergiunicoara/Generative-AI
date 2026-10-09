"""Initialize Neo4j constraints and vector indexes (idempotent)."""

import asyncio

from neo4j import AsyncGraphDatabase

from graphrag.core.config import get_settings
from graphrag.graph.schema_statements import load_schema_statements


async def main():
    cfg = get_settings()
    driver = AsyncGraphDatabase.driver(
        cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password)
    )

    async with driver.session() as session:
        for stmt in load_schema_statements():
            try:
                result = await session.run(stmt)
                await result.consume()  # DDL is lazy — must consume to execute
                print(f"OK: {stmt[:60]}...")
            except Exception as e:
                print(f"WARN: {e}")

    await driver.close()
    print("Neo4j schema initialized.")


if __name__ == "__main__":
    asyncio.run(main())
