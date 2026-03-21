"""Initialize Neo4j constraints and vector indexes (idempotent)."""

import asyncio
from pathlib import Path

from neo4j import AsyncGraphDatabase

from graphrag.core.config import get_settings


async def main():
    cfg = get_settings()
    driver = AsyncGraphDatabase.driver(
        cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password)
    )

    schema_path = Path(__file__).parents[1] / "graphrag" / "graph" / "schema.cypher"
    statements = schema_path.read_text().split(";")

    async with driver.session() as session:
        for stmt in statements:
            stmt = stmt.strip()
            # Skip SQL-style comment lines
            if stmt and not stmt.startswith("--"):
                try:
                    await session.run(stmt)
                    print(f"OK: {stmt[:60]}...")
                except Exception as e:
                    print(f"WARN: {e}")

    await driver.close()
    print("Neo4j schema initialized.")


if __name__ == "__main__":
    asyncio.run(main())
