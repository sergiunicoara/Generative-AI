"""Minimal Neo4j schema init — only needs neo4j driver, no other deps."""
import asyncio
import os
from pathlib import Path
from neo4j import AsyncGraphDatabase


async def main():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "graphrag_dev")

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    schema_path = Path(__file__).parents[1] / "graphrag" / "graph" / "schema.cypher"
    statements = schema_path.read_text(encoding="utf-8").split(";")

    async with driver.session() as session:
        for stmt in statements:
            stmt = stmt.strip()
            if stmt and not stmt.startswith("--"):
                try:
                    await session.run(stmt)
                    print(f"OK: {stmt[:60]}...")
                except Exception as e:
                    print(f"WARN: {e}")

    await driver.close()
    print("Neo4j schema initialized.")


asyncio.run(main())
