"""Minimal Neo4j schema init — only needs neo4j driver, no other deps."""
import asyncio
import importlib.util
import os
from pathlib import Path
from neo4j import AsyncGraphDatabase


async def main():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "graphrag_dev")

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    # Load by file path: this script must not import the graphrag package.
    spec = importlib.util.spec_from_file_location(
        "schema_statements",
        Path(__file__).parents[1] / "graphrag" / "graph" / "schema_statements.py",
    )
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    statements = loader.load_schema_statements()

    async with driver.session() as session:
        for stmt in statements:
            try:
                result = await session.run(stmt)
                await result.consume()  # DDL is lazy — must consume to execute
                print(f"OK: {stmt[:60]}...")
            except Exception as e:
                print(f"WARN: {e}")

    await driver.close()
    print("Neo4j schema initialized.")


asyncio.run(main())
