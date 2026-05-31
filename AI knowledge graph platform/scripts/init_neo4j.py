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
    statements = schema_path.read_text(encoding="utf-8").split(";")

    async with driver.session() as session:
        for stmt in statements:
            # Strip comment lines (a fragment may start with a comment
            # followed by the actual DDL statement on the next line)
            lines = [l for l in stmt.splitlines() if not l.strip().startswith("--")]
            stmt = "\n".join(lines).strip()
            if not stmt:
                continue
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
