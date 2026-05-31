"""Runs ingestion + query consumers concurrently on a single machine.

This ensures both workers share the same Neo4j connection (same node as Neo4j),
avoiding Fly.io cross-node Bolt protocol issues.
"""

import asyncio
import signal
import structlog

from graphrag.messaging.consumers import IngestionConsumer, QueryConsumer

log = structlog.get_logger(__name__)


async def _ensure_schema():
    """Wait for Neo4j and initialize schema using the global connection pool."""
    from pathlib import Path
    from graphrag.graph.neo4j_client import get_neo4j
    schema_path = Path(__file__).parents[1] / "graphrag" / "graph" / "schema.cypher"
    statements = schema_path.read_text(encoding="utf-8").split(";")

    for attempt in range(30):
        try:
            client = get_neo4j()
            await client.run("RETURN 1")
            break
        except Exception as e:
            log.info("combined_worker.schema_waiting", attempt=attempt + 1, error=str(e)[:80])
            await asyncio.sleep(10)
    else:
        log.warning("combined_worker.schema_neo4j_unreachable")
        return

    client = get_neo4j()
    for stmt in statements:
        stmt = stmt.strip()
        if stmt and not stmt.startswith("--"):
            try:
                await client.run(stmt)
            except Exception as e:
                log.warning("combined_worker.schema_warn", error=str(e)[:120])
    log.info("combined_worker.schema_ready")


async def main():
    log.info("combined_worker.starting")
    await _ensure_schema()

    ingest_consumer = IngestionConsumer()
    query_consumer = QueryConsumer()

    ingest_task = asyncio.create_task(ingest_consumer.start(), name="ingestion")
    query_task = asyncio.create_task(query_consumer.start(), name="query")

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda s=sig.name: (
                log.info("combined_worker.signal_received", signal=s),
                ingest_task.cancel(),
                query_task.cancel(),
            ),
        )

    try:
        await asyncio.gather(ingest_task, query_task)
    except asyncio.CancelledError:
        log.info("combined_worker.shutdown_graceful")


if __name__ == "__main__":
    asyncio.run(main())
