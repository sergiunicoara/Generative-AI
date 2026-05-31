"""Entry point: starts QueryConsumer with graceful SIGTERM shutdown."""

import asyncio
import signal
import structlog

from graphrag.messaging.consumers import QueryConsumer

log = structlog.get_logger(__name__)


async def _ensure_schema():
    """Wait for Neo4j and initialize schema using get_neo4j() to warm the global pool."""
    import asyncio
    from pathlib import Path
    from graphrag.graph.neo4j_client import get_neo4j
    schema_path = Path(__file__).parents[1] / "graphrag" / "graph" / "schema.cypher"
    statements = schema_path.read_text(encoding="utf-8").split(";")

    # Retry until the global Neo4j client can connect
    for attempt in range(30):
        try:
            client = get_neo4j()
            await client.run("RETURN 1")  # warm the global pool
            break
        except Exception as e:
            log.info("query_worker.schema_waiting", attempt=attempt + 1, error=str(e)[:80])
            await asyncio.sleep(10)
    else:
        log.warning("query_worker.schema_neo4j_unreachable")
        return

    # Now run schema with the warmed global pool
    client = get_neo4j()
    for stmt in statements:
        stmt = stmt.strip()
        if stmt and not stmt.startswith("--"):
            try:
                await client.run(stmt)
            except Exception as e:
                log.warning("query_worker.schema_warn", error=str(e)[:120])
    log.info("query_worker.schema_ready")


async def main():
    log.info("query_worker.starting")
    await _ensure_schema()
    consumer = QueryConsumer()
    task = asyncio.create_task(consumer.start())

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda s=sig.name: (log.info("worker.signal_received", signal=s), task.cancel()),
        )

    try:
        await task
    except asyncio.CancelledError:
        log.info("query_worker.shutdown_graceful")


if __name__ == "__main__":
    asyncio.run(main())
