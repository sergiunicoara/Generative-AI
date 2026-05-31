"""Entry point: starts IngestionConsumer with graceful SIGTERM shutdown.

On SIGTERM or SIGINT the consumer task is cancelled.  aio-pika's queue
iterator exits cleanly at the next await boundary so the message currently
being processed (if any) finishes before the process exits.  Any unacked
message is requeued by RabbitMQ after the consumer disconnects.
"""

import asyncio
import signal
import structlog

from graphrag.messaging.consumers import IngestionConsumer

log = structlog.get_logger(__name__)


async def _ensure_schema():
    """Initialize Neo4j schema (idempotent) using get_neo4j() to warm the global pool."""
    import asyncio
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
            log.info("ingestion_worker.schema_waiting", attempt=attempt + 1, error=str(e)[:80])
            await asyncio.sleep(10)
    else:
        log.warning("ingestion_worker.schema_neo4j_unreachable")
        return

    client = get_neo4j()
    for stmt in statements:
        stmt = stmt.strip()
        if stmt and not stmt.startswith("--"):
            try:
                await client.run(stmt)  # get_neo4j().run() already consumes
            except Exception as e:
                log.warning("ingestion_worker.schema_warn", error=str(e)[:120])
    log.info("ingestion_worker.schema_ready")


async def main():
    log.info("ingestion_worker.starting")
    await _ensure_schema()
    consumer = IngestionConsumer()
    task = asyncio.create_task(consumer.start())

    # add_signal_handler is not supported on Windows; skip on non-Unix
    import sys
    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig.name: (log.info("worker.signal_received", signal=s), task.cancel()),
            )

    try:
        await task
    except asyncio.CancelledError:
        log.info("ingestion_worker.shutdown_graceful")


if __name__ == "__main__":
    asyncio.run(main())
