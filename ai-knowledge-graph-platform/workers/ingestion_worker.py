"""Entry point: starts IngestionConsumer with graceful SIGTERM shutdown.

On SIGTERM or SIGINT the consumer task is cancelled.  aio-pika's queue
iterator exits cleanly at the next await boundary so the message currently
being processed (if any) finishes before the process exits.  Any unacked
message is requeued by RabbitMQ after the consumer disconnects.
"""

import asyncio
import io
import os
import signal
import sys

# On Windows, stdout/stderr default to the ANSI codepage (cp1252), which
# raises UnicodeEncodeError on non-ASCII text (e.g. Romanian diacritics) in
# log messages — an unhandled UnicodeEncodeError here crashes the whole
# consumer process, silently killing the RabbitMQ consume loop after the
# first such message (see scripts/ingest_corpus.py for the same fix).
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import structlog

from graphrag.messaging.consumers import IngestionConsumer
from graphrag.workers.health_server import HealthServer

log = structlog.get_logger(__name__)

HEALTH_PORT = int(os.getenv("WORKER_HEALTH_PORT", "8081"))


async def _ensure_schema():
    """Initialize Neo4j schema (idempotent) using get_neo4j() to warm the global pool."""
    import asyncio
    from graphrag.graph.neo4j_client import get_neo4j
    from graphrag.graph.schema_statements import load_schema_statements
    statements = load_schema_statements()

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
        try:
            await client.run(stmt)
        except Exception as e:
            log.warning("ingestion_worker.schema_warn", error=str(e)[:120])
    try:
        from graphrag.graph.schema_registry import startup_drift_check
        await startup_drift_check(client)
    except Exception as e:
        log.warning("ingestion_worker.schema_drift_check_failed", error=str(e)[:120])
    log.info("ingestion_worker.schema_ready")


async def main():
    log.info("ingestion_worker.starting")
    health = HealthServer(port=HEALTH_PORT, worker_name="ingestion_worker")
    await health.start()

    await _ensure_schema()
    consumer = IngestionConsumer()
    task = asyncio.create_task(consumer.start())

    health.set_ready()

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
    finally:
        await health.stop()
        from graphrag.core.lifecycle import close_shared_resources
        await close_shared_resources()


if __name__ == "__main__":
    asyncio.run(main())
