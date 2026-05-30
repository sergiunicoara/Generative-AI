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


async def main():
    log.info("ingestion_worker.starting")
    consumer = IngestionConsumer()
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
        log.info("ingestion_worker.shutdown_graceful")


if __name__ == "__main__":
    asyncio.run(main())
