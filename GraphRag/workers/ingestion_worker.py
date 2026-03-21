"""Entry point: starts IngestionConsumer event loop."""

import asyncio
import structlog

from graphrag.messaging.consumers import IngestionConsumer

log = structlog.get_logger(__name__)


async def main():
    log.info("ingestion_worker.starting")
    consumer = IngestionConsumer()
    await consumer.start()


if __name__ == "__main__":
    asyncio.run(main())
