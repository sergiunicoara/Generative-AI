"""Entry point: starts QueryConsumer event loop."""

import asyncio
import structlog

from graphrag.messaging.consumers import QueryConsumer

log = structlog.get_logger(__name__)


async def main():
    log.info("query_worker.starting")
    consumer = QueryConsumer()
    await consumer.start()


if __name__ == "__main__":
    asyncio.run(main())
