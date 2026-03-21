"""Entry point: starts EvaluationConsumer event loop."""

import asyncio
import structlog

from graphrag.messaging.consumers import EvaluationConsumer

log = structlog.get_logger(__name__)


async def main():
    log.info("evaluation_worker.starting")
    consumer = EvaluationConsumer()
    await consumer.start()


if __name__ == "__main__":
    asyncio.run(main())
