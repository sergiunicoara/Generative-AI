"""Entry point: starts EvaluationConsumer with graceful SIGTERM shutdown."""

import asyncio
import signal
import structlog

from graphrag.messaging.consumers import EvaluationConsumer

log = structlog.get_logger(__name__)


async def main():
    log.info("evaluation_worker.starting")
    consumer = EvaluationConsumer()
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
        log.info("evaluation_worker.shutdown_graceful")


if __name__ == "__main__":
    asyncio.run(main())
