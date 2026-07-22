"""Entry point: starts EvaluationConsumer with graceful SIGTERM shutdown."""

import asyncio
import io
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

from graphrag.messaging.consumers import EvaluationConsumer

log = structlog.get_logger(__name__)


async def main():
    log.info("evaluation_worker.starting")
    consumer = EvaluationConsumer()
    task = asyncio.create_task(consumer.start())

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
        log.info("evaluation_worker.shutdown_graceful")


if __name__ == "__main__":
    asyncio.run(main())
