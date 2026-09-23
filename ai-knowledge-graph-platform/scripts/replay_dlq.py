#!/usr/bin/env python
"""Replay messages sitting in a RabbitMQ dead-letter queue back onto their
original work queue.

Requires the DLQ envelope to carry ``original_body_b64`` (added when the
consumer sends a message to its DLQ after exhausting retries --
graphrag/messaging/rabbitmq_client.py's ``consume()``). Older DLQ messages
written before that field existed only have a truncated ``payload_summary``
and cannot be replayed -- this script reports and skips those rather than
guessing at a reconstructed body.

Usage
-----
    # Inspect what's sitting in a DLQ without moving anything
    python scripts/replay_dlq.py graphrag.ingest.queue --dry-run

    # Replay everything currently in the DLQ back onto the original queue
    python scripts/replay_dlq.py graphrag.ingest.queue

    # Replay at most N messages (useful for testing a fix on a small batch
    # before draining the whole backlog)
    python scripts/replay_dlq.py graphrag.ingest.queue --limit 5

Exit codes
----------
    0  Every replayable message was requeued (or --dry-run found none/some)
    1  At least one message could not be replayed (missing original_body_b64,
       decode failure, or a publish error)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import structlog

log = structlog.get_logger("replay_dlq")


async def replay(queue_name: str, dry_run: bool, limit: int | None) -> int:
    from aio_pika import DeliveryMode, Message
    from graphrag.messaging.rabbitmq_client import get_rabbitmq

    client = await get_rabbitmq()
    dlq_name = f"{queue_name}.dlq"

    replayed = 0
    skipped = 0
    errors = 0

    print(f"\n{'='*60}")
    print(f"  DLQ replay: {dlq_name} -> {queue_name}{'  (dry run)' if dry_run else ''}")
    print(f"{'='*60}\n")

    async with client._channel_pool.acquire() as channel:  # noqa: SLF001 - ops script, same pool the client itself uses
        dlq = await channel.declare_queue(dlq_name, durable=True)

        while limit is None or replayed + skipped + errors < limit:
            message = await dlq.get(fail=False)
            if message is None:
                break  # queue drained

            try:
                envelope = json.loads(message.body)
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR  unparseable DLQ envelope (message_id={message.message_id}): {exc}")
                errors += 1
                await message.nack(requeue=True)  # leave it in the DLQ, don't lose it
                continue

            original_b64 = envelope.get("original_body_b64")
            msg_id = envelope.get("message_id", "?")
            if not original_b64:
                print(f"  SKIP   message_id={msg_id}: no original_body_b64 "
                      f"(written before replay support existed) -- payload_summary={envelope.get('payload_summary')}")
                skipped += 1
                await message.nack(requeue=True)  # leave it for manual triage
                continue

            try:
                original_body = base64.b64decode(original_b64)
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR  message_id={msg_id}: base64 decode failed: {exc}")
                errors += 1
                await message.nack(requeue=True)
                continue

            reason = envelope.get("error", "")[:80]
            print(f"  {'WOULD REPLAY' if dry_run else 'REPLAY'}  message_id={msg_id}  "
                  f"original_reason={envelope.get('exception_type')}: {reason}")

            if dry_run:
                await message.nack(requeue=True)  # dry run must not drain the queue
                replayed += 1
                continue

            headers = dict(envelope.get("original_headers") or {})
            headers.pop("x-retry-count", None)  # give it a clean retry budget on replay
            headers.pop("x-last-error", None)
            headers.pop("x-exception-type", None)
            replay_msg = Message(
                original_body,
                delivery_mode=DeliveryMode.PERSISTENT,
                priority=envelope.get("original_priority") or 0,
                headers=headers or None,
                correlation_id=envelope.get("original_correlation_id"),
            )
            await channel.default_exchange.publish(replay_msg, routing_key=queue_name)
            await message.ack()  # only remove from the DLQ once safely republished
            replayed += 1

    print(f"\n{'='*60}")
    print(f"  Replayed: {replayed}   Skipped (unreplayable): {skipped}   Errors: {errors}")
    print(f"{'='*60}\n")

    return 1 if (skipped or errors) else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("queue", help="Original queue name, e.g. graphrag.ingest.queue "
                                       "(its DLQ is <queue>.dlq)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Report what would be replayed without moving any message")
    parser.add_argument("--limit", type=int, default=None, help="Replay at most N messages")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(replay(args.queue, args.dry_run, args.limit)))


if __name__ == "__main__":
    main()
