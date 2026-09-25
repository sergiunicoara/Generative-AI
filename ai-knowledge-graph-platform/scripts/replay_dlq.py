#!/usr/bin/env python
"""Replay messages sitting in a RabbitMQ dead-letter queue back onto their
original work queue.

Replays two kinds of dead-lettered message:

1. Our own consumer's DLQ envelope, carrying ``original_body_b64`` (added
   when the consumer sends a message to its DLQ after exhausting retries --
   graphrag/messaging/rabbitmq_client.py's ``consume()``).
2. A message RabbitMQ itself dead-lettered (e.g. TTL expiry) before it ever
   reached that consumer path -- the broker attaches an ``x-death`` header
   and leaves the body untouched, so that body is replayed as-is.

A message with neither an envelope nor an ``x-death`` header cannot be
replayed -- this script reports and skips those rather than guessing at a
reconstructed body.

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

    # A message we nack(requeue=True) goes straight back into the DLQ and the
    # very next dlq.get() redelivers that same message -- there is nothing
    # else in a `--dry-run` (which never acks) or a triage skip/error path to
    # advance the queue. Without a way to notice "I've already looked at
    # this one", `--dry-run` with no --limit, or a single unparseable/
    # unreplayable message, spins forever re-reporting the same message.
    # Track the body of every message we've put back this run; once one
    # comes back around, the DLQ has cycled and every distinct message has
    # been visited exactly once, so stop instead of looping.
    seen_requeued_bodies: set[bytes] = set()

    print(f"\n{'='*60}")
    print(f"  DLQ replay: {dlq_name} -> {queue_name}{'  (dry run)' if dry_run else ''}")
    print(f"{'='*60}\n")

    async with client._channel_pool.acquire() as channel:  # noqa: SLF001 - ops script, same pool the client itself uses
        dlq = await channel.declare_queue(dlq_name, durable=True)

        while limit is None or replayed + skipped + errors < limit:
            message = await dlq.get(fail=False)
            if message is None:
                break  # queue drained

            if message.body in seen_requeued_bodies:
                await message.nack(requeue=True)  # put it back exactly as found
                print("  (DLQ has cycled back to a message already seen this run -- stopping)")
                break

            try:
                envelope = json.loads(message.body)
                if not isinstance(envelope, dict):
                    envelope = None
            except Exception:  # noqa: BLE001
                envelope = None

            original_b64 = envelope.get("original_body_b64") if envelope else None
            x_death = (message.headers or {}).get("x-death")

            if original_b64:
                # Our own consumer's DLQ envelope (graphrag/messaging/rabbitmq_client.py
                # after retries are exhausted).
                msg_id = envelope.get("message_id", "?")
                try:
                    original_body = base64.b64decode(original_b64)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ERROR  message_id={msg_id}: base64 decode failed: {exc}")
                    errors += 1
                    seen_requeued_bodies.add(message.body)
                    await message.nack(requeue=True)
                    continue
                reason = envelope.get("error", "")[:80]
                reason_label = f"original_reason={envelope.get('exception_type')}: {reason}"
                headers = dict(envelope.get("original_headers") or {})
                priority = envelope.get("original_priority") or 0
                correlation_id = envelope.get("original_correlation_id")
            elif x_death:
                # RabbitMQ itself dead-lettered this message (e.g. TTL expiry) --
                # it never passed through our consumer's DLQ-envelope producer, so
                # the body IS the original payload, untouched, with an x-death
                # header recording why the broker moved it. Previously this whole
                # class of message fell through to "no original_body_b64 (written
                # before replay support existed)" and was skipped forever -- see
                # docs/archive/audits/audit-2026-09-23.md, "Not fixed" #4.
                msg_id = message.message_id or "?"
                original_body = message.body
                death = x_death[0] if isinstance(x_death, list) and x_death else {}
                reason_label = f"broker_dead_letter reason={death.get('reason', '?')}"
                headers = dict(message.headers or {})
                headers.pop("x-death", None)
                priority = message.priority or 0
                correlation_id = message.correlation_id
            else:
                print(f"  SKIP   message_id={message.message_id or '?'}: no original_body_b64 "
                      f"and no x-death header (unreplayable envelope)")
                skipped += 1
                seen_requeued_bodies.add(message.body)
                await message.nack(requeue=True)  # leave it for manual triage
                continue

            print(f"  {'WOULD REPLAY' if dry_run else 'REPLAY'}  message_id={msg_id}  {reason_label}")

            if dry_run:
                seen_requeued_bodies.add(message.body)
                await message.nack(requeue=True)  # dry run must not drain the queue
                replayed += 1
                continue

            headers.pop("x-retry-count", None)  # give it a clean retry budget on replay
            headers.pop("x-last-error", None)
            headers.pop("x-exception-type", None)
            replay_msg = Message(
                original_body,
                delivery_mode=DeliveryMode.PERSISTENT,
                priority=priority,
                headers=headers or None,
                correlation_id=correlation_id,
            )
            await channel.default_exchange.publish(replay_msg, routing_key=queue_name)
            await message.ack()  # only remove from the DLQ once safely republished -- never
            # added to seen_requeued_bodies, since an acked message can't come back around
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
