"""Unit tests for scripts/replay_dlq.py -- the DLQ full-body-replay fix.

Before this fix, a DLQ envelope only kept an 8-key/80-char payload_summary --
every dead-lettered message was permanently unreplayable. The consumer now
also writes `original_body_b64` (see graphrag/messaging/rabbitmq_client.py's
`consume()`), and this script decodes and republishes it.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import replay_dlq  # noqa: E402


def _envelope(**overrides) -> bytes:
    base = {
        "dlq_reason": "max_retries_exceeded",
        "exception_type": "ValueError",
        "error": "boom",
        "retry_count": 3,
        "queue": "graphrag.ingest.queue",
        "message_id": "msg-1",
        "payload_summary": {"a": 1},
        "original_body_b64": base64.b64encode(json.dumps({"a": 1, "b": "x"}).encode()).decode(),
        "original_headers": {"x-retry-count": 3, "x-last-error": "boom", "x-exception-type": "ValueError"},
        "original_priority": 0,
        "original_correlation_id": "corr-1",
    }
    base.update(overrides)
    return json.dumps(base).encode()


def _fake_client_with_dlq_messages(messages: list[bytes]):
    """Build a client whose channel_pool yields a dlq queue that returns
    `messages` in order via `.get(fail=False)`, then None once exhausted."""
    remaining = list(messages)

    async def _get(fail=False):
        if not remaining:
            return None
        body = remaining.pop(0)
        msg = MagicMock()
        msg.body = body
        msg.message_id = "some-id"
        # aio_pika defaults these to None when the publisher didn't set them --
        # an un-set MagicMock attribute would instead auto-vivify a truthy
        # MagicMock, which breaks the `message.headers.get("x-death")` check.
        msg.headers = None
        msg.priority = 0
        msg.correlation_id = None
        msg.ack = AsyncMock()
        msg.nack = AsyncMock()
        return msg

    dlq_queue = MagicMock()
    dlq_queue.get = AsyncMock(side_effect=_get)

    channel = MagicMock()
    channel.declare_queue = AsyncMock(return_value=dlq_queue)
    channel.default_exchange = MagicMock()
    channel.default_exchange.publish = AsyncMock()

    context = AsyncMock()
    context.__aenter__.return_value = channel
    pool = MagicMock()
    pool.acquire.return_value = context

    client = MagicMock()
    client._channel_pool = pool
    return client, channel, dlq_queue


def _fake_client_with_requeue_semantics(messages: list[bytes]):
    """Like `_fake_client_with_dlq_messages`, but a nack(requeue=True) really
    puts the message back into the queue (appended to the tail) instead of
    discarding it -- the behaviour the infinite-loop bug depended on. Only
    ack() permanently removes a message. Used to prove replay() terminates
    even when every message it sees gets nacked back."""
    pending: list[bytes] = list(messages)

    async def _get(fail=False):
        if not pending:
            return None
        body = pending.pop(0)
        msg = MagicMock()
        msg.body = body
        msg.message_id = "some-id"
        msg.headers = None
        msg.priority = 0
        msg.correlation_id = None

        async def _nack(requeue=True):
            if requeue:
                pending.append(body)

        msg.ack = AsyncMock()
        msg.nack = AsyncMock(side_effect=_nack)
        return msg

    dlq_queue = MagicMock()
    dlq_queue.get = AsyncMock(side_effect=_get)

    channel = MagicMock()
    channel.declare_queue = AsyncMock(return_value=dlq_queue)
    channel.default_exchange = MagicMock()
    channel.default_exchange.publish = AsyncMock()

    context = AsyncMock()
    context.__aenter__.return_value = channel
    pool = MagicMock()
    pool.acquire.return_value = context

    client = MagicMock()
    client._channel_pool = pool
    return client, channel, dlq_queue


class TestReplayDlqTerminatesWithRequeueSemantics:
    """Regression tests for the infinite-loop bug: nack(requeue=True) puts a
    message straight back at the front of the queue, so any code path that
    only ever nacks (dry-run, an unparseable envelope, a missing
    original_body_b64) must still terminate rather than re-fetching the same
    message forever."""

    async def test_dry_run_with_no_limit_terminates_and_visits_each_once(self):
        envelopes = [_envelope(message_id="a"), _envelope(message_id="b"), _envelope(message_id="c")]
        client, channel, _ = _fake_client_with_requeue_semantics(envelopes)
        with patch("graphrag.messaging.rabbitmq_client.get_rabbitmq", AsyncMock(return_value=client)):
            exit_code = await asyncio.wait_for(
                replay_dlq.replay("graphrag.ingest.queue", dry_run=True, limit=None),
                timeout=5,
            )

        assert exit_code == 0
        channel.default_exchange.publish.assert_not_awaited()

    async def test_single_unparseable_message_does_not_loop_forever(self):
        client, channel, _ = _fake_client_with_requeue_semantics([b"not json"])
        with patch("graphrag.messaging.rabbitmq_client.get_rabbitmq", AsyncMock(return_value=client)):
            exit_code = await asyncio.wait_for(
                replay_dlq.replay("graphrag.ingest.queue", dry_run=False, limit=None),
                timeout=5,
            )

        assert exit_code == 1  # unreplayable, but must still terminate

    async def test_message_without_original_body_does_not_loop_forever(self):
        envelope = json.loads(_envelope())
        del envelope["original_body_b64"]
        client, channel, _ = _fake_client_with_requeue_semantics([json.dumps(envelope).encode()])
        with patch("graphrag.messaging.rabbitmq_client.get_rabbitmq", AsyncMock(return_value=client)):
            exit_code = await asyncio.wait_for(
                replay_dlq.replay("graphrag.ingest.queue", dry_run=False, limit=None),
                timeout=5,
            )

        assert exit_code == 1

    async def test_mixed_batch_all_distinct_messages_are_still_visited(self):
        """A bad message that keeps getting requeued must not starve the
        good messages behind it in the same run."""
        bad = b"not json"
        good = _envelope(message_id="good-1")
        client, channel, _ = _fake_client_with_requeue_semantics([bad, good])
        with patch("graphrag.messaging.rabbitmq_client.get_rabbitmq", AsyncMock(return_value=client)):
            await asyncio.wait_for(
                replay_dlq.replay("graphrag.ingest.queue", dry_run=False, limit=None),
                timeout=5,
            )

        channel.default_exchange.publish.assert_awaited_once()


def _fake_client_with_raw_messages(messages: list[tuple[bytes, dict]]):
    """Like `_fake_client_with_dlq_messages`, but each entry is a raw
    (body, headers) pair rather than a JSON envelope -- for simulating a
    message RabbitMQ itself dead-lettered (TTL expiry etc.), where the body
    is the original untouched payload and headers carry `x-death`."""
    remaining = list(messages)

    async def _get(fail=False):
        if not remaining:
            return None
        body, headers = remaining.pop(0)
        msg = MagicMock()
        msg.body = body
        msg.message_id = None
        msg.headers = headers
        msg.priority = 0
        msg.correlation_id = None
        msg.ack = AsyncMock()
        msg.nack = AsyncMock()
        return msg

    dlq_queue = MagicMock()
    dlq_queue.get = AsyncMock(side_effect=_get)

    channel = MagicMock()
    channel.declare_queue = AsyncMock(return_value=dlq_queue)
    channel.default_exchange = MagicMock()
    channel.default_exchange.publish = AsyncMock()

    context = AsyncMock()
    context.__aenter__.return_value = channel
    pool = MagicMock()
    pool.acquire.return_value = context

    client = MagicMock()
    client._channel_pool = pool
    return client, channel, dlq_queue


class TestReplayDlqBrokerExpiredMessages:
    """Not fixed #4 (audit-2026-09-23.md): a RabbitMQ TTL expiry dead-letters
    a message with its raw original body and an `x-death` header, not our
    consumer's envelope -- this used to be misreported as "no
    original_body_b64 (written before replay support existed)" and skipped
    forever."""

    async def test_x_death_message_is_replayed_using_raw_body(self):
        raw_body = json.dumps({"job": "ingest", "doc": "AD-2024.txt"}).encode()
        headers = {"x-death": [{"reason": "expired", "queue": "graphrag.ingest.queue"}]}
        client, channel, _ = _fake_client_with_raw_messages([(raw_body, headers)])
        with patch("graphrag.messaging.rabbitmq_client.get_rabbitmq", AsyncMock(return_value=client)):
            exit_code = await replay_dlq.replay("graphrag.ingest.queue", dry_run=False, limit=None)

        assert exit_code == 0
        channel.default_exchange.publish.assert_awaited_once()
        published_msg, kwargs = channel.default_exchange.publish.call_args
        assert kwargs["routing_key"] == "graphrag.ingest.queue"
        assert published_msg[0].body == raw_body

    async def test_x_death_header_is_stripped_from_the_replayed_message(self):
        raw_body = b'{"a": 1}'
        headers = {"x-death": [{"reason": "expired"}], "x-retry-count": 3}
        client, channel, _ = _fake_client_with_raw_messages([(raw_body, headers)])
        with patch("graphrag.messaging.rabbitmq_client.get_rabbitmq", AsyncMock(return_value=client)):
            await replay_dlq.replay("graphrag.ingest.queue", dry_run=False, limit=None)

        published_msg, _ = channel.default_exchange.publish.call_args
        replayed_headers = published_msg[0].headers or {}
        assert "x-death" not in replayed_headers
        assert "x-retry-count" not in replayed_headers

    async def test_message_with_neither_envelope_nor_x_death_is_still_skipped(self):
        """A genuinely legacy/unreplayable message (no envelope, no x-death)
        must still be reported and skipped, not misinterpreted as replayable."""
        client, channel, _ = _fake_client_with_raw_messages([(b"not json, no x-death", None)])
        with patch("graphrag.messaging.rabbitmq_client.get_rabbitmq", AsyncMock(return_value=client)):
            exit_code = await replay_dlq.replay("graphrag.ingest.queue", dry_run=False, limit=None)

        assert exit_code == 1
        channel.default_exchange.publish.assert_not_awaited()


class TestReplayDlq:
    async def test_message_with_full_body_is_decoded_and_republished(self):
        client, channel, _ = _fake_client_with_dlq_messages([_envelope()])
        with patch("graphrag.messaging.rabbitmq_client.get_rabbitmq", AsyncMock(return_value=client)):
            exit_code = await replay_dlq.replay("graphrag.ingest.queue", dry_run=False, limit=None)

        assert exit_code == 0
        channel.default_exchange.publish.assert_awaited_once()
        published_msg, kwargs = channel.default_exchange.publish.call_args
        assert kwargs["routing_key"] == "graphrag.ingest.queue"
        # The republished body must be the real original payload, not the
        # truncated summary.
        assert json.loads(published_msg[0].body) == {"a": 1, "b": "x"}

    async def test_replayed_message_does_not_carry_stale_retry_headers(self):
        client, channel, _ = _fake_client_with_dlq_messages([_envelope()])
        with patch("graphrag.messaging.rabbitmq_client.get_rabbitmq", AsyncMock(return_value=client)):
            await replay_dlq.replay("graphrag.ingest.queue", dry_run=False, limit=None)

        published_msg, _ = channel.default_exchange.publish.call_args
        headers = published_msg[0].headers or {}
        assert "x-retry-count" not in headers
        assert "x-last-error" not in headers

    async def test_message_without_original_body_is_skipped_not_dropped(self):
        envelope = json.loads(_envelope())
        del envelope["original_body_b64"]
        client, channel, dlq_queue = _fake_client_with_dlq_messages([json.dumps(envelope).encode()])
        with patch("graphrag.messaging.rabbitmq_client.get_rabbitmq", AsyncMock(return_value=client)):
            exit_code = await replay_dlq.replay("graphrag.ingest.queue", dry_run=False, limit=None)

        assert exit_code == 1  # unreplayable messages must be visible as a failure, not silently 0
        channel.default_exchange.publish.assert_not_awaited()

    async def test_dry_run_never_publishes_or_drains_the_queue(self):
        client, channel, _ = _fake_client_with_dlq_messages([_envelope()])
        with patch("graphrag.messaging.rabbitmq_client.get_rabbitmq", AsyncMock(return_value=client)):
            exit_code = await replay_dlq.replay("graphrag.ingest.queue", dry_run=True, limit=None)

        assert exit_code == 0
        channel.default_exchange.publish.assert_not_awaited()

    async def test_limit_stops_after_n_messages(self):
        client, channel, _ = _fake_client_with_dlq_messages([_envelope(), _envelope(), _envelope()])
        with patch("graphrag.messaging.rabbitmq_client.get_rabbitmq", AsyncMock(return_value=client)):
            await replay_dlq.replay("graphrag.ingest.queue", dry_run=False, limit=2)

        assert channel.default_exchange.publish.await_count == 2

    async def test_empty_dlq_replays_nothing_and_succeeds(self):
        client, channel, _ = _fake_client_with_dlq_messages([])
        with patch("graphrag.messaging.rabbitmq_client.get_rabbitmq", AsyncMock(return_value=client)):
            exit_code = await replay_dlq.replay("graphrag.ingest.queue", dry_run=False, limit=None)

        assert exit_code == 0
        channel.default_exchange.publish.assert_not_awaited()
