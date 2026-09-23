"""Unit tests for scripts/replay_dlq.py -- the DLQ full-body-replay fix.

Before this fix, a DLQ envelope only kept an 8-key/80-char payload_summary --
every dead-lettered message was permanently unreplayable. The consumer now
also writes `original_body_b64` (see graphrag/messaging/rabbitmq_client.py's
`consume()`), and this script decodes and republishes it.
"""

from __future__ import annotations

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
