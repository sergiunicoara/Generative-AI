"""aio-pika connection pool, publisher and consumer base."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Callable, Awaitable

import aio_pika
import structlog
from aio_pika import ExchangeType, Message, DeliveryMode
from aio_pika.pool import Pool

from graphrag.core.config import get_settings
from graphrag.core.exceptions import MessagingError
from graphrag.observability.correlation import current_correlation_id
from graphrag.observability.operational_metrics import (
    record_consumed, record_dlq, record_message_age, record_publish, record_retry,
)

log = structlog.get_logger(__name__)

MAX_RETRIES = 3


async def _make_connection():
    cfg = get_settings()
    return await aio_pika.connect_robust(cfg.rabbitmq_url)


async def _make_channel(connection_pool: Pool):
    async with connection_pool.acquire() as connection:
        return await connection.channel()


class RabbitMQClient:
    """
    Connection + channel pool with publisher and consumer helpers.
    Uses aio-pika's robust connection (auto-reconnects on failure).
    """

    def __init__(self):
        self._connection_pool: Pool | None = None
        self._channel_pool: Pool | None = None

    async def connect(self):
        self._connection_pool = Pool(_make_connection, max_size=5)
        self._channel_pool = Pool(
            lambda: _make_channel(self._connection_pool), max_size=20
        )
        # Durable exchanges do not retain messages when no queue is bound.
        # Provision every queue before the API can publish so worker startup
        # order or a temporary worker outage cannot silently drop accepted work.
        try:
            await self.ensure_topology()
        except Exception:
            await self.close()
            raise
        log.info("rabbitmq.connected")

    async def close(self):
        if self._channel_pool:
            await self._channel_pool.close()
        if self._connection_pool:
            await self._connection_pool.close()
        self._channel_pool = None
        self._connection_pool = None
        log.info("rabbitmq.closed")

    async def ping(self) -> bool:
        """Cheap liveness check for readiness probes.

        If this process already holds a live connection pool (the normal
        case — the API process connects at startup), reuse it: acquire a
        channel, which round-trips to the broker over the existing
        connection at the same cost `publish()`/`consume()` already pay.

        If nothing is connected yet, this deliberately does NOT fall back to
        `get_rabbitmq()` / `aio_pika.connect_robust()`. That call is
        genuinely unsuitable for a readiness probe: confirmed directly
        against this project's own broker config that `RobustConnection`
        swallows `asyncio.CancelledError` inside its own reconnect state
        machine and treats it as "connection dropped, retry" rather than
        "caller gave up" — cancelling the coroutine does not stop it, it
        just spawns another 5-second retry cycle, forever, as a background
        task nothing can reach again. A readiness check that starts one of
        those on every failed poll while the broker is down is a genuine
        leak, not just a slow endpoint. `ping_reachable()` below is the
        right tool for "is anything listening at all" without touching
        aio_pika's retry machinery.
        """
        if not self._channel_pool:
            return False
        try:
            async with self._channel_pool.acquire() as channel:
                return not channel.is_closed
        except Exception as exc:  # noqa: BLE001
            log.warning("rabbitmq.ping_failed", error=str(exc)[:200])
            return False

    async def ensure_topology(self) -> None:
        """Declare all durable exchanges, work queues, bindings, and DLQs."""
        if not self._channel_pool:
            raise MessagingError("RabbitMQ not connected — call connect() first")
        from graphrag.messaging.exchanges import (
            EVAL_EXCHANGE,
            EVAL_QUEUE,
            EVAL_ROUTING_KEY,
            INGEST_EXCHANGE,
            INGEST_QUEUE,
            INGEST_ROUTING_KEY,
            QUERY_EXCHANGE,
            QUERY_QUEUE,
            QUERY_ROUTING_KEY,
        )

        topology = (
            (INGEST_EXCHANGE, INGEST_QUEUE, INGEST_ROUTING_KEY),
            (QUERY_EXCHANGE, QUERY_QUEUE, QUERY_ROUTING_KEY),
            (EVAL_EXCHANGE, EVAL_QUEUE, EVAL_ROUTING_KEY),
        )
        async with self._channel_pool.acquire() as channel:
            for exchange_name, queue_name, routing_key in topology:
                exchange = await channel.declare_exchange(
                    exchange_name, ExchangeType.TOPIC, durable=True
                )
                dlq_name = f"{queue_name}.dlq"
                await channel.declare_queue(dlq_name, durable=True)
                queue = await channel.declare_queue(
                    queue_name,
                    durable=True,
                    arguments={
                        "x-dead-letter-exchange": "",
                        "x-dead-letter-routing-key": dlq_name,
                        "x-message-ttl": 86_400_000,
                    },
                )
                await queue.bind(exchange, routing_key=routing_key)
        log.info("rabbitmq.topology_ready", queues=len(topology))

    async def publish(
        self,
        exchange_name: str,
        routing_key: str,
        payload: dict,
        priority: int = 0,
    ):
        if not self._channel_pool:
            raise MessagingError("RabbitMQ not connected — call connect() first")

        async with self._channel_pool.acquire() as channel:
            exchange = await channel.declare_exchange(
                exchange_name,
                ExchangeType.TOPIC,
                durable=True,
            )
            body = json.dumps(payload).encode()
            correlation_id = str(payload.get("correlation_id") or current_correlation_id() or "")
            headers = {"x-correlation-id": correlation_id} if correlation_id else {}
            try:
                from opentelemetry.propagate import inject
                inject(headers)
            except ImportError:
                pass
            # Stamp enqueue time so a consumer can measure how long the
            # message actually waited. Queue *depth* cannot distinguish a deep
            # queue that is draining from a shallow one that is stalled; age
            # can, which is why it is the autoscaling signal.
            headers["x-enqueued-at"] = repr(time.time())
            message = Message(
                body,
                delivery_mode=DeliveryMode.PERSISTENT,
                priority=priority,
                correlation_id=correlation_id or None,
                headers=headers or None,
            )
            with record_publish(exchange_name):
                await exchange.publish(message, routing_key=routing_key)
            log.info(
                "rabbitmq.published",
                exchange=exchange_name,
                routing_key=routing_key,
                bytes=len(body),
            )

    async def consume(
        self,
        exchange_name: str,
        queue_name: str,
        routing_key: str,
        handler: Callable[[dict], Awaitable[None]],
    ):
        """Start consuming messages from a queue. Runs until cancelled."""
        if not self._channel_pool:
            raise MessagingError("RabbitMQ not connected — call connect() first")

        async with self._channel_pool.acquire() as channel:
            await channel.set_qos(prefetch_count=1)

            exchange = await channel.declare_exchange(
                exchange_name, ExchangeType.TOPIC, durable=True
            )
            # Dead-letter queue
            dlq_name = f"{queue_name}.dlq"
            await channel.declare_queue(dlq_name, durable=True)

            queue = await channel.declare_queue(
                queue_name,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": "",
                    "x-dead-letter-routing-key": dlq_name,
                    "x-message-ttl": 86400000,  # 24h
                },
            )
            await queue.bind(exchange, routing_key=routing_key)

            log.info(
                "rabbitmq.consuming",
                exchange=exchange_name,
                queue=queue_name,
            )

            async with queue.iterator() as q:
                async for message in q:
                    retries = int(
                        message.headers.get("x-retry-count", 0)
                        if message.headers else 0
                    )
                    _enqueued_at = None
                    if message.headers:
                        try:
                            _enqueued_at = float(message.headers.get("x-enqueued-at") or 0) or None
                        except (TypeError, ValueError):
                            _enqueued_at = None
                    record_message_age(queue_name, _enqueued_at)
                    _handler_started = time.perf_counter()
                    try:
                        payload = json.loads(message.body)
                        if message.correlation_id and not payload.get("correlation_id"):
                            payload["correlation_id"] = message.correlation_id
                        otel_token = None
                        try:
                            from opentelemetry import context as otel_context
                            from opentelemetry.propagate import extract
                            otel_token = otel_context.attach(extract(dict(message.headers or {})))
                        except ImportError:
                            pass
                        try:
                            await handler(payload)
                        finally:
                            if otel_token is not None:
                                otel_context.detach(otel_token)
                        await message.ack()
                        record_consumed(
                            queue_name, "success", time.perf_counter() - _handler_started,
                        )
                    except Exception as exc:  # broad: handler may raise anything; must not kill consumer loop
                        exc_type  = type(exc).__name__
                        exc_msg   = str(exc)[:300]
                        record_consumed(
                            queue_name, "failure", time.perf_counter() - _handler_started,
                        )
                        # Summarise original payload for DLQ — truncate large fields
                        try:
                            raw_payload = json.loads(message.body)
                            payload_summary = {
                                k: (str(v)[:80] if isinstance(v, str) else v)
                                for k, v in list(raw_payload.items())[:8]
                            }
                        except Exception:
                            payload_summary = {"raw": message.body[:200].decode(errors="replace")}

                        log.error(
                            "rabbitmq.handler_error",
                            exception_type=exc_type,
                            error=exc_msg,
                            retries=retries,
                            queue=queue_name,
                            message_id=str(message.message_id or ""),
                            payload_summary=payload_summary,
                        )

                        if retries < MAX_RETRIES:
                            record_retry(queue_name, exc_type)
                            backoff_s = min(2 ** retries, 30)  # 1s, 2s, 4s… cap 30s
                            log.info(
                                "rabbitmq.retry_backoff",
                                backoff_s=backoff_s,
                                attempt=retries + 1,
                            )
                            await asyncio.sleep(backoff_s)

                            new_headers = dict(message.headers or {})
                            new_headers["x-retry-count"]    = retries + 1
                            new_headers["x-last-error"]     = exc_msg
                            new_headers["x-exception-type"] = exc_type
                            retry_msg = Message(
                                message.body,
                                delivery_mode=message.delivery_mode,
                                priority=message.priority or 0,
                                headers=new_headers,
                                correlation_id=message.correlation_id,
                            )
                            await channel.default_exchange.publish(
                                retry_msg, routing_key=queue_name
                            )
                            await message.ack()
                        else:
                            # Build structured DLQ envelope so ops can triage without
                            # parsing raw RabbitMQ headers. `payload_summary` stays
                            # for human triage (grep-able in logs without base64
                            # noise); `original_body_b64` carries the complete,
                            # untruncated message so scripts/replay_dlq.py can
                            # actually requeue it. The prior version only kept the
                            # 8-key/80-char summary, which made every DLQ message
                            # permanently unreplayable -- there was no way to
                            # reconstruct what was actually being retried.
                            import base64
                            dlq_envelope = {
                                "dlq_reason":         "max_retries_exceeded",
                                "exception_type":     exc_type,
                                "error":              exc_msg,
                                "retry_count":        retries,
                                "queue":              queue_name,
                                "message_id":         str(message.message_id or ""),
                                "payload_summary":    payload_summary,
                                "original_body_b64":  base64.b64encode(message.body).decode("ascii"),
                                "original_headers":   dict(message.headers or {}),
                                "original_priority":  message.priority or 0,
                                "original_correlation_id": message.correlation_id,
                            }
                            # Log only the triage-sized fields -- the full base64
                            # body belongs in the DLQ message itself, not in every
                            # structured log line (would bloat log storage for
                            # every large ingest payload).
                            log.error(
                                "rabbitmq.dlq_sent", dlq=dlq_name,
                                dlq_reason=dlq_envelope["dlq_reason"],
                                exception_type=dlq_envelope["exception_type"],
                                error=dlq_envelope["error"],
                                retry_count=dlq_envelope["retry_count"],
                                queue=dlq_envelope["queue"],
                                message_id=dlq_envelope["message_id"],
                                payload_summary=dlq_envelope["payload_summary"],
                            )
                            # Work being discarded permanently. A log line is
                            # not alertable at the rate an operator needs.
                            record_dlq(queue_name, exc_type)
                            dlq_msg = Message(
                                json.dumps(dlq_envelope).encode(),
                                delivery_mode=DeliveryMode.PERSISTENT,
                                headers={
                                    "x-original-queue":  queue_name,
                                    "x-exception-type":  exc_type,
                                    "x-retry-count":     retries,
                                },
                                correlation_id=message.correlation_id,
                            )
                            await channel.default_exchange.publish(dlq_msg, routing_key=dlq_name)
                            await message.ack()  # ack original so it leaves the main queue


_client: RabbitMQClient | None = None
_client_lock: asyncio.Lock | None = None


async def get_rabbitmq() -> RabbitMQClient:
    """Return the singleton RabbitMQClient, safe against concurrent cold-start.

    Without a lock, two coroutines racing on startup both pass the ``None``
    check, each create a connection pool, and one pool leaks silently.
    The inner double-check after acquiring the lock prevents the race while
    keeping the fast path (already connected) lock-free.
    """
    global _client, _client_lock
    # Lazy lock creation — asyncio.Lock() must be created inside an event loop,
    # so we can't create it at module level.  Creating it here is safe because
    # asyncio does not yield between the check and the assignment (no `await`).
    if _client_lock is None:
        _client_lock = asyncio.Lock()
    async with _client_lock:
        if _client is None:
            candidate = RabbitMQClient()
            await candidate.connect()
            _client = candidate
    return _client


async def close_rabbitmq() -> None:
    """Close and reset the process singleton when it was initialized."""
    global _client, _client_lock
    client, _client = _client, None
    if client is not None:
        await client.close()
    _client_lock = None


def get_rabbitmq_if_connected() -> RabbitMQClient | None:
    """Return the singleton client only if it already exists -- never
    connects. For callers (readiness probes) that must not trigger
    `aio_pika.connect_robust()`'s indefinite retry loop as a side effect of
    checking whether it's already up.
    """
    return _client


async def ping_reachable(url: str | None = None, timeout: float = 2.0) -> bool:
    """Raw TCP reachability check against the configured broker, bounded and
    genuinely cancellable — does not touch `aio_pika.connect_robust()` (see
    `RabbitMQClient.ping()`'s docstring for why that path can't be used
    here). Proves "something is listening on this host:port", not a full
    AMQP handshake — sufficient signal for a readiness probe.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url or get_settings().rabbitmq_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 5672
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout,
        )
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001 - best-effort close
            pass
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("rabbitmq.ping_reachable_failed", host=host, port=port, error=str(exc)[:200])
        return False
