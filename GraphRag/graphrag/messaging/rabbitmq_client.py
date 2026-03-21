"""aio-pika connection pool, publisher and consumer base."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Awaitable

import aio_pika
import structlog
from aio_pika import ExchangeType, Message, DeliveryMode
from aio_pika.pool import Pool

from graphrag.core.config import get_settings
from graphrag.core.exceptions import MessagingError

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
        log.info("rabbitmq.connected")

    async def close(self):
        if self._channel_pool:
            await self._channel_pool.close()
        if self._connection_pool:
            await self._connection_pool.close()
        log.info("rabbitmq.closed")

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
            message = Message(
                body,
                delivery_mode=DeliveryMode.PERSISTENT,
                priority=priority,
            )
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
            dlq = await channel.declare_queue(dlq_name, durable=True)

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
                    try:
                        payload = json.loads(message.body)
                        await handler(payload)
                        await message.ack()
                    except Exception as exc:
                        log.error(
                            "rabbitmq.handler_error",
                            error=str(exc),
                            retries=retries,
                        )
                        if retries < MAX_RETRIES:
                            # Republish with incremented retry count instead of nack+requeue
                            new_headers = dict(message.headers or {})
                            new_headers["x-retry-count"] = retries + 1
                            retry_msg = Message(
                                message.body,
                                delivery_mode=message.delivery_mode,
                                headers=new_headers,
                            )
                            await channel.default_exchange.publish(
                                retry_msg, routing_key=queue_name
                            )
                            await message.ack()
                        else:
                            log.error("rabbitmq.dlq_sent", queue=dlq_name)
                            await message.nack(requeue=False)


_client: RabbitMQClient | None = None


async def get_rabbitmq() -> RabbitMQClient:
    global _client
    if _client is None:
        _client = RabbitMQClient()
        await _client.connect()
    return _client
