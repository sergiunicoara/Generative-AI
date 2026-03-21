"""Declare all RabbitMQ exchanges, queues, and dead-letter queues (idempotent)."""

import asyncio

import aio_pika
from aio_pika import ExchangeType

from graphrag.core.config import get_settings
from graphrag.messaging.exchanges import (
    EVAL_EXCHANGE, EVAL_QUEUE, EVAL_ROUTING_KEY,
    INGEST_EXCHANGE, INGEST_QUEUE, INGEST_ROUTING_KEY,
    QUERY_EXCHANGE, QUERY_QUEUE, QUERY_ROUTING_KEY,
)


async def main():
    cfg = get_settings()
    connection = await aio_pika.connect_robust(cfg.rabbitmq_url)
    channel = await connection.channel()

    configs = [
        (INGEST_EXCHANGE, INGEST_QUEUE, INGEST_ROUTING_KEY),
        (QUERY_EXCHANGE, QUERY_QUEUE, QUERY_ROUTING_KEY),
        (EVAL_EXCHANGE, EVAL_QUEUE, EVAL_ROUTING_KEY),
    ]

    for exchange_name, queue_name, routing_key in configs:
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
            },
        )
        await queue.bind(exchange, routing_key=routing_key)
        print(f"Declared: {exchange_name} → {queue_name} (DLQ: {dlq_name})")

    await connection.close()
    print("RabbitMQ initialized.")


if __name__ == "__main__":
    asyncio.run(main())
