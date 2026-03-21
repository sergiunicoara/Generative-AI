"""Consumer loop classes for each pipeline stage."""

from __future__ import annotations

import asyncio
import random

import structlog

from graphrag.core.models import EvalJob, IngestMessage, QueryMessage
from graphrag.core.config import get_settings
from graphrag.messaging.exchanges import (
    EVAL_EXCHANGE, EVAL_QUEUE, EVAL_ROUTING_KEY,
    INGEST_EXCHANGE, INGEST_QUEUE, INGEST_ROUTING_KEY,
    QUERY_EXCHANGE, QUERY_QUEUE, QUERY_ROUTING_KEY,
)
from graphrag.messaging.rabbitmq_client import get_rabbitmq
from graphrag.messaging.publishers import publish_eval_job

log = structlog.get_logger(__name__)


class IngestionConsumer:
    async def start(self):
        from graphrag.agents.ingestion_agent import IngestionAgent
        agent = IngestionAgent()
        mq = await get_rabbitmq()

        async def handle(payload: dict):
            msg = IngestMessage(**payload)
            await agent.run(msg)

        await mq.consume(INGEST_EXCHANGE, INGEST_QUEUE, INGEST_ROUTING_KEY, handle)


class QueryConsumer:
    async def start(self):
        from graphrag.agents.query_agent import QueryAgent
        agent = QueryAgent()
        mq = await get_rabbitmq()
        eval_sample_rate = get_settings().evaluation.get("eval_sample_rate", 0.2)

        async def handle(payload: dict):
            msg = QueryMessage(**payload)
            result = await agent.run(msg)

            # Async RAGAS evaluation on sampled queries
            if random.random() < eval_sample_rate:
                eval_job = EvalJob(
                    query_result=result,
                    ground_truth=msg.ground_truth,
                )
                await publish_eval_job(eval_job)

        await mq.consume(QUERY_EXCHANGE, QUERY_QUEUE, QUERY_ROUTING_KEY, handle)


class EvaluationConsumer:
    async def start(self):
        from graphrag.agents.evaluation_agent import EvaluationAgent
        agent = EvaluationAgent()
        mq = await get_rabbitmq()

        async def handle(payload: dict):
            job = EvalJob(**payload)
            await agent.run(job)

        await mq.consume(EVAL_EXCHANGE, EVAL_QUEUE, EVAL_ROUTING_KEY, handle)
