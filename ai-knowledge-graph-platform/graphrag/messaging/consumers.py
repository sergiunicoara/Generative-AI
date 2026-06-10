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
        from graphrag.retrieval.query_cache import QueryCache
        agent = QueryAgent()
        cache = QueryCache()
        await cache.connect()
        mq = await get_rabbitmq()
        eval_sample_rate = get_settings().evaluation.get("eval_sample_rate", 0.2)

        async def handle(payload: dict):
            msg = QueryMessage(**payload)
            tenant     = getattr(msg, "tenant",     "default")
            session_id = getattr(msg, "session_id", "") or ""

            # ── Cache pre-check (O(1) on hit; skips all 6 retrieval stages) ──
            cached = await cache.get(msg.question, tenant=tenant, session_id=session_id)
            if cached:
                log.info("query_consumer.cache_hit", query_id=msg.query_id)
                from graphrag.retrieval.result_store import get_result_store
                await get_result_store().set(msg.query_id, {
                    "status":     "completed",
                    "query_id":   msg.query_id,
                    "answer":     cached["answer"],
                    "citations":  cached.get("citations", []),
                    "latency_ms": 0,
                    "cache_hit":  True,
                })
                return

            result = await agent.run(msg)

            # ── Cache the result (provenance-aware: keyed by cited entity names) ──
            await cache.set(
                query=msg.question,
                tenant=tenant,
                result={
                    "answer":    result.answer,
                    "contexts":  result.contexts,
                    "citations": result.citations,
                },
                entities_used=list(result.citations),
                session_id=session_id,
            )

            # Persist result via Redis-backed ResultStore so the API process
            # (a separate container) can read it.
            from graphrag.retrieval.result_store import get_result_store
            await get_result_store().set(msg.query_id, {
                "status":     "completed",
                "query_id":   msg.query_id,
                "answer":     result.answer,
                "citations":  result.citations,
                "latency_ms": result.latency_ms,
            })

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
