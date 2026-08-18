"""Consumer loop classes for each pipeline stage."""

from __future__ import annotations

import random

import structlog

from graphrag.core.models import EvalJob, IngestMessage, QueryMessage
from graphrag.core.config import get_settings
from graphrag.core.retry import with_retry
from graphrag.messaging.exchanges import (
    EVAL_EXCHANGE, EVAL_QUEUE, EVAL_ROUTING_KEY,
    INGEST_EXCHANGE, INGEST_QUEUE, INGEST_ROUTING_KEY,
    QUERY_EXCHANGE, QUERY_QUEUE, QUERY_ROUTING_KEY,
)
from graphrag.messaging.rabbitmq_client import get_rabbitmq
from graphrag.messaging.publishers import publish_eval_job
from graphrag.retrieval.result_store import ResultStoreUnavailable

log = structlog.get_logger(__name__)


@with_retry(exceptions=(ResultStoreUnavailable,), max_attempts=3, base_delay_s=0.5)
async def _persist_final_result(store, query_id: str, payload: dict) -> None:
    """Write the completed query result, retrying transient Redis failures.

    A completed retrieval already paid its real LLM/Neo4j cost — losing the
    result to a single blip is worse than 2-3 short retries. If Redis is
    still down after that, this re-raises ResultStoreUnavailable: the caller
    (the RabbitMQ consume loop) treats an unhandled handler exception as a
    failed delivery and nacks/dead-letters the message rather than acking a
    result that was never actually persisted.
    """
    prior = await store.get(query_id) or {}
    await store.set(query_id, {**payload, "steps": prior.get("steps", [])})


class IngestionConsumer:
    async def start(self):
        from graphrag.agents.ingestion_agent import IngestionAgent
        agent = IngestionAgent()
        mq = await get_rabbitmq()

        async def handle(payload: dict):
            msg = IngestMessage(**payload)
            await agent.run(msg)
            # Calibration is deliberately scheduled after successful writes and
            # kept best-effort so a maintenance outage cannot fail ingestion.
            try:
                from graphrag.graph.calibration_scheduler import GNNCalibrationScheduler
                from graphrag.graph.neo4j_client import get_neo4j
                threshold = int(__import__("os").environ.get("GNN_CALIBRATION_THRESHOLD", "100"))
                await GNNCalibrationScheduler(get_neo4j(), threshold).maybe_schedule(
                    msg.tenant, execute=True
                )
            except Exception as exc:  # maintenance path must not nack ingestion
                log.warning("gnn_calibration.schedule_failed", error=str(exc)[:200])

        await mq.consume(INGEST_EXCHANGE, INGEST_QUEUE, INGEST_ROUTING_KEY, handle)


class QueryConsumer:
    async def start(self):
        from graphrag.agents.query_agent import QueryAgent
        agent = QueryAgent()
        mq = await get_rabbitmq()
        eval_sample_rate = get_settings().evaluation.get("eval_sample_rate", 0.2)

        async def handle(payload: dict):
            msg = QueryMessage(**payload)
            from graphrag.observability.correlation import correlation_context
            from graphrag.observability.tracing import trace_span
            with correlation_context(msg.correlation_id), trace_span(
                "query.consume", query_id=msg.query_id, tenant=msg.tenant,
                correlation_id=msg.correlation_id,
            ):
                result = await agent.run(msg)

            # Persist result via Redis-backed ResultStore so the API process
            # (a separate container) can read it. Preserve any progress steps
            # that were pushed during retrieval so the UI can render them.
            # Retries transient failures (_persist_final_result); if Redis is
            # still down after that, this raises and the message is nacked
            # rather than acked as if the result had been delivered.
            from graphrag.retrieval.result_store import get_result_store
            _store = get_result_store()
            await _persist_final_result(_store, msg.query_id, {
                "status":     "completed",
                "query_id":   msg.query_id,
                # Authorizes GET /query/{query_id}: the API compares this
                # against the caller's token tenant before returning the
                # answer. Without it the completed result overwrites the
                # "queued" entry that did carry a tenant, and the read check
                # would fail closed for the legitimate owner.
                "tenant":     msg.tenant,
                "answer":     result.answer,
                "contexts":   result.contexts,
                "citations":  result.citations,
                "latency_ms": result.latency_ms,
                "retrieval_mode": result.retrieval_mode,
                "model_version": result.model_version,
                "cache_hit": result.cache_hit,
                "cache_key": result.cache_key,
                "source_query_id": result.source_query_id,
                "source_trace_id": result.source_trace_id,
                "correlation_id": result.correlation_id,
                "routing_reason": result.routing_reason,
                "policy_result": result.policy_result,
                "policy_reason_code": result.policy_reason_code,
            })

            # Async RAGAS evaluation on sampled queries
            if random.random() < eval_sample_rate:
                eval_job = EvalJob(
                    query_result=result,
                    ground_truth=msg.ground_truth,
                    tenant=msg.tenant,
                    correlation_id=msg.correlation_id,
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
