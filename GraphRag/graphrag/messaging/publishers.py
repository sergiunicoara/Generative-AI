"""High-level publish helpers for each pipeline stage."""

from __future__ import annotations

from graphrag.core.models import Document, EvalJob, IngestMessage, QueryMessage
from graphrag.messaging.exchanges import (
    EVAL_EXCHANGE,
    INGEST_EXCHANGE,
    QUERY_EXCHANGE,
)
from graphrag.messaging.rabbitmq_client import get_rabbitmq


async def publish_document(document: Document, priority: str = "normal") -> str:
    mq = await get_rabbitmq()
    msg = IngestMessage(document=document, priority=priority)
    await mq.publish(
        INGEST_EXCHANGE,
        routing_key=f"doc.file.{priority}",
        payload=msg.model_dump(mode="json"),
        priority=1 if priority == "high" else 0,
    )
    return msg.job_id


async def publish_query(
    question: str,
    mode: str = "hybrid",
    ground_truth: str = "",
    tenant: str = "default",
) -> str:
    mq = await get_rabbitmq()
    msg = QueryMessage(
        question=question,
        mode=mode,
        ground_truth=ground_truth,
        tenant=tenant,
    )
    await mq.publish(
        QUERY_EXCHANGE,
        routing_key=f"query.{tenant}.{mode}",
        payload=msg.model_dump(mode="json"),
    )
    return msg.query_id


async def publish_eval_job(job: EvalJob):
    mq = await get_rabbitmq()
    await mq.publish(
        EVAL_EXCHANGE,
        routing_key="eval.query_complete",
        payload=job.model_dump(mode="json"),
    )
