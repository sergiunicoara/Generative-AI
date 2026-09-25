"""Unit tests for QueryConsumer's failure-handling fixes.

Covers two items from docs/archive/audits/audit-2026-09-23.md's "Not fixed"
list:

#7 -- an eval-job publish failure must not fail the whole handler (which
     would make the RabbitMQ consume loop requeue an already-completed,
     already-paid-for query).
#8 -- a query that raises must have its result-store status set to "failed"
     with error detail, not left at "queued" until the TTL expires.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _query_payload(**overrides) -> dict:
    payload = {
        "query_id": "q-1",
        "question": "What is the status of AD-2024?",
        "mode": "hybrid",
        "tenant": "aerospace",
    }
    payload.update(overrides)
    return payload


async def _run_query_consumer_handler(agent, store, payload, eval_sample_rate=0.0,
                                       publish_eval_job=None):
    """Start QueryConsumer, capture its `handle` closure, and invoke it once
    -- all within the same patch context, since `handle` resolves
    get_result_store()/publish_eval_job at call time, not at start() time."""
    from graphrag.messaging.consumers import QueryConsumer

    mq = MagicMock()
    captured = {}

    async def consume(*_args):
        captured["handler"] = _args[-1]

    mq.consume = consume

    settings = MagicMock()
    settings.evaluation = {"eval_sample_rate": eval_sample_rate}

    with ExitStack() as stack:
        stack.enter_context(patch("graphrag.agents.query_agent.QueryAgent", return_value=agent))
        stack.enter_context(patch("graphrag.messaging.consumers.get_rabbitmq", AsyncMock(return_value=mq)))
        stack.enter_context(patch("graphrag.messaging.consumers.get_settings", return_value=settings))
        stack.enter_context(patch("graphrag.retrieval.result_store.get_result_store", return_value=store))
        if publish_eval_job is not None:
            stack.enter_context(patch("graphrag.messaging.consumers.publish_eval_job", publish_eval_job))

        await QueryConsumer().start()
        await captured["handler"](payload)


class TestFailedQueryIsMarkedFailed:
    async def test_agent_exception_persists_failed_status_and_reraises(self):
        agent = MagicMock()
        agent.run = AsyncMock(side_effect=RuntimeError("neo4j blip"))
        store = AsyncMock()
        store.get = AsyncMock(return_value={})
        store.set = AsyncMock(return_value=None)

        with pytest.raises(RuntimeError):
            await _run_query_consumer_handler(agent, store, _query_payload())

        store.set.assert_awaited_once()
        query_id, payload = store.set.call_args.args
        assert query_id == "q-1"
        assert payload["status"] == "failed"
        assert payload["tenant"] == "aerospace"
        assert "neo4j blip" in payload["error"]
        assert payload["exception_type"] == "RuntimeError"

    async def test_result_store_write_failure_does_not_mask_the_original_exception(self):
        """If even the failed-status write can't land, the real error from
        agent.run must still be what propagates -- not a KeyError/AttributeError
        from the best-effort status write itself."""
        agent = MagicMock()
        agent.run = AsyncMock(side_effect=RuntimeError("original failure"))
        store = AsyncMock()
        store.get = AsyncMock(side_effect=ConnectionError("redis down"))
        store.set = AsyncMock(side_effect=ConnectionError("redis down"))

        with pytest.raises(RuntimeError, match="original failure"):
            await _run_query_consumer_handler(agent, store, _query_payload())

    async def test_successful_query_still_persists_completed_not_failed(self):
        from graphrag.core.models import CitationEvidence, QueryResult

        result = QueryResult(
            question="q", answer="a", citations=["doc-1"],
            evidence=[CitationEvidence(source_id="doc-1", source_label="doc-1")],
        )
        agent = MagicMock()
        agent.run = AsyncMock(return_value=result)
        store = AsyncMock()
        store.get = AsyncMock(return_value={})
        store.set = AsyncMock(return_value=None)

        await _run_query_consumer_handler(agent, store, _query_payload())

        query_id, payload = store.set.call_args.args
        assert query_id == "q-1"
        assert payload["status"] == "completed"


class TestEvalJobPublishFailureIsolation:
    async def test_eval_publish_failure_does_not_raise(self):
        from graphrag.core.models import CitationEvidence, QueryResult

        result = QueryResult(
            question="q", answer="a", citations=["doc-1"],
            evidence=[CitationEvidence(source_id="doc-1", source_label="doc-1")],
        )
        agent = MagicMock()
        agent.run = AsyncMock(return_value=result)
        store = AsyncMock()
        store.get = AsyncMock(return_value={})
        store.set = AsyncMock(return_value=None)

        # Must not raise -- the query already completed and persisted.
        await _run_query_consumer_handler(
            agent, store, _query_payload(), eval_sample_rate=1.0,
            publish_eval_job=AsyncMock(side_effect=RuntimeError("rabbitmq blip")),
        )

        # The completed result was still persisted despite eval publish failing.
        query_id, payload = store.set.call_args.args
        assert payload["status"] == "completed"
