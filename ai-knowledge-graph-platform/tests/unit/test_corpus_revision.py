from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graphrag.graph.corpus_revision import CorpusMutation
from graphrag.graph.neo4j_client import Neo4jClient


async def test_get_corpus_state_is_tenant_scoped() -> None:
    client = Neo4jClient.__new__(Neo4jClient)
    client.run = AsyncMock(return_value=[{"revision": 9, "updating": False}])

    state = await client.get_corpus_state("marketing")

    assert state == {"revision": 9, "updating": False}
    assert client.run.await_args.kwargs["tenant"] == "marketing"
    assert "tenant: $tenant" in client.run.await_args.args[0]


async def test_corpus_revision_advances_atomically() -> None:
    client = Neo4jClient.__new__(Neo4jClient)
    client.run = AsyncMock(return_value=[{"revision": 10}])

    revision = await client.complete_corpus_update("marketing")

    assert revision == 10
    cypher = client.run.await_args.args[0]
    assert "coalesce(s.revision, 0) + CASE WHEN $advance THEN 1 ELSE 0 END" in cypher
    assert client.run.await_args.kwargs["advance"] is True  # the default still advances
    assert "s.updating = remaining > 0" in cypher


async def test_begin_corpus_update_marks_tenant_updating() -> None:
    client = Neo4jClient.__new__(Neo4jClient)
    client.run = AsyncMock(return_value=[])

    await client.begin_corpus_update("marketing")

    cypher = client.run.await_args.args[0]
    assert "s.updating = true" in cypher
    assert client.run.await_args.kwargs["tenant"] == "marketing"


async def test_corpus_mutation_finalizes_after_an_operation_error() -> None:
    neo4j = AsyncMock()
    neo4j.begin_corpus_update = AsyncMock()
    neo4j.complete_corpus_update = AsyncMock(return_value=7)

    with pytest.raises(RuntimeError, match="write failed"):
        async with CorpusMutation(neo4j, "marketing", "manual_correction"):
            raise RuntimeError("write failed")

    neo4j.begin_corpus_update.assert_awaited_once_with(
        "marketing", reason="manual_correction"
    )
    neo4j.complete_corpus_update.assert_awaited_once_with(
        "marketing", reason="manual_correction", outcome="failed"
    )


async def test_corpus_mutation_fails_closed_when_completion_cannot_publish() -> None:
    neo4j = AsyncMock()
    neo4j.begin_corpus_update = AsyncMock()
    neo4j.complete_corpus_update = AsyncMock(side_effect=RuntimeError("unavailable"))

    with pytest.raises(RuntimeError, match="unavailable"):
        async with CorpusMutation(neo4j, "marketing", "pagerank_recompute"):
            pass
