"""One flaky/failing chunk must not invalidate an otherwise-successful document.

Regression for tasks/lessons.md A138: `extract()` used bare `asyncio.gather()`
over per-chunk LLM extraction calls (entity/relation extraction and, separately,
intelligence-artifact extraction). A single chunk raising -- for any reason,
including exceptions unrelated to the extraction logic itself -- propagated out
of `gather()` and killed extraction for the whole document, discarding every
other chunk's already-successful results.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graphrag.agents.ingestion_agent import IngestionAgent
from graphrag.core.models import Document, Entity, IngestMessage, Relation


def _make_agent(extract_side_effect, artifact_side_effect=None):
    agent = IngestionAgent.__new__(IngestionAgent)
    agent._embedder = AsyncMock()
    agent._embedder.embed_chunks = AsyncMock(side_effect=lambda chunks: chunks)
    agent._embedder.embed_texts = AsyncMock(side_effect=lambda texts: [[0.0] for _ in texts])
    agent._extractor = AsyncMock()
    agent._extractor.extract = AsyncMock(side_effect=extract_side_effect)
    if artifact_side_effect is not None:
        agent._artifact_extractor = AsyncMock()
        agent._artifact_extractor.extract = AsyncMock(side_effect=artifact_side_effect)
    else:
        agent._artifact_extractor = None
    return agent


def _make_message(paragraphs: int = 4) -> IngestMessage:
    text = "\n\n".join(f"Paragraph {i} with enough content to form its own chunk." * 20 for i in range(paragraphs))
    doc = Document(filename="f.txt", source_path="f.txt", raw_text=text, tenant="automotive")
    return IngestMessage(document=doc)


@pytest.mark.asyncio
async def test_one_failing_chunk_does_not_invalidate_the_document():
    entity = Entity(name="Acme", type="SUPPLIER", description="d")
    relation = Relation(source_entity_id=entity.id, target_entity_id=entity.id, relation="RELATES_TO")

    calls = {"n": 0}

    async def flaky_extract(chunk):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("Missing credentials. Please pass an api_key, workload_identity, admin_api_key, or set the OPENAI_API_KEY or OPENAI_ADMIN_KEY environment variable.")
        return [entity], [relation]

    agent = _make_agent(flaky_extract)
    message = _make_message()

    result = await agent.extract(message)

    assert len(result["extraction_results"]) == len(result["chunks"])
    failed = [er for er in result["extraction_results"] if er == ([], [])]
    succeeded = [er for er in result["extraction_results"] if er != ([], [])]
    assert len(failed) == 1
    assert len(succeeded) == len(result["chunks"]) - 1
    assert result["manifest"].stage_metrics["extraction"]["chunk_failures"] == 1


@pytest.mark.asyncio
async def test_one_failing_artifact_extraction_does_not_invalidate_the_document():
    entity = Entity(name="Acme", type="SUPPLIER", description="d")
    relation = Relation(source_entity_id=entity.id, target_entity_id=entity.id, relation="RELATES_TO")

    async def ok_extract(chunk):
        return [entity], [relation]

    calls = {"n": 0}

    async def flaky_artifacts(chunk, entity_names):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return []

    agent = _make_agent(ok_extract, artifact_side_effect=flaky_artifacts)
    message = _make_message()

    result = await agent.extract(message)

    assert len(result["artifact_results"]) == len(result["chunks"])
    assert result["manifest"].stage_metrics["extraction"]["artifact_chunk_failures"] == 1
    # entity/relation extraction for every chunk still succeeded despite the
    # unrelated artifact-extraction failure
    assert all(er != ([], []) for er in result["extraction_results"])
