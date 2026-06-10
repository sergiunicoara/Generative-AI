"""Unit tests for CommunitySummarizer — supersession-aware entity formatting."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from graphrag.core.models import Community
from graphrag.graph.community_summarizer import CommunitySummarizer


def _make_community() -> Community:
    return Community(level=0, member_entity_ids=["e1", "e2"], member_count=2, tenant="aerospace")


@pytest.mark.asyncio
async def test_superseded_entity_gets_annotated():
    rows = [
        {
            "name": "FAA AD 2020-05-11",
            "type": "CONCEPT",
            "description": "Requires AOA sensor bracket inspection.",
            "superseded_by": "faa-ad-2024",
        },
        {
            "name": "FAA AD 2024-01-02",
            "type": "CONCEPT",
            "description": "Mandates MCAS v2.0 software update.",
            "superseded_by": None,
        },
    ]

    with patch("graphrag.graph.community_summarizer.get_neo4j") as mock_neo4j, \
         patch("graphrag.graph.community_summarizer.get_llm") as mock_llm, \
         patch("graphrag.graph.community_summarizer.get_embedder") as mock_embedder:
        mock_neo4j.return_value.run = AsyncMock(return_value=rows)
        mock_llm.return_value.generate = AsyncMock(return_value="A summary.")
        mock_embedder.return_value.embed = AsyncMock(return_value=[[0.1, 0.2]])

        summarizer = CommunitySummarizer()
        await summarizer._summarize_one(_make_community())

        prompt = mock_llm.return_value.generate.call_args[0][0]

    assert "FAA AD 2020-05-11" in prompt
    assert "[NOTE: this directive has been superseded" in prompt
    # The current/effective directive must NOT carry the superseded note.
    current_line = next(
        line for line in prompt.splitlines() if "FAA AD 2024-01-02" in line
    )
    assert "[NOTE:" not in current_line


@pytest.mark.asyncio
async def test_non_superseded_entities_get_no_annotation():
    rows = [
        {
            "name": "Boeing 737 MAX",
            "type": "PRODUCT",
            "description": "Narrow-body airliner.",
            "superseded_by": None,
        },
    ]

    with patch("graphrag.graph.community_summarizer.get_neo4j") as mock_neo4j, \
         patch("graphrag.graph.community_summarizer.get_llm") as mock_llm, \
         patch("graphrag.graph.community_summarizer.get_embedder") as mock_embedder:
        mock_neo4j.return_value.run = AsyncMock(return_value=rows)
        mock_llm.return_value.generate = AsyncMock(return_value="A summary.")
        mock_embedder.return_value.embed = AsyncMock(return_value=[[0.1, 0.2]])

        summarizer = CommunitySummarizer()
        await summarizer._summarize_one(_make_community())

        prompt = mock_llm.return_value.generate.call_args[0][0]

    assert "[NOTE:" not in prompt
