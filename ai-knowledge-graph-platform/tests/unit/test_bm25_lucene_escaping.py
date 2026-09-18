"""Regression guard for the 2026-09-18 incident: a raw question containing a
Lucene special character (observed: "/" in "LEAP-1B/CFM56") crashed
db.index.fulltext.queryNodes with a lexical TokenMgrError instead of
searching for it, because bm25_search_chunks/bm25_search_entities passed
the raw question straight into the Lucene query string with no escaping.
Surfaced by a live DRIFT-search benchmark run against real Neo4j; every
existing unit test mocks neo4j.run, so nothing ever sent a real question
through a real Lucene parser before now.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graphrag.graph.neo4j_client import Neo4jClient, _escape_lucene_query


def _client(result=None) -> Neo4jClient:
    client = Neo4jClient.__new__(Neo4jClient)
    client.run = AsyncMock(return_value=result if result is not None else [])
    return client


class TestEscapeLuceneQuery:
    def test_ordinary_text_is_unchanged(self):
        assert _escape_lucene_query("What supersedes the compliance record") == \
            "What supersedes the compliance record"

    @pytest.mark.parametrize("raw,escaped", [
        # The exact incident trigger: "/" opens a Lucene regex term, "-" is
        # Lucene's NOT operator -- both silently syntactic in ordinary text.
        ("LEAP-1B/CFM56", "LEAP\\-1B\\/CFM56"),
        ('what is "certified"?', 'what is \\"certified\\"\\?'),
        ("A&B (and C)", "A\\&B \\(and C\\)"),
        # "$" is not Lucene syntax and must not be touched -- only the
        # colon and brackets are expected to change.
        ("cost: $5 [approx]", "cost\\: $5 \\[approx\\]"),
    ])
    def test_special_characters_are_backslash_escaped(self, raw, escaped):
        assert _escape_lucene_query(raw) == escaped

    def test_every_declared_special_character_gets_escaped(self):
        from graphrag.graph.neo4j_client import _LUCENE_SPECIAL_CHARS
        for ch in _LUCENE_SPECIAL_CHARS:
            assert _escape_lucene_query(ch) == f"\\{ch}"


class TestBm25SearchEscapesBeforeQuerying:
    @pytest.mark.asyncio
    async def test_bm25_search_chunks_escapes_the_query_parameter(self):
        client = _client([])

        await client.bm25_search_chunks("LEAP-1B/CFM56 service bulletin", tenant="aerospace")

        assert client.run.await_args.kwargs["query"] == "LEAP\\-1B\\/CFM56 service bulletin"

    @pytest.mark.asyncio
    async def test_bm25_search_entities_escapes_the_query_parameter(self):
        client = _client([])

        await client.bm25_search_entities("A/C registration G-ABCD", tenant="aerospace")

        assert client.run.await_args.kwargs["query"] == "A\\/C registration G\\-ABCD"
