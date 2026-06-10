"""ADK FunctionTool wrappers for local and global search."""

from __future__ import annotations

import asyncio

from graphrag.retrieval.local_search import LocalSearch
from graphrag.retrieval.global_search import GlobalSearch


def local_search(question: str) -> dict:
    """Run local chunk + entity search for a question."""
    searcher = LocalSearch()
    return asyncio.run(searcher.search(question))


def global_search(question: str) -> dict:
    """Run global community-summary search for a broad question."""
    searcher = GlobalSearch()
    return asyncio.run(searcher.search(question))
