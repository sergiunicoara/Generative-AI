"""Google ADK agent that handles query orchestration."""

from __future__ import annotations

import structlog

from graphrag.agents.base_agent import BaseGraphRAGAgent
from graphrag.core.config import get_settings
from graphrag.core.models import QueryMessage, QueryResult
from graphrag.retrieval.hybrid_retriever import HybridRetriever

log = structlog.get_logger(__name__)


class QueryAgent(BaseGraphRAGAgent):
    def __init__(self):
        self._retriever = HybridRetriever()
        super().__init__("query_agent")

    def _model(self) -> str:
        return get_settings().gemini_query_model

    def _instruction(self) -> str:
        return (
            "You are a GraphRAG query agent. Given a question, use the retrieval tools "
            "to find relevant context from the knowledge graph, then generate a "
            "well-cited, grounded answer. Prefer hybrid search unless the user specifies local or global."
        )

    def _tools(self) -> list:
        try:
            from google.adk.tools import FunctionTool
            from graphrag.agents.tools.retrieval_tools import local_search, global_search
            from graphrag.agents.tools.neo4j_tools import get_neighbors
            return [
                FunctionTool(local_search),
                FunctionTool(global_search),
                FunctionTool(get_neighbors),
            ]
        except ImportError:
            return []

    async def run(self, message: QueryMessage) -> QueryResult:
        log.info(
            "query_agent.start",
            query_id=message.query_id,
            mode=message.mode,
        )
        result = await self._retriever.retrieve_and_answer(
            question=message.question,
            mode=message.mode,
        )
        result.query_id = message.query_id
        log.info(
            "query_agent.done",
            query_id=message.query_id,
            latency_ms=round(result.latency_ms, 1),
            answer=result.answer,
            citations=result.citations,
        )
        return result
