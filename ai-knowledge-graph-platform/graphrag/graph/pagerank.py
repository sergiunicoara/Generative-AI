"""Compute PageRank centrality over the tenant's entity graph via Neo4j GDS."""

from __future__ import annotations

import structlog

from graphrag.core.config import get_settings
from graphrag.graph.neo4j_client import get_neo4j

log = structlog.get_logger(__name__)


class PageRankComputer:
    """Runs GDS PageRank for one tenant and persists scores onto Entity nodes."""

    def __init__(self, tenant: str = "default"):
        self._cfg = get_settings().graph
        self._neo4j = get_neo4j()
        self._tenant = tenant

    async def compute_and_persist(self) -> dict:
        log.info("pagerank.start", tenant=self._tenant)

        scores = await self._neo4j.run_pagerank(
            tenant=self._tenant,
            damping_factor=self._cfg.get("pagerank_damping_factor", 0.85),
            max_iterations=self._cfg.get("pagerank_iterations", 20),
        )

        if not scores:
            log.warning("pagerank.no_entities", tenant=self._tenant)
            return {"tenant": self._tenant, "entities_scored": 0, "top_entity": None}

        await self._neo4j.write_pagerank_scores(self._tenant, scores)

        log.info(
            "pagerank.done",
            tenant=self._tenant,
            entities_scored=len(scores),
            top_entity=scores[0]["name"],
            top_score=round(scores[0]["score"], 4),
        )
        return {
            "tenant": self._tenant,
            "entities_scored": len(scores),
            "top_entity": scores[0]["name"],
            "top_score": round(scores[0]["score"], 4),
        }
