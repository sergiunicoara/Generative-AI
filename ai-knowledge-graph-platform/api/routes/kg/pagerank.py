"""PageRank centrality endpoints — graph-wide entity importance via Neo4j GDS."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.auth.dependencies import require_scope
from graphrag.graph.neo4j_client import get_neo4j

router = APIRouter()


@router.post(
    "/pagerank/compute",
    dependencies=[Depends(require_scope("write"))],
    summary="Compute PageRank centrality and persist scores onto Entity nodes",
)
async def compute_pagerank(tenant: str = "default"):
    from graphrag.graph.pagerank import PageRankComputer
    return await PageRankComputer(tenant=tenant).compute_and_persist()


@router.get(
    "/pagerank/top-entities",
    dependencies=[Depends(require_scope("read"))],
    summary="List the most central entities by PageRank score",
)
async def top_pagerank_entities(tenant: str = "default", top_k: int = 20):
    return {"entities": await get_neo4j().get_top_entities_by_pagerank(tenant, top_k)}
