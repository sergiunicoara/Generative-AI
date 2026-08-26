"""Human-review endpoints for proposed ontology changes discovered in ingestion."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.auth.dependencies import get_tenant, require_scope
from graphrag.graph.neo4j_client import get_neo4j
from graphrag.graph.ontology_proposals import OntologyProposalService

router = APIRouter()


@router.get(
    "/ontology/proposals",
    dependencies=[Depends(require_scope("read"))],
    summary="List tenant-scoped ontology change proposals",
)
async def list_ontology_proposals(
    tenant: str = Depends(get_tenant), status: str = "pending", limit: int = 100,
):
    return {"items": await OntologyProposalService(get_neo4j()).list(tenant, status=status, limit=limit)}


@router.post(
    "/ontology/proposals/{proposal_id}/approve",
    dependencies=[Depends(require_scope("write"))],
    summary="Approve an ontology proposal without changing the active schema",
)
async def approve_ontology_proposal(
    proposal_id: str, tenant: str = Depends(get_tenant), reviewed_by: str = "human",
):
    return await OntologyProposalService(get_neo4j()).decide(
        proposal_id, approve=True, reviewed_by=reviewed_by, tenant=tenant,
    )


@router.post(
    "/ontology/proposals/{proposal_id}/reject",
    dependencies=[Depends(require_scope("write"))],
    summary="Reject an ontology proposal",
)
async def reject_ontology_proposal(
    proposal_id: str, tenant: str = Depends(get_tenant), reviewed_by: str = "human",
):
    return await OntologyProposalService(get_neo4j()).decide(
        proposal_id, approve=False, reviewed_by=reviewed_by, tenant=tenant,
    )
