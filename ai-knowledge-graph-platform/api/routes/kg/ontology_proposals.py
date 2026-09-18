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
    reason: str = "", model_version: str = "",
):
    return await OntologyProposalService(get_neo4j()).decide(
        proposal_id, action="approve", reviewed_by=reviewed_by, tenant=tenant,
        reason=reason, model_version=model_version,
    )


@router.post(
    "/ontology/proposals/{proposal_id}/reject",
    dependencies=[Depends(require_scope("write"))],
    summary="Reject an ontology proposal",
)
async def reject_ontology_proposal(
    proposal_id: str, tenant: str = Depends(get_tenant), reviewed_by: str = "human",
    reason: str = "", model_version: str = "",
):
    return await OntologyProposalService(get_neo4j()).decide(
        proposal_id, action="reject", reviewed_by=reviewed_by, tenant=tenant,
        reason=reason, model_version=model_version,
    )


@router.post(
    "/ontology/proposals/{proposal_id}/edit",
    dependencies=[Depends(require_scope("write"))],
    summary="Approve an ontology proposal with a human-corrected proposed value",
)
async def edit_ontology_proposal(
    proposal_id: str, edited_value: str, tenant: str = Depends(get_tenant),
    reviewed_by: str = "human", reason: str = "", model_version: str = "",
):
    return await OntologyProposalService(get_neo4j()).decide(
        proposal_id, action="edit", reviewed_by=reviewed_by, tenant=tenant,
        reason=reason, model_version=model_version, edited_value=edited_value,
    )


@router.post(
    "/ontology/proposals/{proposal_id}/merge",
    dependencies=[Depends(require_scope("write"))],
    summary="Resolve a proposal by merging it into an existing type or relation",
)
async def merge_ontology_proposal(
    proposal_id: str, merge_target: str, tenant: str = Depends(get_tenant),
    reviewed_by: str = "human", reason: str = "", model_version: str = "",
):
    return await OntologyProposalService(get_neo4j()).decide(
        proposal_id, action="merge", reviewed_by=reviewed_by, tenant=tenant,
        reason=reason, model_version=model_version, merge_target=merge_target,
    )


@router.post(
    "/ontology/proposals/{proposal_id}/defer",
    dependencies=[Depends(require_scope("write"))],
    summary="Defer an ontology proposal for later review",
)
async def defer_ontology_proposal(
    proposal_id: str, tenant: str = Depends(get_tenant), reviewed_by: str = "human",
    reason: str = "", model_version: str = "",
):
    return await OntologyProposalService(get_neo4j()).decide(
        proposal_id, action="defer", reviewed_by=reviewed_by, tenant=tenant,
        reason=reason, model_version=model_version,
    )


@router.post(
    "/ontology/proposals/{proposal_id}/quarantine",
    dependencies=[Depends(require_scope("write"))],
    summary="Quarantine an ontology proposal pending further investigation",
)
async def quarantine_ontology_proposal(
    proposal_id: str, tenant: str = Depends(get_tenant), reviewed_by: str = "human",
    reason: str = "", model_version: str = "",
):
    return await OntologyProposalService(get_neo4j()).decide(
        proposal_id, action="quarantine", reviewed_by=reviewed_by, tenant=tenant,
        reason=reason, model_version=model_version,
    )
