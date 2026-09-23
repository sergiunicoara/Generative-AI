"""Confidence lifecycle transition endpoint."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth.dependencies import get_current_user, get_tenant, require_scope
from graphrag.graph.neo4j_client import get_neo4j

router = APIRouter()


class ConfidenceTransitionRequest(BaseModel):
    src_name: str
    src_type: str
    relation: str
    tgt_name: str
    tgt_type: str
    target_state: str
    reason: str = ""


@router.post("/confidence/transition", dependencies=[Depends(require_scope("write"))])
async def transition_confidence(
    request: ConfidenceTransitionRequest,
    tenant: str = Depends(get_tenant),
    user: dict = Depends(get_current_user),
):
    from graphrag.graph.confidence_lifecycle import ConfidenceLifecycleService
    return await ConfidenceLifecycleService(get_neo4j()).transition_relation(
        **request.model_dump(),
        tenant=tenant,
        changed_by=str(user.get("sub") or "unknown"),
    )
