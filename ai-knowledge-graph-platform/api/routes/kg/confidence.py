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
    from graphrag.graph.corpus_revision import CorpusMutation
    from graphrag.graph.invalidation import EventKind, InvalidationEvent, RelationRef, emit

    neo4j = get_neo4j()
    # Weakening a fact (DISPUTED / RETRACTED) only affects what was built on it:
    # targeted invalidation. Strengthening it can change answers that never cited
    # it, so the tenant revision is bumped as well.
    additive = request.target_state.upper() in ("ASSERTED", "APPROVED")
    async with CorpusMutation(neo4j, tenant, "confidence_transition", advance_revision=additive) as mutation:
        result = await ConfidenceLifecycleService(neo4j).transition_relation(
            **request.model_dump(),
            tenant=tenant,
            changed_by=str(user.get("sub") or "unknown"),
        )
        result["invalidation"] = await emit(InvalidationEvent(
            tenant=tenant, kind=EventKind.RELATION_CHANGED, additive=additive,
            reason=f"confidence {result.get('from')} -> {result.get('to')}",
            actor=str(user.get("sub") or "unknown"),
            cause=str(result.get("event_id") or ""),
            relations=[RelationRef(src_name=request.src_name, src_type=request.src_type,
                                   relation=request.relation, tgt_name=request.tgt_name,
                                   tgt_type=request.tgt_type)],
        ), neo4j)
    result["corpus_revision"] = mutation.revision
    return result
