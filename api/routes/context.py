"""§11 required API: POST /api/v1/context/build, GET /api/v1/claims/{id}/evidence."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_workspace_id
from src.context_graph.builder import ContextGraphBuilder, ContextGraphScope
from src.graph.execution import GraphExecutor
from src.graph.repositories.claim_repository import ClaimRepository
from src.graph.repositories.conversation_repository import ConversationRepository

router = APIRouter(tags=["context"])


class ContextBuildRequest(BaseModel):
    seller_id: str | None = None
    account_id: str | None = None
    opportunity_id: str | None = None
    conversation_id: str | None = None
    subject_id: str | None = None
    max_nodes: int | None = None
    max_tokens: int | None = None


@router.post("/api/v1/context/build")
async def build_context(body: ContextBuildRequest, workspace_id: str = Depends(get_workspace_id)) -> dict:
    # §12: workspace_id always from authenticated context (get_workspace_id),
    # never the request body — ContextBuildRequest deliberately has no
    # workspace_id field at all.
    executor = GraphExecutor()
    builder = ContextGraphBuilder(ClaimRepository(executor))
    scope = ContextGraphScope(
        workspace_id=workspace_id, conversation_id=body.conversation_id, subject_id=body.subject_id,
    )
    kwargs = {}
    if body.max_nodes is not None:
        kwargs["max_nodes"] = body.max_nodes
    if body.max_tokens is not None:
        kwargs["max_tokens"] = body.max_tokens
    result = await builder.build(scope, **kwargs)
    return result.model_dump(mode="json")


@router.get("/api/v1/claims/{claim_id}/evidence")
async def get_claim_evidence(claim_id: str, workspace_id: str = Depends(get_workspace_id)) -> dict:
    executor = GraphExecutor()
    claim_repo = ClaimRepository(executor)
    claim = await claim_repo.get_claim(workspace_id, claim_id)
    if claim is None:
        raise HTTPException(status_code=404, detail="claim not found")

    excerpt = None
    if claim.source_segment_id:
        segment = await ConversationRepository(executor).get_segment(workspace_id, claim.source_segment_id)
        if segment:
            excerpt = segment.text[claim.evidence_char_start:claim.evidence_char_end]

    return {
        "claim_id": claim.claim_id,
        "predicate": claim.predicate,
        "object_value": claim.object_value,
        "object_id": claim.object_id,
        "polarity": claim.polarity.value,
        "source_segment_id": claim.source_segment_id,
        "evidence_char_start": claim.evidence_char_start,
        "evidence_char_end": claim.evidence_char_end,
        "excerpt": excerpt,
        "speaker_role": claim.speaker_role.value,
        "confidence": claim.confidence,
        "adjudication_status": claim.adjudication_status.value,
    }
