"""Contextual knowledge evaluation over HTTP.

A thin adapter over the MCP capability ``kg.knowledge.evaluate@1.0.0``: the
route builds the caller identity from the verified token and calls the same
registry path an MCP client uses, so scopes, tenant binding, argument
validation, size limits and audit are identical on both surfaces.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from api.auth.dependencies import get_current_user, get_tenant, require_scope
from mcp_server.capabilities import build_registry
from mcp_server.capabilities.knowledge_evaluate import CAPABILITY_ID, CAPABILITY_VERSION
from mcp_server.identity import CallerIdentity
from mcp_server.registry import DeniedCapabilityCall

router = APIRouter(prefix="/knowledge", tags=["Contextual Knowledge"])

_DENIAL_STATUS = {
    "missing_scope": 403, "unauthenticated": 401, "tenant_required": 403, "tenant_mismatch": 403,
    "invalid_arg": 422, "timeout": 504, "result_too_large": 413, "not_found": 404,
}
_ERROR_STATUS = {"invalid_request": 422, "no_published_release": 404}


class EvaluationContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer: list[str] | str | None = None
    brand: list[str] | str | None = None
    market: list[str] | str | None = None
    product: list[str] | str | None = None
    product_family: list[str] | str | None = None
    workflow: list[str] | str | None = None
    as_of: datetime | None = None
    known_as: datetime | None = None


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    context: EvaluationContext
    facts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    expect_applicable_knowledge: bool = False
    semantic_candidates: list[str] | None = Field(default=None, max_length=200)


@router.post("/evaluate", dependencies=[Depends(require_scope("read"))])
async def evaluate(
    body: EvaluationRequest,
    user: dict = Depends(get_current_user),
    tenant: str = Depends(get_tenant),
):
    """Evaluate the domain's active knowledge release for a business context."""
    identity = CallerIdentity.from_claims(user, tenant=tenant)
    args: dict[str, Any] = {
        "domain": body.domain,
        "context": body.context.model_dump(mode="json", exclude_none=True),
        "facts": body.facts,
        "expect_applicable_knowledge": body.expect_applicable_knowledge,
    }
    if body.semantic_candidates is not None:
        args["semantic_candidates"] = body.semantic_candidates
    result = await build_registry().call(f"{CAPABILITY_ID}@{CAPABILITY_VERSION}", args, identity)
    if isinstance(result, DeniedCapabilityCall):
        raise HTTPException(
            status_code=_DENIAL_STATUS.get(result.reason, 403),
            detail={"outcome": "denied", "reason": result.reason, "detail": result.detail,
                    "operation_id": result.operation_id},
        )
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=_ERROR_STATUS.get(result["error"]["code"], 400), detail=result["error"])
    return result
