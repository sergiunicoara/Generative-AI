"""Contextual knowledge over HTTP: evaluation, and pack curation.

``POST /knowledge/evaluate`` is a thin adapter over the MCP capability
``kg.knowledge.evaluate@1.0.0``: the route builds the caller identity from the
verified token and calls the same registry path an MCP client uses, so scopes,
tenant binding, argument validation, size limits and audit are identical on
both surfaces.

The pack routes (review queue, publish, rollback, history) administer the
version-controlled packs under ``config/knowledge_packs``.  Reviewer and
publisher identity is the token ``sub``, never a request field; the tenant
comes from the token and must match the pack's tenant; a pack is addressed by
a validated domain name, never by a path.  Publishing runs the same gate as
``publish_pack`` (contract, version, referential and mandatory regression
checks).  Release records can hold restricted knowledge, so these routes
return release metadata and counts only, never items or exceptions.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as PathParam
from pydantic import BaseModel, ConfigDict, Field, ValidationError
import structlog
import yaml

from api.auth.dependencies import get_current_user, get_tenant, require_scope
from graphrag.context_graph.knowledge_evaluation import can_read
from graphrag.context_graph.knowledge_release import (
    KnowledgePack, KnowledgePublicationError, load_cases, load_pack, review_queue,
)
from graphrag.context_graph.knowledge_service import get_knowledge_service
from graphrag.domains.energy.governance_store import GovernanceStoreError
from mcp_server.capabilities import build_registry
from mcp_server.capabilities.knowledge_evaluate import CAPABILITY_ID, CAPABILITY_VERSION, access_from_identity
from mcp_server.identity import CallerIdentity
from mcp_server.registry import DeniedCapabilityCall

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/knowledge", tags=["Contextual Knowledge"])

_ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = _ROOT / "config" / "knowledge_packs"
CASES_DIR = _ROOT / "evals" / "knowledge_eval"
_DOMAIN = r"^[a-z][a-z0-9_]{0,63}$"

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


# ---------------------------------------------------------------------------
# pack curation


class RollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    release_id: str = Field(pattern=r"^[0-9a-f]{32}$")


def _subject(user: dict) -> str:
    subject = str(user.get("sub") or "")
    if not subject:
        raise HTTPException(status_code=403, detail="Token carries no subject to record as reviewer/publisher")
    return subject


def _pack_not_found() -> HTTPException:
    return HTTPException(status_code=404, detail={"code": "pack_not_found", "message": "no such knowledge pack"})


def _load_pack_for(domain: str, tenant: str) -> KnowledgePack:
    """The domain's pack, if it exists for the token's tenant (else 404, so a
    caller cannot probe other tenants' packs)."""
    path = PACK_DIR / f"{domain}.yml"
    if not path.is_file():
        raise _pack_not_found()
    try:
        pack = load_pack(path)
    except (ValidationError, yaml.YAMLError, OSError) as exc:
        raise HTTPException(status_code=422, detail={"code": "invalid_pack", "message": str(exc)[:500]}) from exc
    if pack.tenant != tenant or pack.domain != domain:
        raise _pack_not_found()
    return pack


def _release_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Release metadata and counts; never the (possibly restricted) content."""
    return {
        "release_id": record["release_id"], "version": record.get("version"), "tenant": record["tenant"],
        "domain": record["domain"], "pack_version": record.get("pack_version"),
        "content_hash": record["content_hash"], "published_at": record.get("published_at"),
        "published_by": record.get("published_by"), "rolled_back_from": record.get("rolled_back_from"),
        "checks": record.get("checks", {}), "disclaimer": record.get("disclaimer", ""),
        "item_count": len(record.get("items", [])), "exception_count": len(record.get("exceptions", [])),
    }


Domain = Annotated[str, PathParam(pattern=_DOMAIN)]


@router.get("/packs/{domain}/review-queue", dependencies=[Depends(require_scope("write"))])
async def pack_review_queue(
    domain: Domain, user: dict = Depends(get_current_user), tenant: str = Depends(get_tenant),
):
    """Draft and in-review entries of the domain's pack and what each still needs.

    Entries the caller may not read are left out, as in evaluation.
    """
    pack = _load_pack_for(domain, tenant)
    access = access_from_identity(CallerIdentity.from_claims(user, tenant=tenant))
    readable = pack.model_copy(update={"items": [i for i in pack.items if can_read(i.access, access)]})
    return {"domain": domain, "queue": review_queue(readable)}


@router.post("/packs/{domain}/publish", dependencies=[Depends(require_scope("admin"))])
async def publish_pack_route(
    domain: Domain, user: dict = Depends(get_current_user), tenant: str = Depends(get_tenant),
):
    """Gate and activate the domain's pack as a new release, recorded under the token ``sub``."""
    subject = _subject(user)
    pack = _load_pack_for(domain, tenant)
    cases_path = CASES_DIR / f"{domain}_cases.json"
    cases = load_cases(cases_path, domain=domain) if cases_path.is_file() else []
    service = await get_knowledge_service()
    try:
        record = await service.publish(pack, published_by=subject, cases=cases)
    except KnowledgePublicationError as exc:
        log.info("knowledge.publish_rejected", tenant=tenant, domain=domain, subject=subject,
                 failures=len(exc.failures))
        raise HTTPException(status_code=422, detail={
            "code": "release_rejected", "message": "release rejected; the active release is unchanged",
            "failures": exc.failures[:50]}) from exc
    log.info("knowledge.published", tenant=tenant, domain=domain, subject=subject,
             release_id=record["release_id"], version=record.get("version"))
    return _release_summary(record)


@router.get("/releases/{domain}", dependencies=[Depends(require_scope("read"))])
async def release_history(domain: Domain, tenant: str = Depends(get_tenant)):
    """Release history (metadata only) for the token's tenant."""
    service = await get_knowledge_service()
    history = await service.release_history(tenant, domain)
    return {"releases": [_release_summary(r) for r in history],
            "active_release_id": await service.active_release_id(tenant, domain)}


@router.post("/releases/{domain}/rollback", dependencies=[Depends(require_scope("admin"))])
async def rollback_release(
    domain: Domain, body: RollbackRequest, user: dict = Depends(get_current_user),
    tenant: str = Depends(get_tenant),
):
    """Re-activate an earlier release's content as a new appended release."""
    subject = _subject(user)
    service = await get_knowledge_service()
    try:
        record = await service.rollback(tenant, domain, body.release_id, rolled_back_by=subject)
    except GovernanceStoreError as exc:
        raise HTTPException(status_code=404, detail={
            "code": "release_not_found", "message": "no such release for this tenant and domain"}) from exc
    log.info("knowledge.rolled_back", tenant=tenant, domain=domain, subject=subject,
             release_id=record["release_id"], restored_from=body.release_id)
    return _release_summary(record)
