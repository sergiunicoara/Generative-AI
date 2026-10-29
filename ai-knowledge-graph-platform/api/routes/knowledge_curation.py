"""Knowledge pack review and publication over HTTP.

The reviewer, grantor and publisher identity is always the token ``sub``,
never a request field, and the tenant is always the token tenant.  Reads need
``read`` scope; review decisions need ``write``; importing the deployed pack
and publishing a release change what every caller is served, so they need
``admin``.  Every mutation names the working revision it was based on
(``expected_revision``) and gets 409 if the pack moved in between.

Publication runs the unchanged release gate with the deployed regression
cases; a failed gate is 422 with the failures and leaves the active release
as it was.  See ``graphrag/context_graph/knowledge_curation.py``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, ConfigDict, Field

from api.auth.dependencies import get_current_user, get_tenant, require_scope
from graphrag.context_graph.knowledge_curation import CurationError, KnowledgeCurationService
from graphrag.context_graph.knowledge_release import KnowledgePublicationError
from graphrag.context_graph.knowledge_service import get_knowledge_service

router = APIRouter(prefix="/knowledge/packs", tags=["Contextual Knowledge"])

_STATUS = {"not_found": 404, "invalid": 422, "conflict": 409}
Domain = Annotated[str, Path(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
EntryId = Annotated[str, Path(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:\-]{0,127}$")]


class RevisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=0)


class ReviewRequest(RevisionRequest):
    review_ref: str = Field(min_length=1, max_length=256)
    publish: bool = True


async def curation_service() -> KnowledgeCurationService:
    return KnowledgeCurationService((await get_knowledge_service()).store)


def _actor(user: dict) -> str:
    actor = str(user.get("sub") or "")
    if not actor:
        raise HTTPException(status_code=403, detail="an authenticated reviewer identity is required")
    return actor


def _fail(exc: CurationError) -> HTTPException:
    return HTTPException(status_code=_STATUS.get(exc.code, 400), detail={"code": exc.code, "message": str(exc)})


@router.get("/{domain}", dependencies=[Depends(require_scope("read"))])
async def working_pack(domain: Domain, tenant: str = Depends(get_tenant),
                       service: KnowledgeCurationService = Depends(curation_service)):
    """The working pack: every entry's current version and status, the review
    queue, and the active release."""
    try:
        return await service.summary(tenant, domain)
    except CurationError as exc:
        raise _fail(exc) from exc


@router.get("/{domain}/history", dependencies=[Depends(require_scope("read"))])
async def pack_history(domain: Domain, tenant: str = Depends(get_tenant),
                       service: KnowledgeCurationService = Depends(curation_service)):
    return {"domain": domain, "revisions": await service.history(tenant, domain)}


@router.post("/{domain}/import", dependencies=[Depends(require_scope("admin"))])
async def import_pack(body: RevisionRequest, domain: Domain, tenant: str = Depends(get_tenant),
                      user: dict = Depends(get_current_user),
                      service: KnowledgeCurationService = Depends(curation_service)):
    """Seed the working pack from the deployed pack file (a new revision)."""
    try:
        return await service.import_pack(tenant, domain, actor=_actor(user),
                                         expected_revision=body.expected_revision)
    except CurationError as exc:
        raise _fail(exc) from exc


@router.post("/{domain}/{kind}/{entry_id}/{action}", dependencies=[Depends(require_scope("write"))])
async def review(body: ReviewRequest, kind: Literal["items", "exceptions"],
                 action: Literal["approve", "withdraw"], domain: Domain, entry_id: EntryId,
                 tenant: str = Depends(get_tenant), user: dict = Depends(get_current_user),
                 service: KnowledgeCurationService = Depends(curation_service)):
    """Approve (``publish: false`` stops at approved) or withdraw one entry.
    The token subject becomes ``reviewed_by`` on items and ``granted_by`` on
    exceptions."""
    try:
        return await service.review(
            tenant, domain, kind=kind, entry_id=entry_id, action=action, actor=_actor(user),
            reference=body.review_ref, expected_revision=body.expected_revision, publish=body.publish)
    except CurationError as exc:
        raise _fail(exc) from exc


@router.post("/{domain}/publish", dependencies=[Depends(require_scope("admin"))])
async def publish(body: RevisionRequest, domain: Domain, tenant: str = Depends(get_tenant),
                  user: dict = Depends(get_current_user),
                  service: KnowledgeCurationService = Depends(curation_service)):
    """Gate the working revision and make it the active release."""
    try:
        record = await service.publish(tenant, domain, actor=_actor(user), expected_revision=body.expected_revision)
    except CurationError as exc:
        raise _fail(exc) from exc
    except KnowledgePublicationError as exc:
        raise HTTPException(status_code=422, detail={"code": "gate_failed", "failures": exc.failures}) from exc
    return {key: record.get(key) for key in ("release_id", "version", "content_hash", "published_at",
                                             "published_by", "pack_version", "checks", "item_keys",
                                             "exception_keys")}
