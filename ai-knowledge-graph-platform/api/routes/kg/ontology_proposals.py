"""Human-review endpoints for proposed ontology changes discovered in ingestion."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
import structlog

from api.auth.dependencies import get_current_user, get_tenant, require_scope
from graphrag.curation.exchange import (
    MAX_DECISION_ROWS, MAX_UPLOAD_BYTES, Decision, ExchangeError, export_csv, export_xlsx, parse_decisions,
)
from graphrag.curation.jira import JiraClient, JiraConfig, JiraError, proposal_to_jira_issue
from graphrag.graph.neo4j_client import get_neo4j
from graphrag.graph.ontology_proposals import OntologyProposalService

log = structlog.get_logger(__name__)
router = APIRouter()

CSV_MIME = "text/csv"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_UPLOAD_FORMATS = {CSV_MIME: "csv", XLSX_MIME: "xlsx"}
_JIRA_ERROR_STATUS = {"jira_timeout": 504}


def _reviewer(user: dict) -> str:
    """Reviewer identity is the verified token ``sub`` -- never a request field."""
    subject = str(user.get("sub") or "")
    if not subject:
        raise HTTPException(status_code=403, detail="Token carries no subject to record as reviewer")
    return subject


def get_jira_client() -> JiraClient | None:
    """The configured Jira client, or ``None`` when Jira integration is not enabled."""
    config = JiraConfig.from_env()
    return JiraClient(config) if config else None


@router.get(
    "/ontology/proposals",
    dependencies=[Depends(require_scope("read"))],
    summary="List tenant-scoped ontology change proposals",
)
async def list_ontology_proposals(
    tenant: str = Depends(get_tenant), status: str = "pending", limit: int = 100,
):
    return {"items": await OntologyProposalService(get_neo4j()).list(tenant, status=status, limit=limit)}


@router.get(
    "/ontology/proposals/export",
    dependencies=[Depends(require_scope("read"))],
    summary="Export decided ontology proposals as golden-set/training records",
)
async def export_ontology_proposals(tenant: str = Depends(get_tenant), limit: int = 500):
    return {"items": await OntologyProposalService(get_neo4j()).export_golden_set(tenant, limit=limit)}


@router.get(
    "/ontology/proposals/status-report",
    dependencies=[Depends(require_scope("read"))],
    summary="Ontology proposal counts by status and kind, for partner-team reporting",
)
async def ontology_proposal_status_report(tenant: str = Depends(get_tenant)):
    return await OntologyProposalService(get_neo4j()).status_report(tenant)


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


# -- spreadsheet exchange and Jira -------------------------------------------
# The dotted / multi-segment paths below cannot collide with
# ``/ontology/proposals/export`` or ``/ontology/proposals/{id}/<action>``.

def _attachment(data: bytes, media_type: str, filename: str) -> Response:
    return Response(content=data, media_type=media_type,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get(
    "/ontology/proposals/export.csv",
    dependencies=[Depends(require_scope("read"))],
    summary="Download ontology proposals as a CSV decisions sheet (formula-safe)",
)
async def export_proposals_csv(
    tenant: str = Depends(get_tenant), status: str = "pending",
    limit: int = Query(500, ge=1, le=MAX_DECISION_ROWS),
):
    proposals = await OntologyProposalService(get_neo4j()).list(tenant, status=status, limit=limit)
    return _attachment(export_csv(proposals), f"{CSV_MIME}; charset=utf-8", "ontology-proposals.csv")


@router.get(
    "/ontology/proposals/export.xlsx",
    dependencies=[Depends(require_scope("read"))],
    summary="Download ontology proposals as an XLSX decisions sheet (formula-safe)",
)
async def export_proposals_xlsx(
    tenant: str = Depends(get_tenant), status: str = "pending",
    limit: int = Query(500, ge=1, le=MAX_DECISION_ROWS),
):
    proposals = await OntologyProposalService(get_neo4j()).list(tenant, status=status, limit=limit)
    return _attachment(export_xlsx(proposals), XLSX_MIME, "ontology-proposals.xlsx")


def _too_large() -> HTTPException:
    return HTTPException(status_code=413, detail={"code": "upload_too_large", "message": "upload too large"})


def _upload_format(request: Request) -> str:
    media_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if media_type not in _UPLOAD_FORMATS:
        raise HTTPException(status_code=415, detail={
            "code": "unsupported_media_type", "message": f"send the sheet as {CSV_MIME} or {XLSX_MIME}"})
    return _UPLOAD_FORMATS[media_type]


async def _read_upload(request: Request) -> bytes:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise _too_large()
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise _too_large()
        chunks.append(chunk)
    return b"".join(chunks)


def _item(decision: Decision, outcome: str, error: str | None = None) -> dict:
    item = {"row": decision.row, "proposal_id": decision.proposal_id, "action": decision.action, "outcome": outcome}
    return {**item, "error": error} if error else item


async def _preview(service: OntologyProposalService, tenant: str, decisions: tuple[Decision, ...]) -> list[dict]:
    found = {p["id"]: p for p in await service.get_many(tenant, [d.proposal_id for d in decisions])}
    items = []
    for decision in decisions:
        proposal = found.get(decision.proposal_id)
        if proposal is None:
            items.append(_item(decision, "not_found"))
        elif proposal.get("status") != "pending":
            items.append(_item(decision, "not_pending", f"current status is '{proposal.get('status')}'"))
        else:
            items.append(_item(decision, "will_apply"))
    return items


async def _apply(
    service: OntologyProposalService, tenant: str, reviewer: str, decisions: tuple[Decision, ...],
) -> list[dict]:
    items = []
    for decision in decisions:
        result = await service.decide(
            decision.proposal_id, action=decision.action, reviewed_by=reviewer, tenant=tenant,
            reason=decision.reason, edited_value=decision.edited_value, merge_target=decision.merge_target,
        )
        error = result.get("error")
        items.append(_item(decision, "error", error) if error else _item(decision, "applied"))
    return items


@router.post(
    "/ontology/proposals/decisions/import",
    dependencies=[Depends(require_scope("write"))],
    summary="Validate (dry run, default) or apply a CSV/XLSX decisions sheet",
)
async def import_proposal_decisions(
    request: Request, dry_run: bool = True,
    tenant: str = Depends(get_tenant), user: dict = Depends(get_current_user),
):
    """Upload the sheet as the raw request body (``text/csv`` or the XLSX mime type).

    With ``dry_run=false`` every row must be valid, otherwise nothing is applied.
    The reviewer recorded on each decision is the token ``sub``.
    """
    reviewer = None if dry_run else _reviewer(user)
    fmt = _upload_format(request)
    data = await _read_upload(request)
    try:
        parsed = parse_decisions(data, fmt)
    except ExchangeError as exc:
        raise HTTPException(status_code=422, detail={"code": exc.code, "message": exc.message}) from exc
    errors = [{"row": e.row, "message": e.message} for e in parsed.errors]
    if errors and not dry_run:
        raise HTTPException(status_code=422, detail={
            "code": "invalid_rows", "message": "fix the invalid rows; nothing was applied", "errors": errors})

    service = OntologyProposalService(get_neo4j())
    if dry_run:
        items = await _preview(service, tenant, parsed.decisions)
    else:
        items = await _apply(service, tenant, reviewer, parsed.decisions)
        log.info("ontology_proposal.sheet_imported", tenant=tenant, subject=reviewer,
                 applied=sum(i["outcome"] == "applied" for i in items),
                 failed=sum(i["outcome"] == "error" for i in items))
    return {"dry_run": dry_run, "items": items, "errors": errors, "skipped": parsed.skipped}


@router.post(
    "/ontology/proposals/{proposal_id}/jira",
    dependencies=[Depends(require_scope("write"))],
    summary="Create one Jira ticket for an ontology proposal (when Jira is configured)",
)
async def create_proposal_jira_issue(
    proposal_id: str, tenant: str = Depends(get_tenant), user: dict = Depends(get_current_user),
    jira: JiraClient | None = Depends(get_jira_client),
):
    if jira is None:
        raise HTTPException(status_code=503, detail={
            "code": "jira_not_configured", "message": "Jira integration is not configured"})
    found = await OntologyProposalService(get_neo4j()).get_many(tenant, [proposal_id])
    if not found:
        raise HTTPException(status_code=404, detail={"code": "proposal_not_found", "message": "no such proposal"})
    payload = proposal_to_jira_issue(found[0], project_key=jira.project_key)
    try:
        key = await jira.create_issue(payload)
    except JiraError as exc:
        raise HTTPException(status_code=_JIRA_ERROR_STATUS.get(exc.code, 502),
                            detail={"code": exc.code, "message": exc.message}) from exc
    log.info("ontology_proposal.jira_issue_created", tenant=tenant, proposal_id=proposal_id,
             subject=user.get("sub"), issue_key=key)
    return {"proposal_id": proposal_id, "issue_key": key}
