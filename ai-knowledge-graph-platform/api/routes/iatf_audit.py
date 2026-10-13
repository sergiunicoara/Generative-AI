"""IATF 16949 clause evidence, auditor sign-off and export (docs/iatf-audit.md).

Read routes need ``read`` scope; sign-off needs ``write`` scope and an authenticated subject, which becomes the
reviewer (never a body field). The evidence digest is computed server-side from the evidence the API itself shows,
so a client cannot sign off on evidence it did not see. Only tenants configured under ``iatf_audit.tenants`` in
settings get these routes; any other tenant receives 404, so one tenant's corpus is never served to another.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from api.auth.dependencies import get_current_user, get_tenant, require_scope
from graphrag.core.config import get_settings
from graphrag.domains.automotive import audit
from graphrag.domains.automotive.signoff import DECISIONS, ChainConflict, SignoffStore
from graphrag.graph.neo4j_client import get_neo4j

router = APIRouter(prefix="/audit/iatf", tags=["IATF Audit"])
_ROOT = Path(__file__).resolve().parents[2]


class SignoffRequest(BaseModel):
    decision: Literal["accepted", "rejected", "needs_more_evidence"]
    note: str = Field(default="", max_length=2_000)
    as_of: datetime | None = None


def _corpus_for(tenant: str) -> list[audit.CorpusDocument]:
    cfg = (get_settings().iatf_audit.get("tenants") or {}).get(tenant)
    if not cfg:
        raise HTTPException(status_code=404, detail="no IATF evidence corpus is configured for this tenant")
    return audit.load_corpus(_ROOT / cfg["corpus_dir"], _ROOT / cfg["metadata_dir"])


def _view(tenant: str, as_of: datetime | None, clause_ids: list[str] | None = None) -> dict:
    return audit.audit_view(docs=_corpus_for(tenant), as_of=as_of, clause_ids=clause_ids)


@router.get("/clauses", dependencies=[Depends(require_scope("read"))])
async def clause_coverage(as_of: datetime | None = None, tenant: str = Depends(get_tenant)):
    """Coverage of every catalogued clause, with evidence passages, superseded revisions and conflicts."""
    return _view(tenant, as_of)


@router.get("/clauses/{clause_id}", dependencies=[Depends(require_scope("read"))])
async def clause_detail(clause_id: str, as_of: datetime | None = None, tenant: str = Depends(get_tenant)):
    view = _view(tenant, as_of, [clause_id])
    if not view["clauses"]:
        raise HTTPException(status_code=404, detail="unknown clause")
    store = SignoffStore(get_neo4j())
    clause = view["clauses"][0]
    clause["signoffs"] = await store.entries(tenant, clause_id)
    return clause


@router.post("/clauses/{clause_id}/signoff", dependencies=[Depends(require_scope("write"))])
async def sign_off(clause_id: str, body: SignoffRequest, tenant: str = Depends(get_tenant),
                   user: dict = Depends(get_current_user)):
    reviewer = str(user.get("sub") or "")
    if not reviewer:
        raise HTTPException(status_code=403, detail="an authenticated reviewer identity is required")
    view = _view(tenant, body.as_of, [clause_id])
    if not view["clauses"]:
        raise HTTPException(status_code=404, detail="unknown clause")
    clause = view["clauses"][0]
    try:
        return await SignoffStore(get_neo4j()).append(
            tenant=tenant, clause=clause_id, decision=body.decision, reviewer=reviewer, note=body.note,
            evidence_digest=clause["evidence_digest"], evidence_as_of=view["as_of"])
    except ChainConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/signoffs", dependencies=[Depends(require_scope("read"))])
async def signoffs(tenant: str = Depends(get_tenant)):
    _corpus_for(tenant)
    store = SignoffStore(get_neo4j())
    return {"entries": await store.entries(tenant), "chain": await store.verify(tenant), "decisions": DECISIONS}


@router.get("/export", dependencies=[Depends(require_scope("read"))])
async def export(as_of: datetime | None = None, format: Literal["json", "csv"] = Query("json"),
                 tenant: str = Depends(get_tenant)):
    """Evidence as of a date plus the latest sign-off per clause and whether its evidence digest still matches."""
    view = _view(tenant, as_of)
    store = SignoffStore(get_neo4j())
    latest = await store.latest_by_clause(tenant)
    rows = audit.export_rows(view, latest)
    if format == "csv":
        return PlainTextResponse(audit.to_csv(rows), media_type="text/csv")
    return {"view": view, "signoffs": latest, "chain": await store.verify(tenant), "rows": rows}
