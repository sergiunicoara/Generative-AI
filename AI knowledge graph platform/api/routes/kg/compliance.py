"""GDPR erasure and PII guard endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth.dependencies import require_scope
from graphrag.graph.neo4j_client import get_neo4j

router = APIRouter()


# ── GDPR / Right-to-be-Forgotten ─────────────────────────────────────────────

class ForgetEntityRequest(BaseModel):
    entity_name: str
    entity_type: str
    tenant: str = "default"
    requested_by: str = "dpo"
    request_id: str = ""


class ForgetDocumentRequest(BaseModel):
    doc_id: str
    tenant: str = "default"
    requested_by: str = "dpo"
    request_id: str = ""


@router.post(
    "/gdpr/forget-entity",
    dependencies=[Depends(require_scope("write"))],
    summary="Permanently erase all data for a named entity (GDPR right-to-be-forgotten)",
)
async def gdpr_forget_entity(request: ForgetEntityRequest):
    from graphrag.graph.gdpr import GDPRService
    svc = GDPRService(get_neo4j())
    return await svc.forget_entity(
        entity_name=request.entity_name,
        entity_type=request.entity_type,
        tenant=request.tenant,
        requested_by=request.requested_by,
        request_id=request.request_id,
    )


@router.post(
    "/gdpr/forget-document",
    dependencies=[Depends(require_scope("write"))],
    summary="Erase all data exclusively sourced from a document (GDPR erasure)",
)
async def gdpr_forget_document(request: ForgetDocumentRequest):
    from graphrag.graph.gdpr import GDPRService
    svc = GDPRService(get_neo4j())
    return await svc.forget_document(
        doc_id=request.doc_id,
        tenant=request.tenant,
        requested_by=request.requested_by,
        request_id=request.request_id,
    )


@router.get(
    "/gdpr/audit-log",
    dependencies=[Depends(require_scope("read"))],
    summary="Return the GDPR deletion audit log for a tenant",
)
async def gdpr_audit_log(tenant: str = "default", limit: int = 100):
    from graphrag.graph.gdpr import GDPRService
    svc = GDPRService(get_neo4j())
    return await svc.deletion_audit_log(tenant=tenant, limit=limit)


# ── PII Guard ─────────────────────────────────────────────────────────────────

class PIIScanRequest(BaseModel):
    text: str
    min_confidence: float = 0.80


class PIITagRequest(BaseModel):
    entity_name: str
    entity_type: str
    tenant: str = "default"
    reason: str = ""


@router.post(
    "/pii/scan",
    dependencies=[Depends(require_scope("read"))],
    summary="Scan text for PII patterns (SSN, email, phone, credit card, etc.)",
)
async def pii_scan(request: PIIScanRequest):
    from graphrag.graph.pii_guard import PIIGuard
    guard = PIIGuard(get_neo4j(), min_confidence=request.min_confidence)
    findings = guard.scan_text(request.text)
    return {
        "finding_count": len(findings),
        "findings": [
            {
                "pii_class":  f.pii_class,
                "confidence": f.confidence,
                "offset":     f.start,
                "length":     f.end - f.start,
            }
            for f in findings
        ],
    }


@router.post(
    "/pii/redact",
    dependencies=[Depends(require_scope("read"))],
    summary="Return text with PII replaced by [CLASS_REDACTED] placeholders",
)
async def pii_redact(request: PIIScanRequest):
    from graphrag.graph.pii_guard import PIIGuard
    guard = PIIGuard(get_neo4j(), min_confidence=request.min_confidence)
    return {"redacted_text": guard.redact(request.text)}


@router.post(
    "/pii/tag-entity",
    dependencies=[Depends(require_scope("write"))],
    summary="Mark an entity as PII-sensitive in Neo4j",
)
async def pii_tag_entity(request: PIITagRequest):
    from graphrag.graph.pii_guard import PIIGuard
    guard = PIIGuard(get_neo4j())
    await guard.tag_entity_pii(
        entity_name=request.entity_name,
        entity_type=request.entity_type,
        tenant=request.tenant,
        reason=request.reason,
    )
    return {"status": "tagged"}


@router.post(
    "/pii/auto-tag-persons",
    dependencies=[Depends(require_scope("write"))],
    summary="Tag all PERSON entities in a tenant as PII-sensitive",
)
async def pii_auto_tag_persons(tenant: str = "default"):
    from graphrag.graph.pii_guard import PIIGuard
    guard = PIIGuard(get_neo4j())
    count = await guard.auto_tag_persons(tenant=tenant)
    return {"tagged": count, "tenant": tenant}


@router.get(
    "/pii/scan-document/{doc_id}",
    dependencies=[Depends(require_scope("read"))],
    summary="Scan all chunks of a document for PII (diagnostic only — no mutations)",
)
async def pii_scan_document(doc_id: str, tenant: str = "default"):
    from graphrag.graph.pii_guard import PIIGuard
    guard = PIIGuard(get_neo4j())
    return await guard.scan_document(doc_id=doc_id, tenant=tenant)


@router.get(
    "/pii/inventory",
    dependencies=[Depends(require_scope("read"))],
    summary="List all entities tagged as PII-sensitive",
)
async def pii_inventory(tenant: str = "default", limit: int = 100):
    from graphrag.graph.pii_guard import PIIGuard
    guard = PIIGuard(get_neo4j())
    return await guard.list_pii_entities(tenant=tenant, limit=limit)
