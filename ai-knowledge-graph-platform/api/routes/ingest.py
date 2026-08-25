"""POST /ingest — publish document to the ingestion queue."""

from typing import Any, Literal
from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.auth.dependencies import get_tenant, require_scope
from api.quota import enforce_tenant_quota
from api.limiter import INGEST_LIMIT, rate_limit
from graphrag.core.models import Document
from graphrag.enterprise.models import (
    DocumentAccessPolicy,
    LineageAssertion,
    MetadataEnvelope,
    ObligationDraft,
)
from graphrag.messaging.publishers import publish_document

router = APIRouter()
log = structlog.get_logger(__name__)


class IngestRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    text: str = Field(min_length=1, max_length=8_000_000)
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict, max_length=100)
    metadata_envelope: MetadataEnvelope = Field(default_factory=MetadataEnvelope)
    access_policy: DocumentAccessPolicy = Field(default_factory=DocumentAccessPolicy)
    lineage_assertions: list[LineageAssertion] = Field(default_factory=list, max_length=100)
    obligation_drafts: list[ObligationDraft] = Field(default_factory=list, max_length=500)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    source_id: str | None = Field(default=None, max_length=256)


class IngestResponse(BaseModel):
    job_id: str
    doc_id: str
    status: str = "queued"


@router.post(
    "",
    response_model=IngestResponse,
    # Order matters: scope, then burst protection, then budget. The
    # quota check is the most expensive of the three (it may hit
    # Redis), so it runs only for requests already known to be
    # authorized and within their rate.
    dependencies=[
        Depends(require_scope("write")),
        Depends(rate_limit(INGEST_LIMIT)),
        Depends(enforce_tenant_quota),
    ],
)
async def ingest_document(request: Request, body: IngestRequest, tenant: str = Depends(get_tenant)):
    """Publish a document to the ingestion queue.

    Rate-limited to prevent LLM quota exhaustion and Neo4j write overload.
    Default: 20 requests/minute per client IP (override via GRAPHRAG_RATE_LIMIT_INGEST).
    """
    doc = Document(
        filename=body.filename,
        source_path=body.filename,
        raw_text=body.text,
        metadata=body.metadata,
        metadata_envelope=body.metadata_envelope,
        access_policy=body.access_policy,
        lineage_assertions=body.lineage_assertions,
        obligation_drafts=body.obligation_drafts,
        valid_from=body.valid_from or body.metadata_envelope.effective_from,
        valid_to=body.valid_to or body.metadata_envelope.effective_to,
        tenant=tenant,
        source_id=body.source_id,
    )
    try:
        job_id = await publish_document(doc, priority=body.priority)
    except Exception as exc:
        log.error(
            "ingest.queue_unavailable",
            doc_id=doc.id,
            tenant=tenant,
            correlation_id=getattr(request.state, "correlation_id", ""),
            exception_type=type(exc).__name__,
        )
        raise HTTPException(status_code=503, detail="Queue unavailable") from exc

    return IngestResponse(job_id=job_id, doc_id=doc.id)
