"""POST /ingest — publish document to the ingestion queue."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from api.auth.dependencies import get_current_user, require_scope
from api.limiter import INGEST_LIMIT, limiter
from graphrag.core.models import Document
from graphrag.messaging.publishers import publish_document

router = APIRouter()


class IngestRequest(BaseModel):
    filename: str
    text: str
    priority: str = "normal"
    metadata: dict = {}
    tenant: str = "default"


class IngestResponse(BaseModel):
    job_id: str
    doc_id: str
    status: str = "queued"


@router.post("", response_model=IngestResponse, dependencies=[Depends(require_scope("write"))])
@limiter.limit(INGEST_LIMIT)
async def ingest_document(request: Request, body: IngestRequest):
    """Publish a document to the ingestion queue.

    Rate-limited to prevent LLM quota exhaustion and Neo4j write overload.
    Default: 20 requests/minute per client IP (override via GRAPHRAG_RATE_LIMIT_INGEST).
    """
    doc = Document(
        filename=body.filename,
        source_path=body.filename,
        raw_text=body.text,
        metadata=body.metadata,
        tenant=body.tenant,
    )
    try:
        job_id = await publish_document(doc, priority=body.priority)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {exc}")

    return IngestResponse(job_id=job_id, doc_id=doc.id)
