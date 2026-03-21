"""POST /ingest — publish document to the ingestion queue."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth.dependencies import get_current_user, require_scope
from graphrag.core.models import Document
from graphrag.messaging.publishers import publish_document

router = APIRouter()


class IngestRequest(BaseModel):
    filename: str
    text: str
    priority: str = "normal"
    metadata: dict = {}


class IngestResponse(BaseModel):
    job_id: str
    doc_id: str
    status: str = "queued"


@router.post("", response_model=IngestResponse, dependencies=[Depends(require_scope("write"))])
async def ingest_document(request: IngestRequest):
    doc = Document(
        filename=request.filename,
        source_path=request.filename,
        raw_text=request.text,
        metadata=request.metadata,
    )
    try:
        job_id = await publish_document(doc, priority=request.priority)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {exc}")

    return IngestResponse(job_id=job_id, doc_id=doc.id)
