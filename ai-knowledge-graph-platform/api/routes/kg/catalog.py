"""Document-catalog read API.

Additive only: reads MetadataEnvelope fields merge_document already
flattens onto the Document node (graphrag/graph/neo4j_client.py), plus
IngestionRunManifest run history via the -[:INGESTS]-> edge
upsert_ingestion_manifest already creates. No new node types, no writes --
before this existed, nothing under api/routes/ let a caller list documents
or see which ingestion run produced one (api/routes/ingest.py is write-only,
api/routes/kg/sources.py covers KGSource+mappings, not documents).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.auth.dependencies import get_tenant, require_scope
from graphrag.graph.neo4j_client import get_neo4j

router = APIRouter()


@router.get(
    "/catalog/documents",
    dependencies=[Depends(require_scope("read"))],
    summary="List documents with metadata-envelope filters",
)
async def list_catalog_documents(
    tenant: str = Depends(get_tenant),
    collection: str | None = None,
    classification: str | None = None,
    source_system: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    return await get_neo4j().get_documents_catalog(
        tenant=tenant, collection=collection, classification=classification,
        source_system=source_system, limit=limit, offset=offset,
    )


@router.get(
    "/catalog/documents/{doc_id}",
    dependencies=[Depends(require_scope("read"))],
    summary="Document metadata envelope, ACL, and ingestion-run history",
)
async def get_catalog_document(doc_id: str, tenant: str = Depends(get_tenant)):
    detail = await get_neo4j().get_document_catalog_detail(doc_id, tenant=tenant)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"document {doc_id!r} not found")
    return detail
