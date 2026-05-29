"""Confidence calibration endpoints (Brier score, calibration curve, apply correction)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth.dependencies import require_scope
from graphrag.graph.neo4j_client import get_neo4j

router = APIRouter()


class CalibrationSampleRequest(BaseModel):
    predicted_confidence: float
    actual_outcome: float
    relation: str = ""
    source_doc_id: str = ""
    prompt_version: str = ""
    tenant: str = "default"
    verified_by: str = "admin"


class CalibrationBatchRequest(BaseModel):
    samples: list[CalibrationSampleRequest]
    tenant: str = "default"


@router.post(
    "/calibration/sample",
    dependencies=[Depends(require_scope("write"))],
    summary="Record a calibration data point (predicted vs actual outcome)",
)
async def add_calibration_sample(request: CalibrationSampleRequest):
    from graphrag.graph.confidence_calibration import CalibrationService
    svc = CalibrationService(get_neo4j())
    sid = await svc.add_sample(
        predicted_confidence=request.predicted_confidence,
        actual_outcome=request.actual_outcome,
        relation=request.relation,
        source_doc_id=request.source_doc_id,
        prompt_version=request.prompt_version,
        tenant=request.tenant,
        verified_by=request.verified_by,
    )
    return {"sample_id": sid}


@router.post(
    "/calibration/batch",
    dependencies=[Depends(require_scope("write"))],
    summary="Bulk-record calibration samples from a golden set",
)
async def add_calibration_batch(request: CalibrationBatchRequest):
    from graphrag.graph.confidence_calibration import CalibrationService
    svc = CalibrationService(get_neo4j())
    samples = [s.model_dump() for s in request.samples]
    ids = await svc.add_batch(samples, tenant=request.tenant)
    return {"added": len(ids), "sample_ids": ids}


@router.get(
    "/calibration/summary",
    dependencies=[Depends(require_scope("read"))],
    summary="Compute Brier score, calibration curve, and verdict",
)
async def calibration_summary(tenant: str = "default"):
    from graphrag.graph.confidence_calibration import CalibrationService
    svc = CalibrationService(get_neo4j())
    return await svc.calibration_summary(tenant=tenant)


@router.get(
    "/calibration/apply",
    dependencies=[Depends(require_scope("read"))],
    summary="Apply empirical calibration correction to a raw confidence value",
)
async def apply_calibration(confidence: float, tenant: str = "default"):
    from graphrag.graph.confidence_calibration import CalibrationService
    svc = CalibrationService(get_neo4j())
    calibrated = await svc.apply_calibration(confidence, tenant=tenant)
    return {"raw_confidence": confidence, "calibrated_confidence": calibrated}


@router.post(
    "/calibration/snapshot",
    dependencies=[Depends(require_scope("write"))],
    summary="Persist a calibration snapshot for trend tracking",
)
async def calibration_snapshot(tenant: str = "default", label: str = ""):
    from graphrag.graph.confidence_calibration import CalibrationService
    svc = CalibrationService(get_neo4j())
    snap_id = await svc.persist_snapshot(tenant=tenant, label=label)
    return {"snap_id": snap_id}


@router.get(
    "/calibration/trend",
    dependencies=[Depends(require_scope("read"))],
    summary="Return recent calibration snapshots for trend analysis",
)
async def calibration_trend(tenant: str = "default", limit: int = 10):
    from graphrag.graph.confidence_calibration import CalibrationService
    svc = CalibrationService(get_neo4j())
    return await svc.get_trend(tenant=tenant, limit=limit)
