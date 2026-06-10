"""Assembles all KG sub-routers into a single APIRouter."""

from __future__ import annotations

from fastapi import APIRouter

from api.routes.kg import (
    calibration,
    community,
    compliance,
    embeddings,
    health,
    inference,
    knowledge,
)

router = APIRouter()

router.include_router(calibration.router)
router.include_router(community.router)
router.include_router(compliance.router)
router.include_router(embeddings.router)
router.include_router(health.router)
router.include_router(inference.router)
router.include_router(knowledge.router)
