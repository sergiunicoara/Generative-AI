"""Assembles all KG sub-routers into a single APIRouter."""

from __future__ import annotations

from fastapi import APIRouter

from api.routes.kg import (
    calibration,
    confidence,
    community,
    compliance,
    embeddings,
    health,
    inference,
    knowledge,
    ontology_proposals,
    pagerank,
    review_queue,
    feedback,
    sources,
)

router = APIRouter()

router.include_router(calibration.router)
router.include_router(confidence.router)
router.include_router(community.router)
router.include_router(compliance.router)
router.include_router(embeddings.router)
router.include_router(health.router)
router.include_router(inference.router)
router.include_router(knowledge.router)
router.include_router(ontology_proposals.router)
router.include_router(pagerank.router)
router.include_router(review_queue.router)
router.include_router(feedback.router)
router.include_router(sources.router)
