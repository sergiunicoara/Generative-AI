"""Dependency tracking and targeted invalidation of derived artifacts.

See docs/invalidation.md.
"""
from graphrag.graph.invalidation.models import (
    ALLOWED_TRANSITIONS,
    ArtifactKind,
    ArtifactState,
    Closure,
    EntityRef,
    EventKind,
    InvalidationEvent,
    InvalidTransition,
    RelationRef,
    check_transition,
)
from graphrag.graph.invalidation.service import InvalidationService, emit

__all__ = [
    "ALLOWED_TRANSITIONS", "ArtifactKind", "ArtifactState", "Closure", "EntityRef", "EventKind",
    "InvalidTransition", "InvalidationEvent", "InvalidationService", "RelationRef",
    "check_transition", "emit",
]
