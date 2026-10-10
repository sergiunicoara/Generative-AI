"""Explicit confidence states and guarded transitions for graph knowledge."""

from __future__ import annotations

from enum import Enum
from uuid import uuid4


class ConfidenceState(str, Enum):
    ASSERTED = "ASSERTED"
    INFERRED = "INFERRED"
    DISPUTED = "DISPUTED"
    RETRACTED = "RETRACTED"
    APPROVED = "APPROVED"


_TRANSITIONS: dict[ConfidenceState, frozenset[ConfidenceState]] = {
    ConfidenceState.ASSERTED: frozenset({ConfidenceState.APPROVED, ConfidenceState.DISPUTED, ConfidenceState.RETRACTED}),
    ConfidenceState.INFERRED: frozenset({ConfidenceState.ASSERTED, ConfidenceState.APPROVED, ConfidenceState.DISPUTED, ConfidenceState.RETRACTED}),
    ConfidenceState.DISPUTED: frozenset({ConfidenceState.ASSERTED, ConfidenceState.APPROVED, ConfidenceState.RETRACTED}),
    ConfidenceState.RETRACTED: frozenset(),
    ConfidenceState.APPROVED: frozenset({ConfidenceState.DISPUTED, ConfidenceState.RETRACTED}),
}


def validate_transition(current: str | ConfidenceState, target: str | ConfidenceState) -> None:
    """Raise ValueError unless the requested lifecycle transition is allowed."""
    try:
        source = current if isinstance(current, ConfidenceState) else ConfidenceState(str(current).upper())
        destination = target if isinstance(target, ConfidenceState) else ConfidenceState(str(target).upper())
    except ValueError as exc:
        raise ValueError("unknown confidence state") from exc
    if destination not in _TRANSITIONS[source]:
        raise ValueError(f"invalid confidence transition {source.value} -> {destination.value}")


class ConfidenceLifecycleService:
    """Persist guarded relation-state changes and an immutable audit event."""

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    async def transition_relation(
        self,
        src_name: str,
        src_type: str,
        relation: str,
        tgt_name: str,
        tgt_type: str,
        target_state: str | ConfidenceState,
        *,
        tenant: str = "default",
        changed_by: str = "system",
        reason: str = "",
    ) -> dict:
        rows = await self._neo4j.run(
            """
            MATCH (s:Entity {name: $src_name, type: $src_type, tenant: $tenant})
                  -[r:RELATES_TO {relation: $relation}]->
                  (t:Entity {name: $tgt_name, type: $tgt_type, tenant: $tenant})
            RETURN coalesce(r.confidence_state, 'ASSERTED') AS state
            """,
            src_name=src_name, src_type=src_type, relation=relation,
            tgt_name=tgt_name, tgt_type=tgt_type, tenant=tenant,
        )
        if not rows:
            raise ValueError("relation not found")
        current = rows[0].get("state", ConfidenceState.ASSERTED.value)
        target = target_state if isinstance(target_state, ConfidenceState) else ConfidenceState(str(target_state).upper())
        validate_transition(current, target)
        event_id = str(uuid4())
        await self._neo4j.run(
            """
            MATCH (s:Entity {name: $src_name, type: $src_type, tenant: $tenant})
                  -[r:RELATES_TO {relation: $relation}]->
                  (t:Entity {name: $tgt_name, type: $tgt_type, tenant: $tenant})
            SET r.confidence_state = $target,
                r.confidence_state_changed_at = datetime(),
                // A reviewer's decision is the only thing that verifies or
                // rejects a fact (docs/trust-metadata.md).
                r.verification_status = CASE $target
                    WHEN 'APPROVED' THEN 'VERIFIED'
                    WHEN 'RETRACTED' THEN 'REJECTED'
                    ELSE coalesce(r.verification_status, 'UNVERIFIED') END,
                r.verified_by = CASE WHEN $target IN ['APPROVED', 'RETRACTED']
                    THEN $changed_by ELSE r.verified_by END,
                r.verified_at = CASE WHEN $target IN ['APPROVED', 'RETRACTED']
                    THEN datetime() ELSE r.verified_at END
            CREATE (e:ConfidenceTransition {
                id: $event_id, tenant: $tenant, current_state: $current,
                target_state: $target, changed_by: $changed_by,
                reason: $reason, recorded_at: datetime(),
                src_name: $src_name, src_type: $src_type,
                relation: $relation, tgt_name: $tgt_name, tgt_type: $tgt_type
            })
            """,
            src_name=src_name, src_type=src_type, relation=relation,
            tgt_name=tgt_name, tgt_type=tgt_type, tenant=tenant,
            current=str(current).upper(), target=target.value, changed_by=changed_by,
            reason=reason, event_id=event_id,
        )
        return {"event_id": event_id, "from": str(current).upper(), "to": target.value}
