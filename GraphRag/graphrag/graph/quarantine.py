"""Quarantine service — isolates suspicious subgraphs from retrieval.

Problems solved
---------------
1. Cascading extraction errors — a hallucinated entity or relation early in
   ingestion propagates through alias resolution, community detection, and GNN
   scoring. Validation detects anomalies but does not stop their spread.

2. No human-review lane — flagged issues are logged as warnings but remain
   fully active in retrieval. There is no mechanism to park suspicious data
   until a human reviews it.

Architecture
------------
- Quarantined entities get `quarantined = true` and are excluded from all
  retrieval queries via a WHERE filter.
- QuarantineLog nodes record who quarantined what and why.
- SubGraph quarantine propagates from a seed entity to all entities reachable
  only through it (isolating a suspicious extraction cluster).
- release() requires a reviewer identity — creates an explicit approval record.
"""

from __future__ import annotations

from uuid import uuid4

import structlog

log = structlog.get_logger(__name__)

MAX_QUARANTINE_DEPTH = 5   # max hops to propagate quarantine


class QuarantineService:
    """
    Quarantine suspicious entities and subgraphs.

    Usage::

        svc = QuarantineService(neo4j_client)
        await svc.quarantine_entity("FakeEntity", "CONCEPT",
                                    reason="degree_anomaly", flagged_by="validator")
        await svc.release("FakeEntity", "CONCEPT", released_by="admin@example.com")
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    # ── Quarantine ─────────────────────────────────────────────────────────────

    async def quarantine_entity(
        self,
        entity_name: str,
        entity_type: str,
        reason: str,
        flagged_by: str = "system",
    ) -> None:
        """Flag a single entity as quarantined — excluded from retrieval."""
        log_id = str(uuid4())
        await self._neo4j.run(
            """
            MATCH (e:Entity {name: $name, type: $type})
            SET e.quarantined         = true,
                e.quarantine_reason   = $reason,
                e.quarantined_at      = datetime(),
                e.quarantined_by      = $flagged_by
            CREATE (q:QuarantineLog {
                id:          $log_id,
                entity_name: $name,
                entity_type: $type,
                reason:      $reason,
                flagged_by:  $flagged_by,
                action:      'quarantine',
                logged_at:   datetime()
            })
            """,
            name=entity_name,
            type=entity_type,
            reason=reason,
            flagged_by=flagged_by,
            log_id=log_id,
        )
        log.warning(
            "quarantine.entity_flagged",
            entity=entity_name,
            reason=reason,
            flagged_by=flagged_by,
        )

    async def quarantine_subgraph_from(
        self,
        seed_entity_name: str,
        seed_entity_type: str,
        reason: str,
        flagged_by: str = "system",
        depth: int = 2,
    ) -> int:
        """
        Quarantine the seed entity and all entities reachable from it
        within `depth` hops that have no other incoming paths from
        non-quarantined nodes. Returns count of quarantined entities.
        """
        depth = min(depth, MAX_QUARANTINE_DEPTH)

        # First quarantine the seed itself
        await self.quarantine_entity(seed_entity_name, seed_entity_type, reason, flagged_by)

        # Find reachable entities that are ONLY reachable from this seed
        rows = await self._neo4j.run(
            f"""
            MATCH (seed:Entity {{name: $seed_name}})
            MATCH (seed)-[:RELATES_TO*1..{depth}]->(neighbor:Entity)
            WHERE NOT neighbor.quarantined = true
            // Only quarantine if neighbor has no path from non-quarantined nodes
            //  other than through the seed
            WITH neighbor
            WHERE NOT EXISTS {{
                MATCH (other:Entity)-[:RELATES_TO*1..{depth}]->(neighbor)
                WHERE NOT other.quarantined = true
                  AND other.name <> $seed_name
            }}
            SET neighbor.quarantined       = true,
                neighbor.quarantine_reason = $reason,
                neighbor.quarantined_at    = datetime(),
                neighbor.quarantined_by    = $flagged_by
            RETURN count(neighbor) AS quarantined
            """,
            seed_name=seed_entity_name,
            reason=f"subgraph_from:{seed_entity_name}",
            flagged_by=flagged_by,
        )
        count = (rows[0]["quarantined"] if rows else 0) + 1   # +1 for seed
        log.warning(
            "quarantine.subgraph_flagged",
            seed=seed_entity_name,
            depth=depth,
            total_quarantined=count,
        )
        return count

    # ── Release ────────────────────────────────────────────────────────────────

    async def release(
        self,
        entity_name: str,
        entity_type: str,
        released_by: str,
        note: str = "",
    ) -> None:
        """
        Release a quarantined entity back into the active graph.
        Creates an approval record in QuarantineLog.
        """
        log_id = str(uuid4())
        await self._neo4j.run(
            """
            MATCH (e:Entity {name: $name, type: $type})
            REMOVE e.quarantined
            SET e.quarantine_released_at = datetime(),
                e.quarantine_released_by = $released_by
            CREATE (q:QuarantineLog {
                id:          $log_id,
                entity_name: $name,
                entity_type: $type,
                reason:      $note,
                flagged_by:  $released_by,
                action:      'release',
                logged_at:   datetime()
            })
            """,
            name=entity_name,
            type=entity_type,
            released_by=released_by,
            note=note,
            log_id=log_id,
        )
        log.info(
            "quarantine.entity_released",
            entity=entity_name,
            released_by=released_by,
        )

    # ── Listing ────────────────────────────────────────────────────────────────

    async def list_quarantined(self, limit: int = 100) -> list[dict]:
        """Return all currently quarantined entities."""
        return await self._neo4j.run(
            """
            MATCH (e:Entity)
            WHERE e.quarantined = true
            RETURN e.name            AS entity_name,
                   e.type            AS entity_type,
                   e.quarantine_reason AS reason,
                   e.quarantined_at  AS quarantined_at,
                   e.quarantined_by  AS quarantined_by
            ORDER BY e.quarantined_at DESC
            LIMIT $limit
            """,
            limit=limit,
        )

    async def auto_quarantine_anomalies(
        self, doc_id: str, validation_report: dict
    ) -> int:
        """
        Called automatically by GraphWriter after ingestion validation.
        Quarantines entities flagged as degree anomalies (likely hallucinated hubs).
        Returns count of auto-quarantined entities.
        """
        count = 0
        for issue in validation_report.get("issues", []):
            if issue["type"] == "degree_anomaly":
                entity_name = issue.get("entity", "")
                if entity_name:
                    await self.quarantine_entity(
                        entity_name=entity_name,
                        entity_type="UNKNOWN",
                        reason=f"degree_anomaly:degree={issue.get('degree')}",
                        flagged_by="ingestion_validator",
                    )
                    count += 1
        if count:
            log.warning(
                "quarantine.auto_quarantine",
                doc_id=doc_id,
                count=count,
            )
        return count
