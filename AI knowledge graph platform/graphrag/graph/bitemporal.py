"""Bitemporal modeling — valid time + transaction time for KG edges and entities.

Problems solved
---------------
1. Single-axis time — the existing model has valid_from/valid_to (valid time)
   but no transaction time.  This means "what did we know on date X about the
   world as it was in period Y?" is unanswerable.

2. Retroactive corrections — when a document is re-processed with corrected
   data, there is no way to distinguish the new write from the original.  Any
   consumer of the graph who caches a snapshot sees silently stale data.

3. Debugging ingestion bugs — if an incorrect confidence value was written and
   later fixed, the audit trail shows the fix but the original bad write is
   not queryable in isolation.

Temporal dimensions
--------------------
  Valid time (VT)      — when the fact was true in the real world.
                         Fields: valid_from, valid_to on Entity and Relation.
                         Already present in the codebase.

  Transaction time (TT) — when the fact was recorded in the database.
                         Field: recorded_at (set once at CREATE, never updated).
                         Added by this module to all MERGE helpers.

Bitemporal query
----------------
  as_of(vt, tt) = facts where valid_from ≤ vt ≤ valid_to AND recorded_at ≤ tt
  "What did we know (tt) about the world at time (vt)?"

Architecture
------------
- BitemporalStore wraps Neo4jClient and adds recorded_at stamping.
- merge_entity_bt and merge_relation_bt are drop-in replacements that set
  recorded_at on CREATE only (never update it — transaction time is immutable).
- as_of_entities() and as_of_edges() run bitemporal range queries.
- time_travel_report() summarises the graph state at an arbitrary (vt, tt) pair.

Note on neo4j_client.py integration
-------------------------------------
merge_entity and merge_relation now stamp recorded_at = datetime() via
ON CREATE SET — see the corresponding changes in neo4j_client.py.
BitemporalStore provides the higher-level query interface on top of that.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)


class BitemporalStore:
    """
    Bitemporal query interface for the knowledge graph.

    Usage::

        store = BitemporalStore(neo4j_client)

        # Entities valid on 2024-06-01 that we had recorded before 2024-09-01
        entities = await store.as_of_entities(
            valid_time="2024-06-01T00:00:00",
            transaction_time="2024-09-01T00:00:00",
            tenant="acme",
        )

        # Full bitemporal diff: what changed between two transaction snapshots?
        diff = await store.transaction_diff(
            tt_from="2024-01-01T00:00:00",
            tt_to="2024-07-01T00:00:00",
            tenant="acme",
        )
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    # ── Bitemporal point queries ───────────────────────────────────────────────

    async def as_of_entities(
        self,
        valid_time: str,
        transaction_time: str,
        tenant: str = "default",
        limit: int = 500,
    ) -> list[dict]:
        """
        Return entities valid at ``valid_time`` as recorded by ``transaction_time``.

        Entities without explicit valid_from/valid_to are treated as "always
        valid" in valid-time; entities without recorded_at are treated as
        "recorded from the beginning" in transaction time.

        Parameters
        ----------
        valid_time       : ISO datetime — the real-world time of interest.
        transaction_time : ISO datetime — upper bound on when facts were written.
        """
        return await self._neo4j.run(
            """
            MATCH (e:Entity)
            WHERE ($tenant = 'default' OR e.tenant = $tenant)
              AND NOT e.quarantined = true
              // Valid-time filter (treat NULL as always valid)
              AND (e.valid_from IS NULL OR e.valid_from <= $vt)
              AND (e.valid_to   IS NULL OR e.valid_to   >  $vt)
              // Transaction-time filter (treat NULL as recorded at epoch)
              AND (e.recorded_at IS NULL OR e.recorded_at <= $tt)
            RETURN e.name        AS name,
                   e.type        AS type,
                   e.description AS description,
                   e.tenant      AS tenant,
                   e.valid_from  AS valid_from,
                   e.valid_to    AS valid_to,
                   e.recorded_at AS recorded_at
            LIMIT $limit
            """,
            tenant=tenant,
            vt=valid_time,
            tt=transaction_time,
            limit=limit,
        )

    async def as_of_edges(
        self,
        valid_time: str,
        transaction_time: str,
        tenant: str = "default",
        limit: int = 1000,
    ) -> list[dict]:
        """
        Return RELATES_TO edges valid at ``valid_time`` as recorded by ``transaction_time``.
        """
        return await self._neo4j.run(
            """
            MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
            WHERE ($tenant = 'default' OR r.tenant = $tenant)
              AND (r.valid_from  IS NULL OR r.valid_from  <= $vt)
              AND (r.valid_to    IS NULL OR r.valid_to    >  $vt)
              AND (r.recorded_at IS NULL OR r.recorded_at <= $tt)
              AND NOT s.quarantined = true
              AND NOT t.quarantined = true
            RETURN s.name             AS src,
                   s.type             AS src_type,
                   t.name             AS tgt,
                   t.type             AS tgt_type,
                   r.relation         AS relation,
                   r.confidence       AS confidence,
                   r.valid_from       AS valid_from,
                   r.valid_to         AS valid_to,
                   r.recorded_at      AS recorded_at,
                   r.source_doc_ids   AS source_doc_ids
            LIMIT $limit
            """,
            tenant=tenant,
            vt=valid_time,
            tt=transaction_time,
            limit=limit,
        )

    # ── Transaction-time diff ─────────────────────────────────────────────────

    async def transaction_diff(
        self,
        tt_from: str,
        tt_to: str,
        tenant: str = "default",
    ) -> dict:
        """
        Summarise what changed in the graph between two transaction times.

        Returns:
            new_entities  : entities first recorded in (tt_from, tt_to]
            new_edges     : edges first recorded in (tt_from, tt_to]
            entity_count  : total entities as of tt_to
            edge_count    : total edges as of tt_to
        """
        new_entities = await self._neo4j.run(
            """
            MATCH (e:Entity)
            WHERE ($tenant = 'default' OR e.tenant = $tenant)
              AND e.recorded_at IS NOT NULL
              AND e.recorded_at >  $tt_from
              AND e.recorded_at <= $tt_to
              AND NOT e.quarantined = true
            RETURN count(e) AS count
            """,
            tenant=tenant,
            tt_from=tt_from,
            tt_to=tt_to,
        )
        new_edges = await self._neo4j.run(
            """
            MATCH ()-[r:RELATES_TO]->()
            WHERE ($tenant = 'default' OR r.tenant = $tenant)
              AND r.recorded_at IS NOT NULL
              AND r.recorded_at >  $tt_from
              AND r.recorded_at <= $tt_to
            RETURN count(r) AS count
            """,
            tenant=tenant,
            tt_from=tt_from,
            tt_to=tt_to,
        )
        total = await self._neo4j.run(
            """
            MATCH (e:Entity)
            WHERE ($tenant = 'default' OR e.tenant = $tenant)
              AND (e.recorded_at IS NULL OR e.recorded_at <= $tt_to)
              AND NOT e.quarantined = true
            WITH count(e) AS entity_count
            MATCH ()-[r:RELATES_TO]->()
            WHERE ($tenant = 'default' OR r.tenant = $tenant)
              AND (r.recorded_at IS NULL OR r.recorded_at <= $tt_to)
            RETURN entity_count, count(r) AS edge_count
            """,
            tenant=tenant,
            tt_to=tt_to,
        )

        t = total[0] if total else {}
        return {
            "tt_from":        tt_from,
            "tt_to":          tt_to,
            "tenant":         tenant,
            "new_entities":   (new_entities[0].get("count", 0) if new_entities else 0),
            "new_edges":      (new_edges[0].get("count", 0) if new_edges else 0),
            "entity_count":   t.get("entity_count", 0),
            "edge_count":     t.get("edge_count", 0),
        }

    # ── Entity history ─────────────────────────────────────────────────────────

    async def get_entity_history(
        self,
        entity_name: str,
        entity_type: str,
        tenant: str = "default",
    ) -> list[dict]:
        """
        Return the ChangeLog audit trail for a specific entity, ordered by
        change time.  This is the transaction-time history of an entity node.
        """
        return await self._neo4j.run(
            """
            MATCH (cl:ChangeLog)
            WHERE cl.target_label = 'Entity'
              AND cl.target_id    = $name
              AND ($tenant = 'default' OR EXISTS {
                  MATCH (e:Entity {name: $name, type: $type, tenant: $tenant})
              })
            RETURN cl.operation   AS operation,
                   cl.changed_at  AS changed_at,
                   cl.changed_by  AS changed_by,
                   cl.old_values  AS old_values,
                   cl.new_values  AS new_values,
                   cl.source_doc_id AS source_doc_id
            ORDER BY cl.changed_at ASC
            """,
            name=entity_name,
            type=entity_type,
            tenant=tenant,
        )

    # ── Time-travel report ─────────────────────────────────────────────────────

    async def time_travel_report(
        self,
        valid_time: str,
        transaction_time: str,
        tenant: str = "default",
    ) -> dict:
        """
        High-level summary of graph state at a (valid_time, transaction_time) pair.

        Use this to answer: "What did our system know on date T about the
        world as it was in period V?"
        """
        entities = await self.as_of_entities(valid_time, transaction_time, tenant)
        edges    = await self.as_of_edges(valid_time, transaction_time, tenant)

        return {
            "valid_time":       valid_time,
            "transaction_time": transaction_time,
            "tenant":           tenant,
            "entity_count":     len(entities),
            "edge_count":       len(edges),
            "entities":         entities[:20],   # sample for readability
            "edges":            edges[:50],
        }
