"""Incremental community detection — rebuild only affected communities.

Problem solved
--------------
Full Leiden community detection re-processes the entire entity graph on every
ingestion event.  For large tenants (100k+ entities), this is a multi-second
blocking operation triggered even when only a handful of new entities were added.

Incremental approach
--------------------
Instead of rebuilding all communities, we:
  1. Record a RebuildPoint node after every full community build.
  2. On subsequent ingestions, query for entities with recorded_at > last rebuild.
  3. If the fraction of changed entities is below a threshold, only rebuild
     communities that contain at least one changed entity.
  4. If the fraction exceeds the threshold, fall back to a full rebuild.

This reduces rebuild latency from O(total_entities) to O(changed_entities +
affected_community_sizes) for typical incremental workloads.

Architecture
------------
IncrementalCommunityDetector:
  - get_last_rebuild_point(tenant) → timestamp of last full rebuild
  - get_changed_entities(tenant, since) → entities modified after timestamp
  - get_affected_community_ids(entity_ids, tenant) → community IDs to rebuild
  - rebuild_affected_communities(tenant, dry_run) → partial rebuild
  - should_full_rebuild(tenant, change_fraction_threshold) → bool
  - record_rebuild_point(tenant) → RebuildPoint node id
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import structlog

log = structlog.get_logger(__name__)

# Fraction of entities changed that triggers a full rebuild instead of incremental.
DEFAULT_FULL_REBUILD_THRESHOLD = 0.20   # 20 %


class IncrementalCommunityDetector:
    """
    Detect and rebuild only communities affected by recent entity changes.

    Usage::

        detector = IncrementalCommunityDetector(neo4j_client)

        # After ingesting new documents:
        if await detector.should_full_rebuild(tenant):
            await community_builder.build()          # full Leiden run
            await detector.record_rebuild_point(tenant)
        else:
            report = await detector.rebuild_affected_communities(tenant)
            print(report["communities_rebuilt"])

        # Inspect what changed since last rebuild:
        changed = await detector.get_changed_entities(tenant)
    """

    def __init__(
        self,
        neo4j_client,
        full_rebuild_threshold: float = DEFAULT_FULL_REBUILD_THRESHOLD,
    ):
        self._neo4j     = neo4j_client
        self._threshold = full_rebuild_threshold

    # ── Public API ─────────────────────────────────────────────────────────────

    async def get_last_rebuild_point(
        self,
        tenant: str = "default",
    ) -> datetime | None:
        """Return the timestamp of the most recent community rebuild, or None."""
        rows = await self._neo4j.run(
            """
            MATCH (rp:CommunityRebuildPoint {tenant: $tenant})
            RETURN rp.rebuilt_at AS rebuilt_at
            ORDER BY rp.rebuilt_at DESC
            LIMIT 1
            """,
            tenant=tenant,
        )
        if not rows:
            return None
        raw = rows[0].get("rebuilt_at")
        if raw is None:
            return None
        # Neo4j datetime objects may be strings or native datetime
        if isinstance(raw, str):
            try:
                return datetime.fromisoformat(raw)
            except ValueError:
                return None
        if hasattr(raw, "to_native"):
            return raw.to_native()
        return raw  # already a datetime

    async def get_changed_entities(
        self,
        tenant: str = "default",
        since: datetime | None = None,
    ) -> list[dict]:
        """
        Return entities whose recorded_at is later than *since*.

        If *since* is None, queries from the last CommunityRebuildPoint.
        If no rebuild point exists, returns all entities (full-rebuild scenario).
        """
        if since is None:
            since = await self.get_last_rebuild_point(tenant)

        if since is None:
            # No rebuild point — all entities are "changed"
            rows = await self._neo4j.run(
                """
                MATCH (e:Entity {tenant: $tenant})
                WHERE NOT e.quarantined = true
                RETURN e.name AS name, e.type AS type, e.recorded_at AS recorded_at
                LIMIT 5000
                """,
                tenant=tenant,
            )
            return [dict(r) for r in rows]

        rows = await self._neo4j.run(
            """
            MATCH (e:Entity {tenant: $tenant})
            WHERE NOT e.quarantined = true
              AND e.recorded_at > datetime($since)
            RETURN e.name AS name, e.type AS type, e.recorded_at AS recorded_at
            ORDER BY e.recorded_at DESC
            """,
            tenant=tenant,
            since=since.isoformat() if isinstance(since, datetime) else str(since),
        )
        return [dict(r) for r in rows]

    async def get_affected_community_ids(
        self,
        changed_entity_names: list[str],
        tenant: str = "default",
    ) -> set[str]:
        """Return IDs of communities that contain at least one changed entity."""
        if not changed_entity_names:
            return set()

        rows = await self._neo4j.run(
            """
            UNWIND $names AS name
            MATCH (e:Entity {name: name, tenant: $tenant})-[:MEMBER_OF]->(c:Community {tenant: $tenant})
            RETURN DISTINCT c.id AS community_id
            """,
            names=changed_entity_names,
            tenant=tenant,
        )
        return {r["community_id"] for r in rows}

    async def should_full_rebuild(
        self,
        tenant: str = "default",
        threshold: float | None = None,
    ) -> bool:
        """
        Return True if the fraction of changed entities exceeds the threshold.

        When True, callers should fall back to a full Leiden rebuild.
        """
        thr = threshold if threshold is not None else self._threshold

        changed = await self.get_changed_entities(tenant)
        if not changed:
            return False

        total_rows = await self._neo4j.run(
            """
            MATCH (e:Entity {tenant: $tenant})
            WHERE NOT e.quarantined = true
            RETURN count(e) AS n
            """,
            tenant=tenant,
        )
        total = total_rows[0].get("n", 1) if total_rows else 1
        fraction = len(changed) / max(total, 1)

        log.info(
            "incremental_community.change_fraction",
            changed=len(changed),
            total=total,
            fraction=round(fraction, 4),
            threshold=thr,
            tenant=tenant,
        )
        return fraction >= thr

    async def rebuild_affected_communities(
        self,
        tenant: str = "default",
        dry_run: bool = False,
    ) -> dict:
        """
        Rebuild only communities that contain at least one recently changed entity.

        Process:
          1. Get changed entities since last rebuild.
          2. Find which communities they belong to.
          3. For each affected community, fetch its current member entities.
          4. Re-run Leiden on the subgraph induced by those entities + their
             1-hop neighbours (to capture new edges).
          5. Persist updated community memberships.

        Returns a report dict.
        """
        changed = await self.get_changed_entities(tenant)
        if not changed:
            log.info("incremental_community.no_changes", tenant=tenant)
            return {
                "changed_entities": 0,
                "communities_affected": 0,
                "communities_rebuilt": 0,
                "dry_run": dry_run,
            }

        changed_names = [e["name"] for e in changed]
        affected_ids  = await self.get_affected_community_ids(changed_names, tenant)

        # Also include entities with no community yet (new nodes)
        orphan_rows = await self._neo4j.run(
            """
            UNWIND $names AS name
            MATCH (e:Entity {name: name, tenant: $tenant})
            WHERE NOT EXISTS {
                MATCH (e)-[:MEMBER_OF]->(c:Community {tenant: $tenant})
            }
            RETURN e.name AS name
            """,
            names=changed_names,
            tenant=tenant,
        )
        has_orphans = len(orphan_rows) > 0

        if not affected_ids and not has_orphans:
            log.info("incremental_community.no_affected_communities",
                     changed=len(changed), tenant=tenant)
            await self.record_rebuild_point(tenant)
            return {
                "changed_entities": len(changed),
                "communities_affected": 0,
                "communities_rebuilt": 0,
                "dry_run": dry_run,
            }

        if dry_run:
            return {
                "changed_entities":    len(changed),
                "communities_affected": len(affected_ids),
                "communities_rebuilt": 0,
                "dry_run": True,
                "orphan_entities":     len(orphan_rows),
            }

        # For each affected community, re-cluster its member subgraph
        rebuilt = 0
        for comm_id in affected_ids:
            ok = await self._rebuild_one_community(comm_id, tenant)
            if ok:
                rebuilt += 1

        # Handle orphaned new entities: attach to nearest community by embedding
        orphaned_attached = 0
        if has_orphans:
            orphaned_attached = await self._attach_orphans_to_communities(
                [r["name"] for r in orphan_rows], tenant
            )

        await self.record_rebuild_point(tenant)

        report = {
            "changed_entities":    len(changed),
            "communities_affected": len(affected_ids),
            "communities_rebuilt": rebuilt,
            "orphans_attached":    orphaned_attached,
            "dry_run": False,
        }
        log.info("incremental_community.rebuilt", **report, tenant=tenant)
        return report

    async def record_rebuild_point(self, tenant: str = "default") -> str:
        """
        Store a CommunityRebuildPoint node marking the current time as rebuilt_at.

        Called after every (full or partial) community build so that future
        incremental checks have a correct baseline.
        """
        rp_id = str(uuid4())
        await self._neo4j.run(
            """
            CREATE (rp:CommunityRebuildPoint {
                id:         $id,
                tenant:     $tenant,
                rebuilt_at: datetime()
            })
            """,
            id=rp_id,
            tenant=tenant,
        )
        log.info("incremental_community.rebuild_point_recorded",
                 id=rp_id, tenant=tenant)
        return rp_id

    async def community_change_summary(self, tenant: str = "default") -> dict:
        """Return a diagnostic summary of community change state."""
        last = await self.get_last_rebuild_point(tenant)
        changed = await self.get_changed_entities(tenant)
        affected = await self.get_affected_community_ids(
            [e["name"] for e in changed], tenant
        )
        total_rows = await self._neo4j.run(
            "MATCH (e:Entity {tenant: $tenant}) RETURN count(e) AS n",
            tenant=tenant,
        )
        total = total_rows[0].get("n", 0) if total_rows else 0

        return {
            "tenant":               tenant,
            "last_rebuild_at":      last.isoformat() if last else None,
            "changed_entities":     len(changed),
            "total_entities":       total,
            "change_fraction":      round(len(changed) / max(total, 1), 4),
            "affected_communities": len(affected),
            "full_rebuild_recommended": await self.should_full_rebuild(tenant),
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _rebuild_one_community(
        self,
        community_id: str,
        tenant: str,
    ) -> bool:
        """
        Re-cluster entities in community *community_id* plus their 1-hop neighbours.

        Uses Leiden if available; falls back to keeping existing memberships.
        Returns True if rebuild succeeded.
        """
        try:
            import networkx as nx
        except ImportError:
            log.warning("incremental_community.networkx_missing")
            return False

        # Fetch current members + 1-hop neighbours
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity)-[:MEMBER_OF]->(c:Community {id: $comm_id, tenant: $tenant})
            WITH e
            OPTIONAL MATCH (e)-[r:RELATES_TO]->(n:Entity {tenant: $tenant})
            RETURN DISTINCT e.name AS name, e.type AS type,
                   n.name AS neighbour, r.confidence AS conf
            UNION
            MATCH (e:Entity)-[:MEMBER_OF]->(c:Community {id: $comm_id, tenant: $tenant})
            WITH e
            OPTIONAL MATCH (n:Entity {tenant: $tenant})-[r:RELATES_TO]->(e)
            RETURN DISTINCT e.name AS name, e.type AS type,
                   n.name AS neighbour, r.confidence AS conf
            """,
            comm_id=community_id,
            tenant=tenant,
        )

        G = nx.Graph()
        for r in rows:
            if r.get("name"):
                G.add_node(r["name"])
            if r.get("name") and r.get("neighbour"):
                G.add_edge(r["name"], r["neighbour"],
                           weight=float(r.get("conf") or 1.0))

        if G.number_of_nodes() < 2:
            return False

        try:
            from graspologic.partition import leiden
            partition = leiden(G, resolution=1.0, random_seed=42)
        except ImportError:
            # Can't re-cluster without Leiden; skip (existing memberships stay)
            return False
        except Exception as exc:
            log.warning("incremental_community.leiden_error", error=str(exc))
            return False

        # Update community memberships
        for node_name, comm_int in partition.items():
            # We use the original community_id with a sub-partition suffix
            new_comm_id = f"{community_id}_{comm_int}"
            await self._neo4j.run(
                """
                MATCH (e:Entity {name: $name, tenant: $tenant})
                OPTIONAL MATCH (e)-[m:MEMBER_OF]->(c:Community {tenant: $tenant})
                DELETE m
                WITH e
                MERGE (nc:Community {id: $comm_id, tenant: $tenant})
                ON CREATE SET nc.member_count = 0,
                              nc.detection_method = 'incremental_leiden'
                MERGE (e)-[:MEMBER_OF]->(nc)
                SET nc.member_count = nc.member_count + 1,
                    nc.updated_at   = datetime()
                """,
                name=node_name,
                tenant=tenant,
                comm_id=new_comm_id,
            )
        return True

    async def _attach_orphans_to_communities(
        self,
        orphan_names: list[str],
        tenant: str,
    ) -> int:
        """
        Attach orphaned entities to the nearest community by shared neighbours.

        For each orphan, count RELATES_TO neighbours that are already community
        members and assign the orphan to the community with the most shared members.
        """
        attached = 0
        for name in orphan_names:
            rows = await self._neo4j.run(
                """
                MATCH (e:Entity {name: $name, tenant: $tenant})
                      -[:RELATES_TO]-(n:Entity {tenant: $tenant})
                      -[:MEMBER_OF]->(c:Community {tenant: $tenant})
                RETURN c.id AS comm_id, count(n) AS shared
                ORDER BY shared DESC
                LIMIT 1
                """,
                name=name,
                tenant=tenant,
            )
            if rows and rows[0].get("comm_id"):
                await self._neo4j.run(
                    """
                    MATCH (e:Entity {name: $name, tenant: $tenant})
                    MATCH (c:Community {id: $comm_id, tenant: $tenant})
                    MERGE (e)-[:MEMBER_OF]->(c)
                    SET c.member_count = c.member_count + 1,
                        c.updated_at   = datetime()
                    """,
                    name=name,
                    tenant=tenant,
                    comm_id=rows[0]["comm_id"],
                )
                attached += 1
        return attached
