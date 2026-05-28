"""Build hierarchical communities from the Neo4j entity graph using Leiden algorithm."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import networkx as nx
import structlog

from graphrag.core.config import get_settings
from graphrag.core.models import Community
from graphrag.graph.neo4j_client import get_neo4j

log = structlog.get_logger(__name__)


class CommunityBuilder:
    """
    Fetches entities + relations from Neo4j, runs Leiden community detection
    (via graspologic), and writes Community nodes back to Neo4j.
    """

    def __init__(self, tenant: str = "default"):
        self._cfg = get_settings().graph
        self._neo4j = get_neo4j()
        self._tenant = tenant

    async def build(self) -> list[Community]:
        log.info("community_builder.start", tenant=self._tenant)

        entities = await self._neo4j.get_all_entities(tenant=self._tenant)
        relations = await self._neo4j.get_all_relations(tenant=self._tenant)

        if not entities:
            log.warning("community_builder.no_entities", tenant=self._tenant)
            return []

        G = self._build_networkx_graph(entities, relations)
        communities = self._run_leiden(G)

        # Surface community quality — if all communities are fallback-tagged,
        # global search will run on connected-components structure, not Leiden
        # hierarchy.  Operators should see this in production logs.
        fallback_count = sum(
            1 for c in communities
            if c.summary.startswith("[fallback:")
        )
        if fallback_count == len(communities) and communities:
            log.error(
                "community_builder.quality_degraded",
                algorithm="connected_components",
                community_count=len(communities),
                tenant=self._tenant,
                impact="global search operating on flat components, not Leiden hierarchy",
                fix="pip install graspologic  OR  set graph.require_leiden=true to fail fast",
            )
        else:
            log.info(
                "community_builder.quality_ok",
                algorithm="leiden",
                community_count=len(communities),
                tenant=self._tenant,
            )

        await self._neo4j.clear_communities(tenant=self._tenant)
        for community in communities:
            community.tenant = self._tenant
            await self._neo4j.merge_community(community)

        log.info("community_builder.done", count=len(communities), tenant=self._tenant)
        return communities

    def _build_networkx_graph(
        self, entities: list[dict], relations: list[dict]
    ) -> nx.Graph:
        G = nx.Graph()
        for e in entities:
            G.add_node(e["id"], name=e["name"], type=e["type"])
        for r in relations:
            G.add_edge(r["source_id"], r["target_id"], weight=r.get("weight", 1.0))
        return G

    def _run_leiden(self, G: nx.Graph) -> list[Community]:
        try:
            from graspologic.partition import leiden
        except ImportError:
            require_leiden = self._cfg.get("require_leiden", False)
            if require_leiden:
                raise RuntimeError(
                    "graspologic is not installed but graph.require_leiden=true. "
                    "Install it with: pip install graspologic"
                )
            log.error(
                "community_builder.graspologic_missing",
                impact=(
                    "Falling back to connected-components — hierarchical "
                    "community structure lost. Global-search quality will be "
                    "materially lower. Install graspologic to restore full Leiden."
                ),
                fix="pip install graspologic",
            )
            communities = self._fallback_components(G)
            # Tag communities so downstream code can see they are low-quality
            for c in communities:
                c.summary = "[fallback: connected_components — graspologic missing]"
            return communities

        nodes = list(G.nodes())
        if len(nodes) < 2:
            return []

        # Run Leiden at each level of resolution
        communities: list[Community] = []
        resolutions = [
            self._cfg.get("leiden_resolution", 1.0) * (0.5**level)
            for level in range(self._cfg.get("community_levels", 3))
        ]

        for level, resolution in enumerate(resolutions):
            partition = leiden(G, resolution=resolution, random_seed=42)
            # partition maps node_id -> community_int
            community_map: dict[int, list[str]] = {}
            for node_id, community_int in partition.items():
                community_map.setdefault(community_int, []).append(node_id)

            min_size = self._cfg.get("min_community_size", 3)
            for members in community_map.values():
                if len(members) >= min_size:
                    communities.append(
                        Community(
                            id=str(uuid4()),
                            level=level,
                            member_entity_ids=members,
                            member_count=len(members),
                        )
                    )

        return communities

    def _fallback_components(self, G: nx.Graph) -> list[Community]:
        """Fallback: use connected components as communities (level 0 only)."""
        communities = []
        for component in nx.connected_components(G):
            members = list(component)
            if len(members) >= 2:
                communities.append(
                    Community(
                        id=str(uuid4()),
                        level=0,
                        member_entity_ids=members,
                        member_count=len(members),
                    )
                )
        return communities
