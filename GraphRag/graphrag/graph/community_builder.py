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

    def __init__(self):
        self._cfg = get_settings().graph
        self._neo4j = get_neo4j()

    async def build(self) -> list[Community]:
        log.info("community_builder.start")

        entities = await self._neo4j.get_all_entities()
        relations = await self._neo4j.get_all_relations()

        if not entities:
            log.warning("community_builder.no_entities")
            return []

        G = self._build_networkx_graph(entities, relations)
        communities = self._run_leiden(G)

        for community in communities:
            await self._neo4j.merge_community(community)

        log.info("community_builder.done", count=len(communities))
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
            log.warning("community_builder.graspologic_missing, falling back to connected_components")
            return self._fallback_components(G)

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
