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

        # Record a rebuild point so the incremental detector has a fresh baseline.
        # Without this, community_change_summary() treats all entities as "changed".
        try:
            from graphrag.graph.incremental_community import IncrementalCommunityDetector
            detector = IncrementalCommunityDetector(self._neo4j)
            await detector.record_rebuild_point(self._tenant)
        except Exception as exc:  # pragma: no cover
            log.warning("community_builder.rebuild_point_failed", error=str(exc))

        log.info("community_builder.done", count=len(communities), tenant=self._tenant)
        return communities

    def _build_networkx_graph(
        self, entities: list[dict], relations: list[dict]
    ) -> nx.Graph:
        G = nx.Graph()
        for e in entities:
            G.add_node(e["id"], name=e["name"], type=e["type"])
        for r in relations:
            # Use confidence as weight when weight property is absent/null
            weight = r.get("weight") or r.get("confidence") or 1.0
            G.add_edge(r["source_id"], r["target_id"], weight=float(weight))
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

    # ── HDBSCAN semantic community detection ──────────────────────────────────

    async def build_semantic_communities(self) -> dict:
        """
        Build semantic communities using HDBSCAN on entity embeddings.

        This is a *parallel signal* to Leiden structural communities — not a
        replacement.  Running both allows operators to detect divergence:
        - High overlap  → structure and semantics agree (high confidence).
        - Low overlap   → structurally connected entities are semantically
                          distant (possible KG quality issue).

        Results are written as Community nodes tagged with
        ``detection_method: 'hdbscan'`` so they can be queried separately.

        Returns a summary dict including:
          - ``n_semantic_communities``  : HDBSCAN cluster count (excl. noise)
          - ``n_noise_entities``        : HDBSCAN noise points (label = -1)
          - ``n_structural_communities``: Leiden community count for comparison
          - ``overlap_score``           : Jaccard-based community overlap [0, 1]
          - ``divergence_score``        : 1 - overlap_score
        """
        try:
            import numpy as np
        except ImportError:
            log.error("community_builder.numpy_missing",
                      fix="pip install numpy")
            return {"error": "numpy_missing"}

        try:
            import hdbscan as hdbscan_lib
        except ImportError:
            log.warning(
                "community_builder.hdbscan_missing",
                fix="pip install hdbscan",
                impact="semantic community signal unavailable",
            )
            return {"error": "hdbscan_missing"}

        # Fetch entities with embeddings
        rows = await self._neo4j.get_all_entities(tenant=self._tenant)
        embed_rows = await self._neo4j.run(
            """
            MATCH (e:Entity {tenant: $tenant})
            WHERE e.embedding IS NOT NULL AND size(e.embedding) > 0
              AND NOT e.quarantined = true
            RETURN e.name AS name, e.embedding AS embedding
            """,
            tenant=self._tenant,
        )

        if len(embed_rows) < 10:
            log.warning("community_builder.hdbscan_too_few_embeddings",
                        n=len(embed_rows), tenant=self._tenant)
            return {"error": "too_few_embeddings", "n": len(embed_rows)}

        names      = [r["name"] for r in embed_rows]
        embeddings = np.array([r["embedding"] for r in embed_rows], dtype=np.float32)

        # L2-normalise for cosine-like distance
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings = embeddings / norms

        # Run HDBSCAN — min_cluster_size configurable; default 5
        min_cluster_size = self._cfg.get("hdbscan_min_cluster_size", 5)
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",      # on L2-normalised vectors ≈ cosine
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(embeddings)

        # Write Community nodes tagged with detection_method='hdbscan'
        cluster_map: dict[int, list[str]] = {}
        for name, label in zip(names, labels):
            cluster_map.setdefault(int(label), []).append(name)

        noise_entities = cluster_map.pop(-1, [])   # HDBSCAN uses -1 for noise
        semantic_comms: list[Community] = []
        for label, members in cluster_map.items():
            sem_comm = Community(
                id=str(uuid4()),
                level=0,
                member_entity_ids=members,
                member_count=len(members),
            )
            sem_comm.summary = f"[hdbscan:cluster_{label}]"
            semantic_comms.append(sem_comm)

        # Write to Neo4j as hdbscan communities (separate from Leiden)
        for comm in semantic_comms:
            comm.tenant = self._tenant
            await self._neo4j.run(
                """
                MERGE (c:Community {id: $id, tenant: $tenant})
                SET c.member_count       = $count,
                    c.level              = 0,
                    c.detection_method   = 'hdbscan',
                    c.updated_at         = datetime()
                """,
                id=comm.id,
                tenant=self._tenant,
                count=comm.member_count,
            )
            for member_name in comm.member_entity_ids:
                await self._neo4j.run(
                    """
                    MATCH (e:Entity {name: $name, tenant: $tenant})
                    MATCH (c:Community {id: $comm_id, tenant: $tenant})
                    MERGE (e)-[:SEMANTIC_MEMBER_OF]->(c)
                    """,
                    name=member_name,
                    tenant=self._tenant,
                    comm_id=comm.id,
                )

        # Compute overlap with structural (Leiden) communities
        overlap_score = await self._compute_community_overlap(
            semantic_comms, method="hdbscan"
        )

        summary = {
            "n_semantic_communities":  len(semantic_comms),
            "n_noise_entities":        len(noise_entities),
            "total_entities_clustered": len(names) - len(noise_entities),
            "overlap_score":           round(overlap_score, 4),
            "divergence_score":        round(1 - overlap_score, 4),
        }

        if overlap_score < 0.4:
            log.warning(
                "community_builder.high_structural_semantic_divergence",
                divergence=round(1 - overlap_score, 4),
                tenant=self._tenant,
                impact="Structure and semantics disagree — review entity relations",
            )
        else:
            log.info("community_builder.hdbscan_done", **summary, tenant=self._tenant)

        return summary

    async def _compute_community_overlap(
        self,
        semantic_comms: list[Community],
        method: str = "hdbscan",
    ) -> float:
        """
        Compute average Jaccard overlap between semantic and structural communities.

        For each semantic community S:
          best_j = max over all structural communities L of Jaccard(S, L)
        overlap_score = mean(best_j over all semantic communities)

        Returns 0.0 if no structural communities exist or no overlap is found.
        """
        if not semantic_comms:
            return 0.0

        # Load structural communities
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity {tenant: $tenant})-[:MEMBER_OF]->(c:Community {tenant: $tenant})
            WHERE NOT (c.detection_method = 'hdbscan')
            RETURN c.id AS comm_id, collect(e.name) AS members
            """,
            tenant=self._tenant,
        )
        if not rows:
            return 0.0

        structural: list[set] = [set(r["members"]) for r in rows]

        jaccard_scores: list[float] = []
        for sem_comm in semantic_comms:
            sem_set = set(sem_comm.member_entity_ids)
            best_j  = 0.0
            for struct_set in structural:
                intersection = len(sem_set & struct_set)
                union        = len(sem_set | struct_set)
                j = intersection / union if union > 0 else 0.0
                if j > best_j:
                    best_j = j
            jaccard_scores.append(best_j)

        return sum(jaccard_scores) / len(jaccard_scores)
