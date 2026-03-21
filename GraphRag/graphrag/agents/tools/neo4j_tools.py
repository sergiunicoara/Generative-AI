"""ADK FunctionTool wrappers for Neo4j operations."""

from __future__ import annotations

import asyncio

from graphrag.graph.neo4j_client import get_neo4j


def search_graph(query_text: str, top_k: int = 10) -> list[dict]:
    """Search the knowledge graph for chunks similar to query_text."""
    from graphrag.ingestion.embedder import Embedder
    embedder = Embedder()
    embedding = asyncio.run(embedder.embed_text(query_text))
    neo4j = get_neo4j()
    return asyncio.run(neo4j.vector_search_chunks(embedding, top_k=top_k))


def get_community(community_id: str) -> dict | None:
    """Retrieve a specific community node by ID."""
    neo4j = get_neo4j()
    results = asyncio.run(
        neo4j.run(
            "MATCH (c:Community {id: $id}) RETURN c.summary AS summary, c.level AS level",
            id=community_id,
        )
    )
    return results[0] if results else None


def get_neighbors(entity_name: str) -> list[dict]:
    """Get entities that are directly related to the given entity."""
    neo4j = get_neo4j()
    return asyncio.run(
        neo4j.run(
            """
            MATCH (e:Entity {name: $name})-[r:RELATES_TO]-(neighbor:Entity)
            RETURN neighbor.name AS name, neighbor.type AS type, r.relation AS relation
            LIMIT 20
            """,
            name=entity_name,
        )
    )
