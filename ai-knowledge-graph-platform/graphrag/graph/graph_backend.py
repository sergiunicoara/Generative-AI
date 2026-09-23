"""``GraphBackend`` -- the property-graph counterpart to ``triplestore.py``'s
``SPARQLSource`` Protocol.

``Neo4jClient`` (neo4j_client.py) has 60 async methods and no backend
abstraction -- every one of its callers (~100 files) imports ``get_neo4j()``
and depends on a concrete ``Neo4jClient``. This module does **not** change
that. It extracts a Protocol covering the 9 methods below, which
``Neo4jClient`` already satisfies structurally with no code changes, so that
an alternative backend (``gremlin_client.py``'s ``GremlinBackend``, for
Neptune/Cosmos DB's Gremlin API) can implement the same shape for the
retrieval-critical subset.

What this Protocol deliberately does NOT cover
------------------------------------------------
``Neo4jClient`` has ~50 other methods with no Gremlin equivalent here:
schema/corpus-lifecycle admin (``init_schema``, ``begin_corpus_update``,
``advance_corpus_revision``, ...), document/chunk/structured-data ingestion
(``merge_document``, ``merge_chunk``, ``merge_structured_tables``, ...),
batch variants (``merge_entities_batch``, ``merge_relations_batch``, ...),
community detection (``merge_community``, ``clear_communities``), vector/BM25
search (``vector_search_chunks``, ``bm25_search_entities``, ...), multi-hop
traversal (``get_multihop_chunks``), and PageRank/GDS-dependent analytics
(``run_pagerank``, ``get_top_entities_by_pagerank``, ...). None of these are
part of ``GraphBackend`` and no ``GremlinBackend`` method exists for them --
same honesty convention this package's ``triplestore.py`` uses for Neptune's
``load()`` gap (documented, not silently omitted).

This also does not introduce a ``get_graph_backend()`` accessor or migrate
any ``get_neo4j()`` call site -- the platform's actual runtime backend is
unchanged. ``docs/roadmap.md`` itself scopes Gremlin/Neptune as
"interoperability targets, not additional sources of truth."
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from graphrag.core.models import Entity, Relation


@runtime_checkable
class GraphBackend(Protocol):
    """What a caller needs from a property-graph backend for entity/relation
    CRUD and 1-hop retrieval. ``Neo4jClient`` satisfies this structurally;
    ``gremlin_client.GremlinBackend`` implements it explicitly.
    """

    async def close(self) -> None: ...

    async def entity_exists(
        self, name: str, entity_type: str, tenant: str = "default",
    ) -> bool: ...

    async def merge_entity(self, entity: Entity, tenant: str = "default") -> None: ...

    async def merge_mentions(
        self, chunk_id: str, entity_name: str, entity_type: str, tenant: str = "default",
    ) -> None: ...

    async def merge_relation(
        self,
        rel: Relation,
        src_name: str,
        src_type: str,
        tgt_name: str,
        tgt_type: str,
        tenant: str = "default",
    ) -> None: ...

    async def get_entity_neighbors(
        self,
        chunk_ids: list[str],
        as_of: str | None = None,
        tenant: str = "default",
        transaction_at: str | None = None,
    ) -> list[dict]: ...

    async def get_relations_for_entity(
        self,
        name: str,
        type: str,  # noqa: A002 -- matches Neo4jClient's (name, type) vocabulary
        tenant: str = "default",
        as_of: str | None = None,
        limit: int = 25,
    ) -> list[dict]: ...

    async def get_all_entities(self, tenant: str = "default") -> list[dict]: ...

    async def get_all_relations(self, tenant: str = "default") -> list[dict]: ...


__all__ = ["GraphBackend"]
