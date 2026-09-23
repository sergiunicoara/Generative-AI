"""Gremlin (Apache TinkerPop) implementation of ``GraphBackend`` --
Neptune / Cosmos DB Gremlin API as an alternative to ``Neo4jClient``.

Scope
-----
Implements exactly the 9 methods ``graph_backend.GraphBackend`` declares:
``close``, ``entity_exists``, ``merge_entity``, ``merge_mentions``,
``merge_relation``, ``get_entity_neighbors``, ``get_relations_for_entity``,
``get_all_entities``, ``get_all_relations``. See ``graph_backend.py``'s
module docstring for the ~50 ``Neo4jClient`` methods this deliberately does
NOT cover (schema/admin, batch variants, vector/BM25 search, PageRank/GDS,
community detection, ...) -- no Gremlin equivalent exists for any of them.

This module does not touch ``get_neo4j()`` or any of its ~100 call sites --
the platform's actual runtime backend is unchanged. See ``docs/roadmap.md``,
which scopes Gremlin/Neptune as "interoperability targets, not additional
sources of truth."

Verification status -- read this before trusting a translation
------------------------------------------------------------------
Every traversal below is unit-tested against a mocked ``_submit`` boundary
(the query-building logic runs for real; only the network call is faked --
see ``tests/unit/test_gremlin_client.py``, mirroring how
``tests/unit/test_neo4j_client_embeddings.py`` mocks ``Neo4jClient.run``).
**None of it has ever been executed against a live Neptune, Cosmos DB, or
Gremlin Server instance** -- no such infrastructure is available in this
environment (the same Docker-unavailable constraint already documented for
Neo4j/Redis/RabbitMQ in the audit backlog). Treat this as a documented,
carefully-reasoned starting point, not a verified claim -- the exact status
``triplestore.py`` already gives Stardog, RDFox, and Virtuoso.

Known, documented behavioral differences from ``Neo4jClient``
------------------------------------------------------------------
- ``merge_entity`` and ``merge_relation`` are each a find-or-create traversal
  followed by a *separate* conditional-refresh traversal, not one atomic
  statement the way Cypher's ``MERGE ... ON CREATE ... ON MATCH`` is. There
  is a small window between the two where a concurrent writer could race.
  Cypher's single-statement MERGE has no such window.
- ``get_entity_neighbors`` / ``get_relations_for_entity`` reject (raise
  ``NotImplementedError``) rather than silently ignore ``as_of`` /
  ``transaction_at`` -- their Cypher versions filter edges by temporal
  validity; that filter has no translation here that was ever checked
  against a real store, so returning unfiltered rows under a temporal
  parameter would silently be wrong.
- ``merge_relation`` does not port ``Neo4jClient.merge_relation``'s second,
  conditional write of chunk-span/extraction-model provenance (its own
  ``if rel.chunk_span_start is not None or rel.extraction_model:`` branch)
  -- out of scope for this core subset, not silently dropped.
- ``source_doc_ids`` (a native Cypher list property on the Neo4j edge) is
  stored as a JSON string property (``source_doc_ids_json``) here, the same
  pattern this codebase already uses for ``Document.metadata_envelope_json``
  in ``neo4j_client.py`` ("node properties cannot hold a nested map/list
  portably") -- TinkerPop edge properties are single-cardinality scalars,
  not natively list-valued, so this is the same honest workaround, not a
  new invention.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import GraphTraversal, GraphTraversalSource, __
from gremlin_python.process.traversal import Order, P

from graphrag.core.connector_url_safety import assert_safe_connector_url
from graphrag.core.models import Entity, Relation

_GREMLIN_SCHEMES = frozenset({"ws", "wss"})


class GremlinBackend:
    """See module docstring. Implements ``graph_backend.GraphBackend``."""

    def __init__(
        self, url: str, *, username: str | None = None, password: str | None = None,
    ) -> None:
        assert_safe_connector_url(url, context="GremlinBackend url", schemes=_GREMLIN_SCHEMES)
        auth_kwargs: dict[str, str] = {}
        if username is not None:
            auth_kwargs["username"] = username
        if password is not None:
            auth_kwargs["password"] = password
        self._connection = DriverRemoteConnection(url, "g", **auth_kwargs)
        self._g: GraphTraversalSource = traversal().with_(self._connection)

    async def close(self) -> None:
        await asyncio.to_thread(self._connection.close)

    async def _submit(
        self, build: Callable[[GraphTraversalSource], GraphTraversal],
    ) -> list[Any]:
        """The sole traversal-execution boundary every method below funnels
        through -- deliberately mirrors ``Neo4jClient.run()`` being the one
        thing ``test_neo4j_client_embeddings.py`` mocks. ``build`` receives
        ``self._g`` and returns a ready-to-submit traversal; tests replace
        this whole method with an ``AsyncMock`` so ``build`` is constructed
        but never called, and the real query-building logic in each public
        method still runs for real against the mocked return value.
        """
        t = build(self._g)
        return await asyncio.to_thread(t.toList)

    # ── Entity CRUD ──────────────────────────────────────────────────────

    async def entity_exists(
        self, name: str, entity_type: str, tenant: str = "default",
    ) -> bool:
        rows = await self._submit(
            lambda g: g.V().hasLabel("Entity")
            .has("name", name).has("type", entity_type).has("tenant", tenant)
            .count()
        )
        return bool(rows and rows[0] > 0)

    async def merge_entity(self, entity: Entity, tenant: str = "default") -> None:
        now = datetime.now(timezone.utc).isoformat()
        source_type = (
            entity.source_type if isinstance(entity.source_type, str) else entity.source_type.value
        )
        rows = await self._submit(
            lambda g: g.V().hasLabel("Entity")
            .has("name", entity.name).has("type", entity.type).has("tenant", tenant)
            .fold()
            .coalesce(
                __.unfold(),
                __.addV("Entity")
                .property("name", entity.name)
                .property("type", entity.type)
                .property("tenant", tenant)
                .property("id", entity.id)
                .property("description", entity.description)
                .property("embedding_json", json.dumps(entity.embedding))
                .property("source_type", source_type)
                .property("source_doc_id", entity.source_doc_id)
                .property("extraction_model", entity.extraction_model)
                .property("prompt_version", entity.prompt_version)
                .property("resolution_status", entity.resolution_status)
                .property("resolution_method", entity.resolution_method)
                .property("created_at", now)
                .property("recorded_at", now),
            )
            .project("id", "description", "embedding_json")
            .by("id").by("description").by("embedding_json")
        )
        if not rows:
            return
        row = rows[0]
        if row["id"] == entity.id:
            return  # just created -- ON-CREATE fields above already cover it
        # Pre-existing vertex: refresh mirrors Neo4jClient.merge_entity's
        # ON MATCH semantics (description only if currently empty; embedding
        # only if the incoming one is non-empty). See module docstring for
        # the non-atomicity this two-step approach carries vs. Cypher MERGE.
        new_description = entity.description if row["description"] == "" else row["description"]
        new_embedding_json = json.dumps(entity.embedding) if entity.embedding else row["embedding_json"]
        await self._submit(
            lambda g: g.V().hasLabel("Entity")
            .has("name", entity.name).has("type", entity.type).has("tenant", tenant)
            .property("description", new_description)
            .property("embedding_json", new_embedding_json)
            .property("updated_at", now)
        )

    async def merge_mentions(
        self, chunk_id: str, entity_name: str, entity_type: str, tenant: str = "default",
    ) -> None:
        await self._submit(
            lambda g: g.V().hasLabel("Chunk").has("id", chunk_id).has("tenant", tenant).as_("c")
            .V().hasLabel("Entity").has("name", entity_name).has("type", entity_type).has("tenant", tenant).as_("e")
            .coalesce(
                __.select("c").outE("MENTIONS").where(__.inV().where(P.eq("e"))),
                __.addE("MENTIONS").from_("c").to("e"),
            )
        )

    # ── Relation CRUD ────────────────────────────────────────────────────

    async def merge_relation(
        self,
        rel: Relation,
        src_name: str,
        src_type: str,
        tgt_name: str,
        tgt_type: str,
        tenant: str = "default",
    ) -> None:
        source_type = rel.source_type if isinstance(rel.source_type, str) else rel.source_type.value
        constraint_type = (
            rel.constraint_type if isinstance(rel.constraint_type, str) else rel.constraint_type.value
        )
        existing = await self._submit(
            lambda g: g.V().hasLabel("Entity").has("name", src_name).has("type", src_type).has("tenant", tenant)
            .outE("RELATES_TO").has("relation", rel.relation)
            .where(__.inV().has("name", tgt_name).has("type", tgt_type).has("tenant", tenant))
            .project("confidence", "source_doc_ids_json")
            .by(__.coalesce(__.values("confidence"), __.constant(None)))
            .by(__.coalesce(__.values("source_doc_ids_json"), __.constant("[]")))
        )
        prior_confidence = existing[0]["confidence"] if existing else None
        prior_docs: list[str] = json.loads(existing[0]["source_doc_ids_json"]) if existing else []

        # Bayesian accumulation, same guard against double-counting a
        # repeat ingest of the same document as Neo4jClient.merge_relation.
        if rel.source_doc_id in prior_docs:
            new_docs = prior_docs
            new_confidence = prior_confidence if prior_confidence is not None else rel.confidence
        else:
            new_docs = [*prior_docs, rel.source_doc_id]
            new_confidence = (
                rel.confidence if prior_confidence is None
                else 1.0 - (1.0 - prior_confidence) * (1.0 - rel.confidence)
            )

        now = datetime.now(timezone.utc).isoformat()
        await self._submit(
            lambda g: g.V().hasLabel("Entity").has("name", src_name).has("type", src_type).has("tenant", tenant).as_("s")
            .V().hasLabel("Entity").has("name", tgt_name).has("type", tgt_type).has("tenant", tenant).as_("t")
            .coalesce(
                __.select("s").outE("RELATES_TO").has("relation", rel.relation)
                .where(__.inV().where(P.eq("t"))),
                __.addE("RELATES_TO").from_("s").to("t")
                .property("relation", rel.relation)
                .property("recorded_at", now),
            )
            .property("weight", rel.weight)
            .property("extracted_at", rel.extracted_at.isoformat())
            .property("source_doc_id", rel.source_doc_id)
            .property("source_type", source_type)
            .property("constraint_type", constraint_type)
            .property("confidence_state", rel.confidence_state)
            .property("valid_from", rel.valid_from.isoformat() if rel.valid_from else "")
            .property("valid_to", rel.valid_to.isoformat() if rel.valid_to else "")
            .property("tenant", tenant)
            .property("source_doc_ids_json", json.dumps(new_docs))
            .property("confidence", new_confidence)
        )

    # ── 1-hop traversal ──────────────────────────────────────────────────

    async def get_entity_neighbors(
        self,
        chunk_ids: list[str],
        as_of: str | None = None,
        tenant: str = "default",
        transaction_at: str | None = None,
    ) -> list[dict]:
        if as_of is not None or transaction_at is not None:
            raise NotImplementedError(
                "GremlinBackend.get_entity_neighbors: as_of/transaction_at "
                "temporal filtering has no verified Gremlin translation -- "
                "see this module's docstring."
            )
        return await self._submit(
            lambda g: g.V().hasLabel("Chunk").has("id", P.within(chunk_ids))
            .out("MENTIONS").hasLabel("Entity")
            .not_(__.has("quarantined", True))
            .project("entity", "type", "description", "neighbors")
            .by("name").by("type").by("description")
            .by(
                __.both("RELATES_TO").has("tenant", tenant)
                .not_(__.has("quarantined", True))
                .values("name").dedup().fold()
            )
        )

    async def get_relations_for_entity(
        self,
        name: str,
        type: str,  # noqa: A002 -- matches Neo4jClient's (name, type) vocabulary
        tenant: str = "default",
        as_of: str | None = None,
        limit: int = 25,
    ) -> list[dict]:
        if as_of is not None:
            raise NotImplementedError(
                "GremlinBackend.get_relations_for_entity: as_of temporal "
                "filtering has no verified Gremlin translation -- see this "
                "module's docstring."
            )
        return await self._submit(
            lambda g: g.V().hasLabel("Entity").has("name", name).has("type", type).has("tenant", tenant)
            .not_(__.has("quarantined", True)).as_("e")
            .bothE("RELATES_TO")
            .where(__.otherV().not_(__.has("quarantined", True)))
            .order().by("confidence", Order.desc)
            .limit(limit)
            .project("name", "type", "weight", "confidence", "extracted_at", "source_doc_id", "direction")
            .by(__.otherV().values("name"))
            .by(__.otherV().values("type"))
            .by(__.coalesce(__.values("weight"), __.constant(None)))
            .by(__.coalesce(__.values("confidence"), __.constant(1.0)))
            .by(__.coalesce(__.values("extracted_at"), __.constant(None)))
            .by(__.coalesce(__.values("source_doc_id"), __.constant(None)))
            .by(__.choose(__.outV().where(P.eq("e")), __.constant("outgoing"), __.constant("incoming")))
        )

    # ── Bulk read ────────────────────────────────────────────────────────

    async def get_all_entities(self, tenant: str = "default") -> list[dict]:
        # Faithfully chunk-anchored (Chunk-MENTIONS->Entity), matching
        # Neo4jClient.get_all_entities's actual contract -- not a naive
        # vertex scan. See graph_backend.py's docstring on why.
        return await self._submit(
            lambda g: g.V().hasLabel("Chunk").has("tenant", tenant)
            .out("MENTIONS").hasLabel("Entity")
            .not_(__.has("quarantined", True))
            .dedup()
            .project("id", "name", "type")
            .by("id").by("name").by("type")
        )

    async def get_all_relations(self, tenant: str = "default") -> list[dict]:
        return await self._submit(
            lambda g: g.E().hasLabel("RELATES_TO").has("tenant", tenant)
            .where(__.outV().not_(__.has("quarantined", True)))
            .where(__.inV().not_(__.has("quarantined", True)))
            .project("source_id", "target_id", "relation", "weight")
            .by(__.outV().values("id"))
            .by(__.inV().values("id"))
            .by("relation")
            .by(__.coalesce(__.values("weight"), __.values("confidence"), __.constant(1.0)))
        )


__all__ = ["GremlinBackend"]
