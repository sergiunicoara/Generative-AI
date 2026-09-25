"""Entity type taxonomy — SUBCLASS_OF hierarchy for type-aware retrieval.

Problems solved
---------------
1. Flat type namespace — querying for "Agent" entities requires knowing all
   subtypes (PERSON, ORG) in advance and hard-coding them in every query.
   When new subtypes are added the queries silently miss them.

2. No semantic inheritance — the ontology knows "CEO_OF: (PERSON, ORG)" but
   nothing about the fact that PERSON ⊂ Agent.  A query for Agent entities
   cannot use relation rules defined for PERSON.

3. Type-coercion blindness — when an extractor produces "EXECUTIVE" instead
   of "PERSON", there is no mechanism to say "EXECUTIVE is a PERSON" and
   inherit all PERSON constraints.

Architecture
------------
- EntityType nodes in Neo4j connected by SUBCLASS_OF edges.
- TypeTaxonomy manages the hierarchy and provides transitive-closure helpers.
- Default hierarchy ships with the package; caller can extend it at runtime.
- Tenancy: SUBCLASS_OF edges with no ``tenant`` property are the shared
  platform hierarchy (the defaults below). ``register_subclass(..., tenant=)``
  writes an edge tagged with that tenant, visible only in that tenant's view,
  so one tenant's admin cannot reshape another tenant's type expansion.
- Retrieval integration: expand_type(query_type) returns all subtypes so that
  vector search / BM25 / multi-hop traversal can include subtype entities
  without caller knowledge of the full type tree.

Default hierarchy
-----------------
    Agent
    ├── PERSON
    └── ORG

    Artifact
    └── PRODUCT

    Place
    └── LOCATION

    Temporal
    └── EVENT

    Abstract
    └── CONCEPT
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)

# Canonical default hierarchy expressed as (child, parent) pairs.
# All type strings are UPPER_SNAKE_CASE to match entity.type conventions.
DEFAULT_HIERARCHY: list[tuple[str, str]] = [
    ("PERSON",   "AGENT"),
    ("ORG",      "AGENT"),
    ("PRODUCT",  "ARTIFACT"),
    ("LOCATION", "PLACE"),
    ("EVENT",    "TEMPORAL"),
    ("CONCEPT",  "ABSTRACT"),
]


class TypeTaxonomy:
    """
    Manage entity type hierarchy and provide subtype expansion.

    Usage::

        taxonomy = TypeTaxonomy(neo4j_client)
        await taxonomy.load()

        subtypes = taxonomy.get_subtypes("AGENT")       # ["PERSON", "ORG"]
        ancestors = taxonomy.get_ancestors("PERSON")    # ["AGENT"]
        expanded  = taxonomy.expand_type("AGENT")       # ["AGENT", "PERSON", "ORG"]

        # Add a custom subtype at runtime
        await taxonomy.register_subclass("REGULATOR", parent="ORG")

        # Query expansion: retrieve all entities of type Agent (including subtypes)
        rows = await taxonomy.query_by_type("AGENT", tenant="acme")
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client
        # Platform (shared) adjacency: parent → children, child → parents
        self._children:  dict[str, set[str]] = {}
        self._parents:   dict[str, set[str]] = {}
        # Tenant-scoped extensions layered on top of the platform hierarchy
        self._tenant_children: dict[str, dict[str, set[str]]] = {}
        self._tenant_parents:  dict[str, dict[str, set[str]]] = {}
        self._loaded = False

    # ── Bootstrap ──────────────────────────────────────────────────────────────

    async def load(self, extra_pairs: list[tuple[str, str]] | None = None) -> None:
        """
        Seed Neo4j with default hierarchy and load into memory.

        ``extra_pairs`` allows callers (or config) to extend the hierarchy
        without modifying source code.
        """
        pairs = list(DEFAULT_HIERARCHY) + (extra_pairs or [])

        for child, parent in pairs:
            await self._neo4j.run(
                """
                MERGE (child:EntityType {name: $child})
                MERGE (parent:EntityType {name: $parent})
                MERGE (child)-[:SUBCLASS_OF]->(parent)
                """,
                child=child,
                parent=parent,
            )

        # Pull full hierarchy from Neo4j (includes any previously persisted additions)
        rows = await self._neo4j.run(
            """
            MATCH (child:EntityType)-[r:SUBCLASS_OF]->(parent:EntityType)
            RETURN child.name AS child, parent.name AS parent, r.tenant AS tenant
            """
        )
        self._children.clear()
        self._parents.clear()
        self._tenant_children.clear()
        self._tenant_parents.clear()
        for row in rows:
            self._index(row["child"], row["parent"], row.get("tenant"))

        self._loaded = True
        log.info(
            "type_taxonomy.loaded",
            types=len(self._children) + len(self._parents),
            pairs=len(rows),
        )

    async def register_subclass(
        self, child: str, parent: str, tenant: str | None = None,
    ) -> None:
        """Add a SUBCLASS_OF edge and update the in-memory index.

        With ``tenant`` the edge belongs to that tenant only; without it the
        edge joins the shared platform hierarchy (offline/bootstrap use only --
        never reachable from a tenant-scoped API caller).
        """
        child  = child.upper()
        parent = parent.upper()
        if tenant is None:
            await self._neo4j.run(
                """
                MERGE (c:EntityType {name: $child})
                MERGE (p:EntityType {name: $parent})
                MERGE (c)-[:SUBCLASS_OF]->(p)
                """,
                child=child,
                parent=parent,
            )
        else:
            await self._neo4j.run(
                """
                MERGE (c:EntityType {name: $child})
                MERGE (p:EntityType {name: $parent})
                MERGE (c)-[:SUBCLASS_OF {tenant: $tenant}]->(p)
                """,
                child=child,
                parent=parent,
                tenant=tenant,
            )
        self._index(child, parent, tenant)
        log.info("type_taxonomy.subclass_registered", child=child, parent=parent, tenant=tenant)

    def _index(self, child: str, parent: str, tenant: str | None) -> None:
        if tenant is None:
            children, parents = self._children, self._parents
        else:
            children = self._tenant_children.setdefault(tenant, {})
            parents = self._tenant_parents.setdefault(tenant, {})
        children.setdefault(parent, set()).add(child)
        parents.setdefault(child, set()).add(parent)

    def _children_of(self, node: str, tenant: str | None) -> set[str]:
        """Never mutated by any caller -- may return an internal set by
        reference (no copy) when there's nothing tenant-specific to merge,
        which is the common case during a transitive walk over a taxonomy
        where only a few nodes carry a tenant extension.
        """
        base = self._children.get(node)
        ext = self._tenant_children.get(tenant, {}).get(node) if tenant is not None else None
        if not ext:
            return base if base is not None else set()
        return (base | ext) if base else ext

    def _parents_of(self, node: str, tenant: str | None) -> set[str]:
        """See _children_of -- same no-copy-when-no-extension contract."""
        base = self._parents.get(node)
        ext = self._tenant_parents.get(tenant, {}).get(node) if tenant is not None else None
        if not ext:
            return base if base is not None else set()
        return (base | ext) if base else ext

    # ── Traversal helpers ──────────────────────────────────────────────────────

    def get_subtypes(
        self, type_name: str, transitive: bool = True, tenant: str | None = None,
    ) -> list[str]:
        """
        Return all subtypes of ``type_name``.

        Parameters
        ----------
        transitive : if True (default) return the full transitive closure
                     (all descendants, not just direct children).
        """
        type_name = type_name.upper()
        if transitive:
            return list(self._transitive_children(type_name, tenant))
        return list(self._children_of(type_name, tenant))

    def get_ancestors(
        self, type_name: str, transitive: bool = True, tenant: str | None = None,
    ) -> list[str]:
        """Return all ancestor types of ``type_name``."""
        type_name = type_name.upper()
        if transitive:
            return list(self._transitive_parents(type_name, tenant))
        return list(self._parents_of(type_name, tenant))

    def expand_type(self, type_name: str, tenant: str | None = None) -> list[str]:
        """
        Return ``type_name`` plus all of its transitive subtypes.

        Use this to expand a query type before passing to Neo4j so that
        results include entities of any subtype.

        Example::
            taxonomy.expand_type("AGENT") → ["AGENT", "PERSON", "ORG"]
        """
        type_name = type_name.upper()
        return [type_name] + self.get_subtypes(type_name, tenant=tenant)

    def is_subclass_of(self, child: str, parent: str, tenant: str | None = None) -> bool:
        """Return True if ``child`` is a (transitive) subclass of ``parent``."""
        return parent.upper() in self._transitive_parents(child.upper(), tenant)

    def least_common_ancestor(
        self, type_a: str, type_b: str, tenant: str | None = None,
    ) -> str | None:
        """
        Return the most-specific common ancestor of two types.

        Useful for ontology-guided merge decisions — if "EXECUTIVE" and
        "MANAGER" both reduce to "PERSON" you can safely merge without
        type clash.
        """
        a_ancestors = {type_a.upper()} | self._transitive_parents(type_a.upper(), tenant)
        b_ancestors = {type_b.upper()} | self._transitive_parents(type_b.upper(), tenant)
        common = a_ancestors & b_ancestors
        if not common:
            return None
        # Prefer the most-specific: pick the one with the most ancestors
        return max(common, key=lambda t: len(self._transitive_parents(t, tenant)))

    # ── Neo4j query integration ────────────────────────────────────────────────

    async def query_by_type(
        self,
        type_name: str,
        tenant: str = "default",
        include_subtypes: bool = True,
        limit: int = 100,
    ) -> list[dict]:
        """
        Fetch entities of ``type_name`` (and subtypes if ``include_subtypes``).

        This is the primary retrieval integration point — call this instead
        of querying by a single entity type so that the full type hierarchy
        is respected.
        """
        types = (
            self.expand_type(type_name, tenant=tenant) if include_subtypes else [type_name.upper()]
        )
        return await self._neo4j.run(
            """
            UNWIND $types AS t
            MATCH (e:Entity {type: t})
            WHERE (e.tenant = $tenant)
              AND coalesce(e.quarantined, false) = false
            RETURN DISTINCT e.name        AS name,
                            e.type        AS type,
                            e.description AS description,
                            e.tenant      AS tenant
            LIMIT $limit
            """,
            types=types,
            tenant=tenant,
            limit=limit,
        )

    async def get_schema(self, tenant: str) -> list[dict]:
        """Return the platform hierarchy plus ``tenant``'s own extensions."""
        return await self._neo4j.run(
            """
            MATCH (child:EntityType)-[r:SUBCLASS_OF]->(parent:EntityType)
            WHERE r.tenant IS NULL OR r.tenant = $tenant
            RETURN child.name AS child, parent.name AS parent,
                   CASE WHEN r.tenant IS NULL THEN 'platform' ELSE 'tenant' END AS scope
            ORDER BY parent.name, child.name
            """,
            tenant=tenant,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _transitive_children(self, root: str, tenant: str | None = None) -> set[str]:
        visited: set[str] = set()
        stack = list(self._children_of(root, tenant))
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(self._children_of(node, tenant))
        return visited

    def _transitive_parents(self, node: str, tenant: str | None = None) -> set[str]:
        visited: set[str] = set()
        stack = list(self._parents_of(node, tenant))
        while stack:
            p = stack.pop()
            if p not in visited:
                visited.add(p)
                stack.extend(self._parents_of(p, tenant))
        return visited


# ── Module-level singleton ─────────────────────────────────────────────────────

_taxonomy: TypeTaxonomy | None = None


def get_type_taxonomy(neo4j_client=None) -> TypeTaxonomy:
    global _taxonomy
    if _taxonomy is None:
        if neo4j_client is None:
            from graphrag.graph.neo4j_client import get_neo4j
            neo4j_client = get_neo4j()
        _taxonomy = TypeTaxonomy(neo4j_client)
    return _taxonomy
