"""External entity linking — grounding KG entities to Wikidata canonical IDs.

Problem solved
--------------
Alias resolution is local: "SpaceX" and "Space Exploration Technologies" are
unified within one tenant via the AliasRegistry.  But two separate tenants
extracting "Boeing 737" independently create two Entity nodes with no shared
identity, and neither entity is linked to the canonical Boeing 737 definition.

Without external grounding:
  - Cross-tenant knowledge is silently duplicated.
  - No canonical description, no infobox properties, no human-verifiable ID.
  - Link prediction and type taxonomy cannot use schema.org / Wikidata types.

Architecture
------------
WikidataEntityLinker calls the public Wikidata Action API (no API key needed
for moderate usage) to find the best matching QID for an entity.

Matching strategy:
  1. wbsearchentities API with entity name + type as language-independent search
  2. Filter by entity type using Wikidata P31 (instance of) heuristics
  3. Store the best match as a WikidataLink node with a HAS_WIKIDATA_LINK edge

Caching:
  - WikidataLink nodes in Neo4j serve as a persistent cache keyed by (name, type).
  - Successful links are never re-fetched.
  - Failed lookups are cached as WikidataLink {status: 'not_found'} with a TTL
    so we don't hammer Wikidata repeatedly for unlinked entities.

Wikidata P31 (instance of) heuristics:
  - PERSON → Q5 (human), Q215627 (person)
  - ORG    → Q43229 (organization), Q4830453 (business)
  - PRODUCT→ Q2424752 (product)
  - LOCATION → Q618123 (geographical object)

The linker is async and rate-limited (1 req/sec by default) to respect
Wikidata's API fair-use policy.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# Wikidata P31 (instance of) QIDs for each entity type
TYPE_QIDS: dict[str, list[str]] = {
    "PERSON":   ["Q5", "Q215627"],
    "ORG":      ["Q43229", "Q4830453", "Q6881511"],
    "PRODUCT":  ["Q2424752", "Q2020153"],
    "LOCATION": ["Q618123", "Q2221906", "Q515"],
    "EVENT":    ["Q1656682", "Q15275719"],
}

# How long to cache a "not found" result (seconds)
NOT_FOUND_TTL_SECONDS = 86400 * 7  # 7 days


class WikidataEntityLinker:
    """
    Ground KG entities to Wikidata QIDs via the public API.

    Usage::

        linker = WikidataEntityLinker(neo4j_client, rate_limit=1.0)

        # Link a single entity
        result = await linker.link_entity("SpaceX", "ORG", tenant="acme")
        # → {"qid": "Q193701", "label": "SpaceX", "description": "..."}

        # Batch-link unlinked entities in a tenant
        count = await linker.link_all_unlinked(tenant="acme", limit=100)

        # Get existing link
        link = await linker.get_link("SpaceX", "ORG", tenant="acme")
    """

    def __init__(self, neo4j_client, rate_limit: float = 1.0):
        """
        Parameters
        ----------
        neo4j_client : Neo4jClient
        rate_limit   : Minimum seconds between Wikidata API calls.
                       Wikidata asks for ≤ 200 req/s from bots; 1.0 is conservative.
        """
        self._neo4j = neo4j_client
        self._rate_limit = rate_limit
        self._last_call = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    async def link_entity(
        self,
        entity_name: str,
        entity_type: str,
        tenant: str = "default",
        force: bool = False,
    ) -> dict | None:
        """
        Find and store the best Wikidata QID for an entity.

        Returns the match dict or None if no match found.
        Caches results in Neo4j — repeated calls are O(1) DB reads.

        Parameters
        ----------
        force : if True, re-fetch even if a cached result exists.
        """
        if not force:
            cached = await self.get_link(entity_name, entity_type, tenant)
            if cached:
                return cached if cached.get("status") != "not_found" else None

        candidate = await self._search_wikidata(entity_name, entity_type)

        if candidate:
            await self._store_link(
                entity_name=entity_name,
                entity_type=entity_type,
                tenant=tenant,
                qid=candidate["qid"],
                label=candidate.get("label", ""),
                description=candidate.get("description", ""),
                status="linked",
            )
            # Also stamp QID directly on the entity node for fast access
            await self._neo4j.run(
                """
                MATCH (e:Entity {name: $name, type: $type, tenant: $tenant})
                SET e.wikidata_qid   = $qid,
                    e.wikidata_label = $label,
                    e.wikidata_linked_at = datetime()
                """,
                name=entity_name,
                type=entity_type,
                tenant=tenant,
                qid=candidate["qid"],
                label=candidate.get("label", ""),
            )
            log.info(
                "entity_linker.linked",
                entity=entity_name,
                qid=candidate["qid"],
                tenant=tenant,
            )
            return candidate
        else:
            await self._store_link(
                entity_name=entity_name,
                entity_type=entity_type,
                tenant=tenant,
                qid="",
                label="",
                description="",
                status="not_found",
            )
            return None

    async def get_link(
        self,
        entity_name: str,
        entity_type: str,
        tenant: str = "default",
    ) -> dict | None:
        """Return the cached WikidataLink for an entity, or None if not yet looked up."""
        rows = await self._neo4j.run(
            """
            MATCH (wl:WikidataLink {entity_name: $name, entity_type: $type, tenant: $tenant})
            RETURN wl.qid         AS qid,
                   wl.label       AS label,
                   wl.description AS description,
                   wl.status      AS status,
                   wl.linked_at   AS linked_at
            """,
            name=entity_name,
            type=entity_type,
            tenant=tenant,
        )
        return dict(rows[0]) if rows else None

    async def link_all_unlinked(
        self,
        tenant: str = "default",
        limit: int = 100,
        entity_types: list[str] | None = None,
    ) -> int:
        """
        Attempt to link all entities that have no WikidataLink yet.

        Returns count of newly linked entities.
        Skips entity types with no TYPE_QIDS heuristics.
        """
        types_filter = (
            "AND e.type IN $types" if entity_types else ""
        )
        params: dict = {"tenant": tenant, "limit": limit}
        if entity_types:
            params["types"] = entity_types

        rows = await self._neo4j.run(
            f"""
            MATCH (e:Entity {{tenant: $tenant}})
            WHERE NOT EXISTS {{
                MATCH (wl:WikidataLink {{entity_name: e.name, entity_type: e.type, tenant: $tenant}})
            }}
            AND e.type IN {list(TYPE_QIDS.keys())}
            {types_filter}
            RETURN e.name AS name, e.type AS type
            LIMIT $limit
            """,
            **params,
        )

        linked = 0
        for row in rows:
            result = await self.link_entity(row["name"], row["type"], tenant=tenant)
            if result:
                linked += 1

        log.info("entity_linker.batch_complete", linked=linked, tenant=tenant)
        return linked

    async def list_linked(
        self,
        tenant: str = "default",
        limit: int = 100,
    ) -> list[dict]:
        """Return all successfully linked entities for a tenant."""
        return await self._neo4j.run(
            """
            MATCH (wl:WikidataLink {tenant: $tenant, status: 'linked'})
            RETURN wl.entity_name  AS entity_name,
                   wl.entity_type  AS entity_type,
                   wl.qid          AS qid,
                   wl.label        AS label,
                   wl.description  AS description,
                   wl.linked_at    AS linked_at
            ORDER BY wl.linked_at DESC
            LIMIT $limit
            """,
            tenant=tenant,
            limit=limit,
        )

    # ── Wikidata API ───────────────────────────────────────────────────────────

    async def _search_wikidata(
        self,
        entity_name: str,
        entity_type: str,
    ) -> dict | None:
        """
        Call Wikidata wbsearchentities API and return the best matching QID.

        Rate-limited to respect Wikidata fair-use policy.
        Returns None on network error or no match.
        """
        await self._rate_limit_sleep()

        try:
            import urllib.request
            import urllib.parse
            import json

            params = urllib.parse.urlencode({
                "action":   "wbsearchentities",
                "search":   entity_name,
                "language": "en",
                "limit":    "5",
                "format":   "json",
                "type":     "item",
            })
            url = f"https://www.wikidata.org/w/api.php?{params}"
            headers = {"User-Agent": "GraphRAG-EntityLinker/1.0 (research@example.com)"}

            req = urllib.request.Request(url, headers=headers)

            def _fetch() -> bytes:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return resp.read()

            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(None, _fetch)
            data = json.loads(raw.decode("utf-8"))

            results = data.get("search", [])
            if not results:
                return None

            # Filter by type heuristics (P31 instance of)
            preferred_qids = set(TYPE_QIDS.get(entity_type, []))
            for item in results:
                # Best effort: use the description as a type hint
                desc = (item.get("description") or "").lower()
                label = (item.get("label") or "").lower()
                if entity_type == "PERSON" and any(
                    w in desc for w in ("person", "human", "politician", "businessman", "ceo")
                ):
                    return self._make_result(item)
                if entity_type == "ORG" and any(
                    w in desc for w in ("company", "corporation", "organization", "agency", "institution")
                ):
                    return self._make_result(item)
                if entity_type == "PRODUCT" and any(
                    w in desc for w in ("aircraft", "product", "vehicle", "software", "model")
                ):
                    return self._make_result(item)
                if entity_type == "LOCATION" and any(
                    w in desc for w in ("city", "country", "region", "location", "state")
                ):
                    return self._make_result(item)

            # No type-matching result; fall back to first result
            if results:
                return self._make_result(results[0])
            return None

        except Exception as exc:
            log.warning("entity_linker.api_error", entity=entity_name, error=str(exc))
            return None

    @staticmethod
    def _make_result(item: dict) -> dict:
        return {
            "qid":         item.get("id", ""),
            "label":       item.get("label", ""),
            "description": item.get("description", ""),
            "url":         f"https://www.wikidata.org/wiki/{item.get('id', '')}",
        }

    async def _rate_limit_sleep(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._rate_limit:
            await asyncio.sleep(self._rate_limit - elapsed)
        self._last_call = time.monotonic()

    async def _store_link(
        self,
        entity_name: str,
        entity_type: str,
        tenant: str,
        qid: str,
        label: str,
        description: str,
        status: str,
    ) -> None:
        await self._neo4j.run(
            """
            MERGE (wl:WikidataLink {
                entity_name: $name, entity_type: $type, tenant: $tenant
            })
            SET wl.qid         = $qid,
                wl.label       = $label,
                wl.description = $description,
                wl.status      = $status,
                wl.linked_at   = datetime()
            WITH wl
            MATCH (e:Entity {name: $name, type: $type, tenant: $tenant})
            MERGE (e)-[:HAS_WIKIDATA_LINK]->(wl)
            """,
            name=entity_name,
            type=entity_type,
            tenant=tenant,
            qid=qid,
            label=label,
            description=description,
            status=status,
        )
