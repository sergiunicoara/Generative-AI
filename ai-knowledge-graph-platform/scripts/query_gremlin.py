"""Smoke-check a live Gremlin endpoint (Neptune, Cosmos DB, or any TinkerPop
Gremlin Server) through GremlinBackend -- the first real, non-test caller of
graphrag/graph/gremlin_client.py.

Neo4j remains the source of truth (see docs/adr/0001-property-graph-over-
triple-store.md's Cypher-side counterpart to this same principle); this
script does not migrate or dual-write anything. It exists to let an
operator confirm a configured Gremlin endpoint is actually reachable and
returns real data through this codebase's own GraphBackend abstraction --
the same role load_blazegraph.py plays for a SPARQL vendor, adapted to a
read-only smoke check since Gremlin write semantics here are for entity/
relation ingestion, not a bulk file load.

Usage
-----
  docker run -d -p 8182:8182 tinkerpop/gremlin-server:3.7.2
  python scripts/query_gremlin.py --url ws://localhost:8182/gremlin --tenant acme

  # or via GREMLIN_URL env var (see gremlin_client.gremlin_source_from_env):
  GREMLIN_URL=ws://localhost:8182/gremlin python scripts/query_gremlin.py --tenant acme
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import structlog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.graph.gremlin_client import GremlinBackend  # noqa: E402  # ROOT must be added to sys.path first.

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

log = structlog.get_logger(__name__)

DEFAULT_URL = "ws://localhost:8182/gremlin"


async def run(url: str, tenant: str = "default") -> dict[str, int]:
    """Connect to ``url`` and run three real GraphBackend reads for
    ``tenant``. Returns entity/relation counts -- raises on connection or
    protocol failure rather than swallowing it, since the whole point is to
    prove the endpoint actually works.
    """
    backend = GremlinBackend(url)
    try:
        entities = await backend.get_all_entities(tenant=tenant)
        relations = await backend.get_all_relations(tenant=tenant)
        sample_exists = (
            await backend.entity_exists(entities[0]["name"], entities[0]["type"], tenant=tenant)
            if entities else None
        )
        log.info(
            "query_gremlin.checked",
            url=url, tenant=tenant,
            entity_count=len(entities), relation_count=len(relations),
        )
        return {
            "entity_count": len(entities),
            "relation_count": len(relations),
            "sample_entity_exists": sample_exists,
        }
    finally:
        await backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-check a live Gremlin endpoint through GremlinBackend"
    )
    parser.add_argument("--url", default=os.getenv("GREMLIN_URL", "").strip() or DEFAULT_URL,
                         help=f"Gremlin Server ws(s):// URL (default: {DEFAULT_URL}, "
                              "or GREMLIN_URL env var)")
    parser.add_argument("--tenant", default="default",
                         help="Tenant to query (default: default)")
    args = parser.parse_args()

    result = asyncio.run(run(args.url, tenant=args.tenant))
    print(f"[OK] {args.url} (tenant={args.tenant}): "
          f"{result['entity_count']} entities, {result['relation_count']} relations, "
          f"sample_entity_exists={result['sample_entity_exists']}")


if __name__ == "__main__":
    main()
