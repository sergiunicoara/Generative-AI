"""Export business definitions to Apache Atlas and relational builds to OpenLineage.

    python scripts/export_catalog.py atlas --pack config/knowledge_packs/<domain>.yml --tenant <t>
    python scripts/export_catalog.py atlas ... --push          # needs ATLAS_BASE_URL, ATLAS_USERNAME,
                                                               # ATLAS_PASSWORD, ATLAS_GLOSSARY_GUID
    python scripts/export_catalog.py openlineage --tenant <t> --source-id <s>
    python scripts/export_catalog.py openlineage ... --push    # needs OPENLINEAGE_URL (optional OPENLINEAGE_API_KEY)

Without --push the payloads are printed only (dry run). The Atlas export uses
the access of the --sub/--group reader: restricted definitions are left out
unless that reader may see them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.context_graph.knowledge_release import load_pack  # noqa: E402
from graphrag.enterprise.catalog_export import (  # noqa: E402
    AtlasClient, AtlasConfig, OpenLineageClient, atlas_terms, run_event,
)
from graphrag.enterprise.models import AccessContext  # noqa: E402
from graphrag.ingestion.build_ledger import BuildRecord  # noqa: E402


async def _atlas(args) -> int:
    pack = load_pack(args.pack)
    config = AtlasConfig.from_env()
    access = AccessContext.from_claims({"sub": args.sub, "groups": args.group or []})
    terms = atlas_terms(pack.definitions, tenant=args.tenant, access=access,
                        glossary_guid=config.glossary_guid if config else "DRY-RUN")
    if not args.push:
        print(json.dumps(terms, indent=2, default=str))
        return 0
    if config is None:
        print("Atlas is not configured (ATLAS_* environment variables)", file=sys.stderr)
        return 2
    print(json.dumps(await AtlasClient(config).upsert_terms(terms), indent=2))
    return 0


async def _openlineage(args) -> int:
    from graphrag.graph.neo4j_client import get_neo4j

    rows = await get_neo4j().run(
        """
        MATCH (b:IngestionBuild {tenant: $tenant, source_id: $source_id})
        RETURN b {.*} AS build ORDER BY b.started_at DESC LIMIT $limit
        """, tenant=args.tenant, source_id=args.source_id, limit=args.limit)
    events = []
    for row in rows or []:
        b = dict(row["build"])
        build = BuildRecord(tenant=b["tenant"], source_id=b["source_id"], source_uri=b.get("source_uri", ""),
                            mapping_id=b.get("mapping_id", ""), mapping_version=b.get("mapping_version", ""),
                            code_version=b.get("code_version", ""), build_id=b["id"],
                            started_at=b.get("started_at", ""), finished_at=b.get("finished_at", ""),
                            inputs=json.loads(b.get("inputs_json") or "[]"),
                            expectations=json.loads(b.get("expectations_json") or "[]"),
                            outcome=b.get("outcome", ""), outputs=json.loads(b.get("outputs_json") or "{}"),
                            error=b.get("error", ""))
        events.append(run_event(build))
    if not args.push:
        print(json.dumps(events, indent=2, default=str))
        return 0
    client = OpenLineageClient.from_env()
    if client is None:
        print("OpenLineage is not configured (OPENLINEAGE_URL)", file=sys.stderr)
        return 2
    for event in events:
        await client.emit(event)
    print(f"emitted {len(events)} run event(s)")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="target", required=True)
    atlas = sub.add_parser("atlas")
    atlas.add_argument("--pack", type=Path, required=True)
    atlas.add_argument("--tenant", required=True)
    atlas.add_argument("--sub", default="catalog-export")
    atlas.add_argument("--group", action="append")
    atlas.add_argument("--push", action="store_true")
    lineage = sub.add_parser("openlineage")
    lineage.add_argument("--tenant", required=True)
    lineage.add_argument("--source-id", required=True)
    lineage.add_argument("--limit", type=int, default=20)
    lineage.add_argument("--push", action="store_true")
    args = ap.parse_args(argv)
    return asyncio.run(_atlas(args) if args.target == "atlas" else _openlineage(args))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
