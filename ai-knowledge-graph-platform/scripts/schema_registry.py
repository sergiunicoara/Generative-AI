"""Operate the versioned schema registry (docs/schema-registry.md).

  python scripts/schema_registry.py show      --tenant acme
  python scripts/schema_registry.py drift     --tenant acme
  python scripts/schema_registry.py activate  --tenant acme            # register + activate what the files define now
  python scripts/schema_registry.py activate  --tenant acme --version-id <id>   # reactivate a retained version
  python scripts/schema_registry.py rollback  --tenant acme
  python scripts/schema_registry.py deactivate --tenant acme

The tenant is always explicit and every query is scoped to it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from graphrag.core.config import get_settings
from graphrag.graph.neo4j_client import get_neo4j
from graphrag.graph.ontology_registry import OntologyRegistry
from graphrag.graph.schema_registry import SchemaRegistry, dataset_id_for


async def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["show", "drift", "activate", "rollback", "deactivate"])
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--dataset", default=None, help="defaults to the tenant's configured dataset")
    ap.add_argument("--version-id", default=None)
    args = ap.parse_args(argv)

    cfg = get_settings()
    neo4j = get_neo4j()
    registry = SchemaRegistry(neo4j)
    dataset_id = args.dataset or dataset_id_for(args.tenant, cfg.ontology or {})
    types = list(cfg.ingestion.get("entity_types", ["PERSON", "ORG", "PRODUCT", "CONCEPT", "LOCATION", "EVENT"]))

    if args.command == "show":
        out = await registry.versions(args.tenant, dataset_id)
    elif args.command == "drift":
        out = await OntologyRegistry(neo4j, tenant=args.tenant).check_schema_drift(types)
    elif args.command == "activate" and args.version_id:
        out = await registry.activate(args.tenant, dataset_id, args.version_id)
    elif args.command == "activate":
        reg = OntologyRegistry(neo4j, tenant=args.tenant)
        doc, uri, onto_cfg = reg._load_definitions(types)
        identity = reg._build_identity(doc, uri, onto_cfg)
        out = {**await registry.register_and_activate(identity), "label": identity.label}
    elif args.command == "rollback":
        out = await registry.rollback(args.tenant, dataset_id)
    else:
        out = {"deactivated": await registry.deactivate(args.tenant, dataset_id)}
    print(json.dumps(out, indent=2, default=str))
    return 1 if args.command == "drift" and out.get("status") == "drift" else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
