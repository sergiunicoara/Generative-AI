#!/usr/bin/env python
"""entity_type_migration.py — Cascade-rename an entity type in the graph.

Usage
-----
    # Dry-run: see how many entities would be affected
    python scripts/entity_type_migration.py --old-type EXEC --new-type PERSON --dry-run

    # Execute rename for a specific tenant
    python scripts/entity_type_migration.py --old-type EXEC --new-type PERSON --tenant acme

    # Rename across all tenants (default only)
    python scripts/entity_type_migration.py --old-type EXECUTIVE --new-type PERSON

What it does
------------
Delegates to OntologyRegistry.rename_entity_type() which cascades the rename to:
  1. Entity nodes          (Entity.type)
  2. WikidataLink nodes    (WikidataLink.entity_type)
  3. RELATES_TO edges      (edge.src_type / edge.tgt_type metadata)
  4. Statement nodes       (stmt.src_type / stmt.tgt_type / stmt.subject_type)
  5. OntologyVersion       (adds the new type, creates OntologyMigration audit node)

The rename is idempotent: running it twice for the same (old, new) pair is safe.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main(args: argparse.Namespace) -> None:
    from graphrag.graph.neo4j_client import get_neo4j
    from graphrag.graph.ontology_registry import get_ontology_registry

    neo4j    = get_neo4j()
    registry = get_ontology_registry(neo4j_client=neo4j)

    # Load registry so it has allowed_types initialised
    try:
        from graphrag.core.config import get_settings
        cfg          = get_settings()
        entity_types = getattr(cfg, "entity_types", None) or [
            "PERSON", "ORG", "PRODUCT", "LOCATION", "EVENT", "CONCEPT",
        ]
        await registry.load(entity_types=entity_types)
    except Exception as exc:
        print(f"[entity_type_migration] WARNING: Could not load registry: {exc}",
              file=sys.stderr)

    old_type = args.old_type.strip().upper()
    new_type = args.new_type.strip().upper()
    tenant   = args.tenant
    dry_run  = args.dry_run

    print(
        f"[entity_type_migration] "
        f"{'DRY RUN — ' if dry_run else ''}"
        f"Renaming '{old_type}' → '{new_type}'  tenant={tenant!r}"
    )

    report = await registry.rename_entity_type(
        old_type=old_type,
        new_type=new_type,
        tenant=tenant,
        dry_run=dry_run,
    )

    if dry_run:
        print(
            f"[entity_type_migration] Would rename "
            f"{report.get('entities_renamed', 0)} entities."
        )
        print("[entity_type_migration] Pass without --dry-run to apply.")
    else:
        status = report.get("status", "unknown")
        print(
            f"[entity_type_migration] Status: {status}  |  "
            f"entities renamed: {report.get('entities_renamed', 0)}  |  "
            f"WikidataLinks updated: {report.get('wikidata_updated', 0)}"
        )
        if status == "complete":
            print("[entity_type_migration] Migration applied successfully.")
        elif status == "no_entities":
            print("[entity_type_migration] No entities found with that type — nothing to do.")
        elif status == "no_op":
            print("[entity_type_migration] Old and new types are identical — no action taken.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cascade-rename an entity type across the GraphRAG knowledge graph."
    )
    parser.add_argument(
        "--old-type", required=True, dest="old_type",
        help="Current entity type to rename (e.g. EXEC)",
    )
    parser.add_argument(
        "--new-type", required=True, dest="new_type",
        help="Target entity type (e.g. PERSON)",
    )
    parser.add_argument(
        "--tenant", default="default",
        help="Tenant scope for the rename (default: 'default')",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Count affected entities without making changes",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
