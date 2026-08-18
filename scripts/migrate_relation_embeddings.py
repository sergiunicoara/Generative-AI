"""Migrate RelationEmbedding nodes to tenant-scoped identity (F13).

Problem
-------
RelationEmbedding nodes used to be keyed on relation name alone:
``MERGE (re:RelationEmbedding {relation: $rel})``. Relation names are shared
vocabulary across tenants (e.g. "SUPERSEDES"), so one tenant's TransE training
run silently overwrote the vector every other tenant's link prediction read.

The nodes come from two sources with very different ownership:

  - ``source = 'derived'`` — a pure function of the relation NAME
    (``_derive_relation_embedding``: SHA-256 hash of the name seeds a fixed
    RNG draw). Identical for every tenant by construction. Safe to keep as one
    shared node.
  - ``source = 'trained'`` — TransE fitted to ONE tenant's edges. Genuinely
    tenant-specific, and this is the actual leak.

Migration strategy
-------------------
Existing ``trained`` nodes carry no record of which tenant trained them —
there was never a tenant property to read. Guessing an owner from context
(e.g. "the only tenant with edges for that relation") would be attributing
data to a tenant based on inference, not fact, which is worse than admitting
the provenance is gone.

So this migration DELETES existing ``trained`` nodes rather than migrate them.
Nothing breaks: the code already falls back to the deterministic derived
vector when no stored embedding exists, so deleting a trained node degrades
retrieval to the documented fallback, not an error. Each tenant can re-run
``POST /kg/edge-embeddings/train`` to regenerate its own, now correctly
tenant-scoped, trained embeddings.

Existing ``derived`` nodes are NOT deleted — they're correct as shared nodes.
They're backfilled with ``tenant = '__derived__'`` (edge_embeddings.DERIVED_SCOPE)
so the new tenant-aware read/write queries recognize them; a node already
carrying that value is left untouched (idempotent).

Usage::

    python scripts/migrate_relation_embeddings.py            # dry run (default)
    python scripts/migrate_relation_embeddings.py --apply     # execute
"""

from __future__ import annotations

import argparse
import asyncio

from graphrag.graph.edge_embeddings import DERIVED_SCOPE
from graphrag.graph.neo4j_client import get_neo4j


async def migrate(apply: bool) -> None:
    neo4j = get_neo4j()

    counts = await neo4j.run(
        """
        MATCH (re:RelationEmbedding)
        RETURN re.source AS source,
               re.tenant IS NOT NULL AS already_scoped,
               count(re) AS n
        """
    )
    print("Current RelationEmbedding nodes:")
    for row in counts:
        print(f"  source={row['source']!r} already_scoped={row['already_scoped']} n={row['n']}")

    if not counts:
        print("Nothing to migrate.")
        return

    if not apply:
        print(
            "\nDry run only. Re-run with --apply to:\n"
            "  1. DELETE every trained node (unattributable provenance --"
            " falls back to the derived vector until re-trained per-tenant)\n"
            f"  2. SET tenant = {DERIVED_SCOPE!r} on every unscoped derived node"
        )
        return

    deleted = await neo4j.run(
        """
        MATCH (re:RelationEmbedding {source: 'trained'})
        WHERE re.tenant IS NULL
        DETACH DELETE re
        RETURN count(re) AS n
        """
    )
    print(f"Deleted {deleted[0]['n'] if deleted else 0} unattributed trained node(s).")

    backfilled = await neo4j.run(
        """
        MATCH (re:RelationEmbedding {source: 'derived'})
        WHERE re.tenant IS NULL
        SET re.tenant = $derived
        RETURN count(re) AS n
        """,
        derived=DERIVED_SCOPE,
    )
    print(f"Backfilled {backfilled[0]['n'] if backfilled else 0} derived node(s) with tenant={DERIVED_SCOPE!r}.")

    remaining = await neo4j.run(
        "MATCH (re:RelationEmbedding) WHERE re.tenant IS NULL RETURN count(re) AS n"
    )
    left = remaining[0]["n"] if remaining else 0
    if left:
        print(f"WARNING: {left} RelationEmbedding node(s) still have no tenant — investigate before relying on this migration.")
    else:
        print("Done — every remaining RelationEmbedding node carries a tenant.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    asyncio.run(migrate(parser.parse_args().apply))
