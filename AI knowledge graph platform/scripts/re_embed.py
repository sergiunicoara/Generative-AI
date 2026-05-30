#!/usr/bin/env python
"""re_embed.py â€” Re-embed entities whose embedding model/version is stale.

Usage
-----
    python scripts/re_embed.py --tenant acme --model text-embedding-004
    python scripts/re_embed.py --tenant acme --model text-embedding-004 --dry-run
    python scripts/re_embed.py --tenant acme --batch-size 64 --limit 5000

Options
-------
  --tenant       Tenant to process (default: "default")
  --model        New embedding model name (default: from config)
  --version      Model version tag (default: "latest")
  --batch-size   Entities per batch (default: 128)
  --limit        Max entities to queue per run (default: 10000)
  --dry-run      Queue stale entities but do NOT re-embed; print count and exit
  --force        Re-embed ALL entities, even non-stale ones
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running from repo root without pip install
sys.path.insert(0, str(Path(__file__).parent.parent))


async def main(args: argparse.Namespace) -> None:
    from graphrag.core.config import get_settings
    from graphrag.graph.neo4j_client import get_neo4j
    from graphrag.graph.embedding_registry import EmbeddingRegistry

    cfg = get_settings()
    neo4j = get_neo4j()
    registry = EmbeddingRegistry(neo4j)

    model   = args.model or getattr(cfg, "embedding_model", "text-embedding-004")
    version = args.version or getattr(cfg, "embedding_version", "latest")
    tenant  = args.tenant

    print(f"[re_embed] tenant={tenant!r}  model={model!r}  version={version!r}")

    # â”€â”€ Compatibility check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    expected_dim = getattr(cfg, "embedding_dim", 768)
    compat = await registry.check_compatibility(
        current_model=model,
        current_version=version,
        expected_dim=expected_dim,
        tenant=tenant,
    )
    if compat.get("blocking"):
        print(f"[re_embed] ERROR: {compat.get('reason')}", file=sys.stderr)
        sys.exit(1)
    if compat.get("action_required"):
        print(f"[re_embed] WARNING: {compat.get('reason')}")

    # â”€â”€ Queue stale entities â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    queued = await registry.queue_re_embed(
        model=model,
        version=version,
        tenant=tenant,
        limit=args.limit,
        force=args.force,
    )
    print(f"[re_embed] Entities queued for re-embedding: {queued}")

    if args.dry_run:
        print("[re_embed] --dry-run: no embeddings written.")
        return

    if queued == 0:
        print("[re_embed] Nothing to do â€” all embeddings are current.")
        return

    # â”€â”€ Build embedder callable â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    async def embedder(texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using the configured embedding service."""
        try:
            from graphrag.retrieval.embedder import embed_batch
            return await embed_batch(texts, model=model)
        except Exception as exc:
            print(f"[re_embed] Embedding error: {exc}", file=sys.stderr)
            raise

    # â”€â”€ Apply re-embedding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"[re_embed] Starting re-embedding with batch_size={args.batch_size} ...")
    result = await registry.apply_re_embedding(
        embedder=embedder,
        model=model,
        version=version,
        tenant=tenant,
        batch_size=args.batch_size,
    )

    print(
        f"[re_embed] Done.  "
        f"embedded={result.get('embedded', 0)}  "
        f"failed={result.get('failed', 0)}  "
        f"skipped={result.get('skipped', 0)}"
    )

    # â”€â”€ Record version in audit node â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    await registry.record_version(
        model=model,
        version=version,
        dim=expected_dim,
        notes=f"re_embed.py run: {result.get('embedded', 0)} entities updated",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-embed stale entities in the knowledge graph."
    )
    parser.add_argument("--tenant",     default="default",
                        help="Tenant to process (default: 'default')")
    parser.add_argument("--model",      default="",
                        help="Embedding model name (default: from config)")
    parser.add_argument("--version",    default="latest",
                        help="Model version tag (default: 'latest')")
    parser.add_argument("--batch-size", type=int, default=128,
                        dest="batch_size",
                        help="Entities per batch (default: 128)")
    parser.add_argument("--limit",      type=int, default=10000,
                        help="Max entities to queue per run (default: 10000)")
    parser.add_argument("--dry-run",    action="store_true", dest="dry_run",
                        help="Queue only â€” do not write embeddings")
    parser.add_argument("--force",      action="store_true",
                        help="Re-embed ALL entities, even non-stale")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))

