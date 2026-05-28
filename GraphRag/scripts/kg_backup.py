#!/usr/bin/env python
"""kg_backup.py — Export and restore the GraphRAG knowledge graph.

Usage
-----
    # Backup
    python scripts/kg_backup.py backup --tenant acme --output kg_backup.ndjson
    python scripts/kg_backup.py backup --tenant acme --output s3://my-bucket/backups/kg.ndjson

    # Restore
    python scripts/kg_backup.py restore --input kg_backup.ndjson --tenant acme
    python scripts/kg_backup.py restore --input s3://my-bucket/backups/kg.ndjson --tenant acme

    # List backups in an S3 prefix
    python scripts/kg_backup.py list --prefix s3://my-bucket/backups/

File format
-----------
NDJSON — one JSON object per line.  Each line is tagged with a ``_type`` field:
  {"_type": "entity",   "name": ..., "type": ..., "tenant": ..., ...}
  {"_type": "relation", "src": ...,  "tgt": ...,  "relation": ..., ...}
  {"_type": "chunk",    "id": ...,   "text": ..., ...}
  {"_type": "meta",     "tenant": ..., "exported_at": ..., "version": "1.0"}

S3 support
----------
If the output/input path begins with ``s3://``, the script uses boto3.
Install boto3 with: pip install boto3
AWS credentials must be available via the standard boto3 chain
(env vars, ~/.aws/credentials, IAM role, etc.).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BACKUP_VERSION = "1.0"


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _is_s3(path: str) -> bool:
    return path.startswith("s3://")


def _parse_s3(path: str) -> tuple[str, str]:
    """Return (bucket, key) from an s3://bucket/key URI."""
    without_scheme = path[5:]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


def _open_write(path: str):
    """Return a file-like object opened for writing (local or S3 buffer)."""
    if _is_s3(path):
        import io
        return io.StringIO()   # collect into memory then upload
    return open(path, "w", encoding="utf-8")


def _write_ndjson_line(f, obj: dict) -> None:
    f.write(json.dumps(obj, default=str) + "\n")


def _upload_s3(buf, path: str) -> None:
    try:
        import boto3
    except ImportError:
        print("ERROR: boto3 not installed. Run: pip install boto3", file=sys.stderr)
        sys.exit(1)
    bucket, key = _parse_s3(path)
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue().encode("utf-8"))
    print(f"[kg_backup] Uploaded to s3://{bucket}/{key}")


def _open_read_lines(path: str):
    """Return an iterable of raw JSON lines from local file or S3."""
    if _is_s3(path):
        try:
            import boto3
        except ImportError:
            print("ERROR: boto3 not installed. Run: pip install boto3", file=sys.stderr)
            sys.exit(1)
        bucket, key = _parse_s3(path)
        s3   = boto3.client("s3")
        obj  = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read().decode("utf-8")
        return body.splitlines()
    return open(path, "r", encoding="utf-8").readlines()


# ── Backup ────────────────────────────────────────────────────────────────────

async def do_backup(args: argparse.Namespace) -> None:
    from graphrag.graph.neo4j_client import get_neo4j

    neo4j  = get_neo4j()
    tenant = args.tenant
    output = args.output
    is_s3  = _is_s3(output)

    print(f"[kg_backup] Backing up tenant={tenant!r} → {output}")

    f = _open_write(output)
    n_entities = n_relations = n_chunks = 0

    # ── Metadata header ──────────────────────────────────────────────────────
    meta = {
        "_type":       "meta",
        "tenant":      tenant,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "version":     BACKUP_VERSION,
    }
    _write_ndjson_line(f, meta)

    # ── Entities ─────────────────────────────────────────────────────────────
    entity_rows = await neo4j.run(
        """
        MATCH (e:Entity {tenant: $tenant})
        RETURN e.name         AS name,
               e.type         AS type,
               e.tenant       AS tenant,
               e.description  AS description,
               e.confidence   AS confidence,
               e.valid_from   AS valid_from,
               e.valid_to     AS valid_to,
               e.wikidata_qid AS wikidata_qid,
               e.quarantined  AS quarantined
        """,
        tenant=tenant,
    )
    for row in entity_rows:
        _write_ndjson_line(f, {"_type": "entity", **{k: v for k, v in row.items() if v is not None}})
    n_entities = len(entity_rows)
    print(f"[kg_backup]   entities : {n_entities}")

    # ── Relations ─────────────────────────────────────────────────────────────
    rel_rows = await neo4j.run(
        """
        MATCH (s:Entity {tenant: $tenant})-[r:RELATES_TO]->(t:Entity {tenant: $tenant})
        RETURN s.name          AS src,
               s.type          AS src_type,
               t.name          AS tgt,
               t.type          AS tgt_type,
               r.relation      AS relation,
               r.confidence    AS confidence,
               r.source_doc_ids AS source_doc_ids,
               r.valid_from    AS valid_from,
               r.valid_to      AS valid_to,
               r.source_type   AS source_type,
               r.tenant        AS tenant
        """,
        tenant=tenant,
    )
    for row in rel_rows:
        _write_ndjson_line(f, {"_type": "relation", **{k: v for k, v in row.items() if v is not None}})
    n_relations = len(rel_rows)
    print(f"[kg_backup]   relations: {n_relations}")

    # ── Chunks ────────────────────────────────────────────────────────────────
    chunk_rows = await neo4j.run(
        """
        MATCH (c:Chunk {tenant: $tenant})
        RETURN c.id           AS id,
               c.text         AS text,
               c.document_id  AS document_id,
               c.chunk_index  AS chunk_index,
               c.tenant       AS tenant,
               c.redacted     AS redacted
        """,
        tenant=tenant,
    )
    for row in chunk_rows:
        if row.get("redacted"):   # skip GDPR-redacted chunks
            continue
        _write_ndjson_line(f, {"_type": "chunk", **{k: v for k, v in row.items() if v is not None}})
    n_chunks = len(chunk_rows)
    print(f"[kg_backup]   chunks   : {n_chunks}")

    if is_s3:
        _upload_s3(f, output)
    else:
        f.close()
        print(f"[kg_backup] Written to {output}")

    print(
        f"[kg_backup] Backup complete — "
        f"{n_entities} entities, {n_relations} relations, {n_chunks} chunks"
    )


# ── Restore ───────────────────────────────────────────────────────────────────

async def do_restore(args: argparse.Namespace) -> None:
    from graphrag.graph.neo4j_client import get_neo4j

    neo4j          = get_neo4j()
    input_path     = args.input
    override_tenant = args.tenant    # if set, override tenant from file

    print(f"[kg_backup] Restoring from {input_path} ...")

    lines = _open_read_lines(input_path)
    n_entities = n_relations = n_chunks = 0
    skipped = 0

    for raw_line in lines:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            skipped += 1
            continue

        record_type = obj.pop("_type", None)
        tenant = override_tenant or obj.get("tenant", "default")

        if record_type == "meta":
            print(
                f"[kg_backup] File metadata: "
                f"exported_at={obj.get('exported_at')}, "
                f"version={obj.get('version')}"
            )
            continue

        if record_type == "entity":
            await neo4j.run(
                """
                MERGE (e:Entity {name: $name, type: $type, tenant: $tenant})
                ON CREATE SET e.description  = $description,
                              e.confidence   = $confidence,
                              e.valid_from   = $valid_from,
                              e.wikidata_qid = $wikidata_qid,
                              e.recorded_at  = datetime()
                ON MATCH SET  e.description  = $description
                """,
                name=obj.get("name", ""),
                type=obj.get("type", "CONCEPT"),
                tenant=tenant,
                description=obj.get("description", ""),
                confidence=obj.get("confidence", 1.0),
                valid_from=obj.get("valid_from"),
                wikidata_qid=obj.get("wikidata_qid"),
            )
            n_entities += 1

        elif record_type == "relation":
            await neo4j.run(
                """
                MATCH (s:Entity {name: $src, type: $src_type, tenant: $tenant})
                MATCH (t:Entity {name: $tgt, type: $tgt_type, tenant: $tenant})
                MERGE (s)-[r:RELATES_TO {relation: $relation, tenant: $tenant}]->(t)
                ON CREATE SET r.confidence    = $confidence,
                              r.source_type   = $source_type,
                              r.source_doc_ids = $source_doc_ids,
                              r.recorded_at   = datetime()
                """,
                src=obj.get("src", ""),
                src_type=obj.get("src_type", "CONCEPT"),
                tgt=obj.get("tgt", ""),
                tgt_type=obj.get("tgt_type", "CONCEPT"),
                tenant=tenant,
                relation=obj.get("relation", "RELATED_TO"),
                confidence=obj.get("confidence", 1.0),
                source_type=obj.get("source_type", "backup"),
                source_doc_ids=obj.get("source_doc_ids") or [],
            )
            n_relations += 1

        elif record_type == "chunk":
            await neo4j.run(
                """
                MERGE (c:Chunk {id: $id, tenant: $tenant})
                ON CREATE SET c.text        = $text,
                              c.document_id = $document_id,
                              c.chunk_index = $chunk_index
                """,
                id=obj.get("id", ""),
                tenant=tenant,
                text=obj.get("text", ""),
                document_id=obj.get("document_id", ""),
                chunk_index=obj.get("chunk_index", 0),
            )
            n_chunks += 1

    print(
        f"[kg_backup] Restore complete — "
        f"{n_entities} entities, {n_relations} relations, {n_chunks} chunks  "
        f"(skipped {skipped} malformed lines)"
    )


# ── List S3 ───────────────────────────────────────────────────────────────────

def do_list(args: argparse.Namespace) -> None:
    prefix = args.prefix
    if not _is_s3(prefix):
        # Local directory listing
        p = Path(prefix)
        if not p.exists():
            print(f"[kg_backup] Path does not exist: {prefix}", file=sys.stderr)
            sys.exit(1)
        for f in sorted(p.glob("*.ndjson")):
            print(f"  {f}")
        return

    try:
        import boto3
    except ImportError:
        print("ERROR: boto3 not installed. Run: pip install boto3", file=sys.stderr)
        sys.exit(1)

    bucket, key_prefix = _parse_s3(prefix)
    s3  = boto3.client("s3")
    res = s3.list_objects_v2(Bucket=bucket, Prefix=key_prefix)
    for obj in res.get("Contents", []):
        print(f"  s3://{bucket}/{obj['Key']}  ({obj['Size']} bytes,  {obj['LastModified']})")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export or restore the GraphRAG knowledge graph."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # backup
    bp = sub.add_parser("backup", help="Export graph to NDJSON file")
    bp.add_argument("--tenant", default="default")
    bp.add_argument("--output", required=True,
                    help="Output path (local file or s3://bucket/key)")

    # restore
    rp = sub.add_parser("restore", help="Import graph from NDJSON file")
    rp.add_argument("--input", required=True,
                    help="Input path (local file or s3://bucket/key)")
    rp.add_argument("--tenant", default="",
                    help="Override tenant from file (optional)")

    # list
    lp = sub.add_parser("list", help="List backup files at a path or S3 prefix")
    lp.add_argument("--prefix", required=True,
                    help="Local directory or s3://bucket/prefix/")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "backup":
        asyncio.run(do_backup(args))
    elif args.command == "restore":
        asyncio.run(do_restore(args))
    elif args.command == "list":
        do_list(args)
