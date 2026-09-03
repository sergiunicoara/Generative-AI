"""Bulk-load a tenant's RDF export into a live Blazegraph SPARQL endpoint.

Blazegraph is an optional interoperability sink -- Neo4j remains the source
of truth (see docs/adr/0001-property-graph-over-triple-store.md), and the
Turtle files produced by ``scripts/export_rdf.py`` remain the canonical
export. This script just mirrors that export into a real, queryable
SPARQL 1.1 endpoint, for tools/workflows that expect one.

Loading uses Blazegraph's SPARQL 1.1 Graph Store HTTP Protocol support: a
raw Turtle document POSTed directly to a namespace's ``/sparql`` endpoint
with ``Content-Type: text/turtle`` is bulk-inserted into that namespace's
default graph.

Usage
-----
  docker compose up -d blazegraph
  python scripts/export_rdf.py --tenant acme
  python scripts/load_blazegraph.py --tenant acme
      # loads exports/acme/graph_export.ttl into the "kb" namespace

  python scripts/load_blazegraph.py --tenant acme --namespace acme_kb
  python scripts/load_blazegraph.py --tenant acme --endpoint http://blazegraph-host:9999
  python scripts/load_blazegraph.py --input custom/path.ttl --namespace kb
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx
import structlog

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

log = structlog.get_logger(__name__)

DEFAULT_ENDPOINT = "http://localhost:9999"


def load(
    ttl_path: Path,
    endpoint: str = DEFAULT_ENDPOINT,
    namespace: str = "kb",
    timeout: float = 60.0,
) -> int:
    """POST a Turtle file to a Blazegraph namespace's SPARQL endpoint.

    Returns the HTTP status code from Blazegraph on success; raises for
    network errors or a non-2xx response.
    """
    if not ttl_path.exists():
        raise FileNotFoundError(f"Turtle file not found: {ttl_path}")

    url = f"{endpoint.rstrip('/')}/blazegraph/namespace/{namespace}/sparql"
    body = ttl_path.read_bytes()

    resp = httpx.post(
        url,
        content=body,
        headers={"Content-Type": "text/turtle"},
        timeout=timeout,
    )
    resp.raise_for_status()
    log.info(
        "load_blazegraph.loaded",
        path=str(ttl_path),
        endpoint=url,
        bytes=len(body),
        status=resp.status_code,
    )
    return resp.status_code


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load an RDF Turtle export into a Blazegraph SPARQL endpoint"
    )
    parser.add_argument("--tenant", default=None,
                        help="Tenant whose export to load "
                             "(exports/<tenant>/graph_export.ttl)")
    parser.add_argument("--input", default=None,
                        help="Explicit Turtle file path (overrides --tenant)")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                        help=f"Blazegraph base URL (default: {DEFAULT_ENDPOINT})")
    parser.add_argument("--namespace", default="kb",
                        help="Blazegraph namespace to load into (default: kb)")
    args = parser.parse_args()

    if args.input:
        ttl_path = Path(args.input)
    elif args.tenant:
        export_dir = Path(os.getenv("GRAPHRAG_RDF_EXPORT_DIR", "exports"))
        ttl_path = export_dir / args.tenant / "graph_export.ttl"
    else:
        parser.error("Provide either --tenant or --input")
        return

    status = load(ttl_path, endpoint=args.endpoint, namespace=args.namespace)
    print(f"[OK] Loaded {ttl_path} into {args.endpoint} "
          f"(namespace={args.namespace}, status={status})")


if __name__ == "__main__":
    main()
