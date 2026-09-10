#!/usr/bin/env python
"""
Ingest a relational/tabular source into Neo4j using a declarative R2RML mapping.

r2rml_to_mapping() (graphrag/ingestion/r2rml.py) and FederatedOBDAIngestor
existed but were exercised only by tests/unit/test_r2rml_obda.py -- nothing
at runtime ever parsed a real .ttl mapping file. This is that entry point.

Deliberately a CLI, not an API route and not the RabbitMQ ingest queue:
  - the queue (POST /ingest -> IngestionConsumer -> IngestionAgent) exists to
    decouple chunking/embedding/LLM extraction on raw text, which is slow and
    worth retrying async. R2RML rows arrive already typed via the declarative
    mapping -- there is no LLM extraction step to decouple.
  - both existing relational demos (demo_sustainability_relational.py,
    demo_sustainability_e2e.py) call RelationalGraphIngestor.ingest()
    synchronously, in-process -- this follows the same, already-established
    pattern rather than inventing a new one.
  - an API route would have to accept a database connection string / file
    path in a request body, which is a bigger surface than this needs.

Usage:
    python scripts/ingest_r2rml.py --mapping ontology/mappings/supply-chain.r2rml.ttl \
        --sqlite path/to/source.db --tenant sustainability --source-id supplier-db

    python scripts/ingest_r2rml.py --mapping ... --postgres-url postgresql+asyncpg://... \
        --tenant sustainability --source-id supplier-db --validate-only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import structlog  # noqa: E402  # ROOT must be added to sys.path first.

from graphrag.ingestion.graph_writer import GraphWriter  # noqa: E402
from graphrag.ingestion.r2rml import R2RMLMappingError, r2rml_to_mapping  # noqa: E402
from graphrag.ingestion.relational import (  # noqa: E402
    ExcelWorkbookConnector,
    PostgreSQLSourceConnector,
    RelationalGraphIngestor,
    SQLiteSourceConnector,
    TabularSourceConnector,
)

log = structlog.get_logger("ingest_r2rml")


def _build_connector(args: argparse.Namespace) -> TabularSourceConnector:
    sources = [
        ("sqlite", args.sqlite, lambda: SQLiteSourceConnector(args.sqlite)),
        ("postgres_url", args.postgres_url, lambda: PostgreSQLSourceConnector(args.postgres_url)),
        ("excel", args.excel, lambda: ExcelWorkbookConnector(args.excel)),
    ]
    given = [(name, build) for name, value, build in sources if value]
    if len(given) != 1:
        names = ", ".join(f"--{name.replace('_', '-')}" for name, _, _ in sources)
        raise SystemExit(f"exactly one of {names} is required (got {len(given)})")
    return given[0][1]()


async def main(args: argparse.Namespace) -> None:
    connector = _build_connector(args)
    try:
        mapping = r2rml_to_mapping(
            args.mapping,
            mapping_id=args.mapping_id or Path(args.mapping).stem,
            version=args.version,
            source_id=args.source_id,
            tenant=args.tenant,
        )
    except R2RMLMappingError as exc:
        log.error("ingest_r2rml.mapping_rejected", error=str(exc))
        raise SystemExit(1) from exc

    # FederatedOBDAIngestor.ingest() already calls validate() as a preflight
    # before any write (see its module docstring), so this is not duplicated
    # work -- it's the --validate-only early exit, and a clearer failure
    # message before touching the connector's write path at all.
    ingestor = RelationalGraphIngestor(connector, GraphWriter(changed_by="ingest-r2rml-cli"))
    report = await ingestor.validate(mapping)
    if not report.valid:
        log.error("ingest_r2rml.validation_failed", errors=report.errors)
        raise SystemExit(1)
    log.info(
        "ingest_r2rml.validated",
        tenant=report.tenant,
        entity_rows=report.entity_rows,
        relation_rows=report.relation_rows,
    )
    if args.validate_only:
        return

    result = await ingestor.ingest(mapping)
    log.info(
        "ingest_r2rml.ingested",
        tenant=result.tenant,
        entity_rows=result.entity_rows,
        relation_rows=result.relation_rows,
        shacl_conforms=result.shacl_conforms,
    )
    print(
        f"Ingested {result.entity_rows} relational rows and "
        f"{result.relation_rows} relation rows for tenant {result.tenant}."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mapping", required=True, help="Path to an R2RML .ttl mapping file")
    parser.add_argument("--mapping-id", default="", help="Defaults to the mapping file's stem")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--sqlite", help="Path to a local SQLite database")
    parser.add_argument("--postgres-url", help="postgresql+asyncpg:// or postgresql:// URL")
    parser.add_argument("--excel", help="Path to an .xlsx workbook")
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Run mapping + SHACL validation and exit without writing to Neo4j",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
