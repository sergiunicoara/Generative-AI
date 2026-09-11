#!/usr/bin/env python
"""
Execute an R2RML mapping into real RDF triples and write them as Turtle.

r2rml_to_mapping() (graphrag/ingestion/r2rml.py) parses an R2RML mapping but
only ever feeds it to RelationalGraphIngestor, which writes to Neo4j -- no
code path turned a relational row into an actual RDF triple via R2RML before
this script. materialize_r2rml() (graphrag/ingestion/r2rml_rdf.py) is that
missing execution step: it walks the mapping graph directly against a
TabularSourceConnector and returns a populated rdflib.Graph.

This script only materializes Turtle to disk. Loading it into a real
triplestore reuses the existing, unmodified scripts/load_blazegraph.py --
no loading logic is duplicated here.

Usage:
    python scripts/materialize_r2rml_rdf.py \
        --mapping ontology/mappings/energy-assets.r2rml.ttl \
        --sqlite artifacts/energy-demo-sap.sqlite \
        --output artifacts/energy-demo-graph.ttl

Then, to prove it end to end against a real triplestore:
    docker compose up -d blazegraph
    python scripts/load_blazegraph.py --input artifacts/energy-demo-graph.ttl --namespace kb

    # The "maintenance SPARQL query" this pipeline was built to answer --
    # which assets currently have open work orders -- against
    # http://localhost:9999/bigdata/namespace/kb/sparql:
    #
    #   PREFIX energy: <https://example.energy.demo/ontology#>
    #   PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    #   SELECT ?asset ?assetLabel ?workOrder ?status WHERE {
    #     ?workOrder a energy:WorkOrder ;
    #                energy:status ?status ;
    #                energy:concernsAsset ?asset .
    #     ?asset rdfs:label ?assetLabel .
    #     FILTER(?status = "open")
    #   }
    #   ORDER BY ?asset
    #
    # against the shipped energy-assets fixture (scripts/create_energy_demo_sqlite.py)
    # this returns exactly WT-01/WO-9001 and WT-02/WO-9002 -- WO-9003 (status
    # "closed") is correctly excluded.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import structlog  # noqa: E402  # ROOT must be added to sys.path first.

from ingest_r2rml import _build_connector  # noqa: E402  # same directory, on sys.path via script invocation.

from graphrag.ingestion.r2rml import R2RMLMappingError  # noqa: E402
from graphrag.ingestion.r2rml_rdf import materialize_r2rml  # noqa: E402

log = structlog.get_logger("materialize_r2rml_rdf")


async def main(args: argparse.Namespace) -> None:
    connector = _build_connector(args)
    graph = await materialize_r2rml(args.mapping, connector)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=str(output_path), format="turtle")
    log.info(
        "materialize_r2rml_rdf.materialized",
        mapping=args.mapping,
        triples=len(graph),
        output=str(output_path),
    )
    print(f"Materialized {len(graph)} triples from {args.mapping} -> {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mapping", required=True, help="Path to an R2RML .ttl mapping file")
    parser.add_argument("--output", required=True, help="Path to write the materialized Turtle to")
    parser.add_argument("--sqlite", help="Path to a local SQLite database")
    parser.add_argument("--postgres-url", help="postgresql+asyncpg:// or postgresql:// URL")
    parser.add_argument("--excel", help="Path to an .xlsx workbook")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main(_parse_args()))
    except R2RMLMappingError as exc:
        log.error("materialize_r2rml_rdf.mapping_rejected", error=str(exc))
        raise SystemExit(1) from exc
