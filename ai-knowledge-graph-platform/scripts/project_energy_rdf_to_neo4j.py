#!/usr/bin/env python
"""Project the published Energy RDF graph into a rebuildable Neo4j read model.

This is deliberately a one-way projection: RDF remains the authoritative
evidence graph and Neo4j is an optional traversal/GraphRAG optimization.  It
never reads from Neo4j to change RDF.  The projector rejects blank nodes,
unknown Energy vocabulary, repeated literals, and untyped resources rather
than inventing an LPG representation.

Requires a reachable Neo4j configured through the normal application settings.

Usage:
    python scripts/project_energy_rdf_to_neo4j.py
    python scripts/project_energy_rdf_to_neo4j.py --source-sqlite artifacts/energy-demo-sap.sqlite
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.domains.energy.demo import EnergyDemoService, TENANT  # noqa: E402
from graphrag.domains.energy.lpg_projection import project_to_neo4j  # noqa: E402
from graphrag.graph.neo4j_client import close_neo4j, get_neo4j  # noqa: E402


async def _run(source_sqlite: Path | None, tenant: str) -> None:
    service = EnergyDemoService(source_db=source_sqlite)
    try:
        report = await project_to_neo4j(service.graph, get_neo4j(), tenant=tenant)
    finally:
        await close_neo4j()
    print(
        "Projected published Energy RDF to Neo4j read model: "
        f"{report.node_count} nodes, {report.relationship_count} relationships, "
        f"{report.source_graph_triples} source triples (tenant={report.tenant})."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sqlite", type=Path, help="Optional SAP-shaped SQLite source")
    parser.add_argument("--tenant", default=TENANT, help="Trusted tenant to apply to all Neo4j rows")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    asyncio.run(_run(arguments.source_sqlite, arguments.tenant))
