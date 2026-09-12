#!/usr/bin/env python
"""Run the complete deterministic Energy Asset Intelligence scenario.

The default run is safe and local: it creates the synthetic SAP-shaped
SQLite export, executes R2RML/RML into RDF, publishes through SHACL, runs the
version-controlled SPARQL advisory, demonstrates history, abstention and
tenant isolation, exports Turtle, and builds the governed Neo4j read-model
mutation batches without writing to Neo4j.

Use ``--live-neo4j`` to execute the same projection against the configured
Neo4j client. RDF remains authoritative; this flag only writes the optional
rebuildable read model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from rdflib import Graph
from rdflib.namespace import RDF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.core.models import Entity  # noqa: E402
from graphrag.domains.energy.demo import ASSET, ENERGY, EnergyDemoService, TENANT  # noqa: E402
from graphrag.domains.energy.lpg_projection import project_to_neo4j  # noqa: E402
from graphrag.graph.neo4j_client import close_neo4j, get_neo4j  # noqa: E402
from scripts.create_energy_demo_sqlite import create  # noqa: E402


class _RecordingProjectionTarget:
    """Capture the real Neo4j batch contract for a dry-run demonstration."""

    def __init__(self) -> None:
        self.entities: list[Entity] = []
        self.relationships: list[dict[str, Any]] = []

    async def merge_entities_batch(self, entities: list[Entity], tenant: str = "default") -> list[dict[str, Any]]:
        self.entities = entities
        self.entity_tenant = tenant
        return []

    async def merge_relations_batch(self, rows: list[dict[str, Any]], tenant: str = "default") -> None:
        self.relationships = rows
        self.relationship_tenant = tenant


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"E2E assertion failed: {message}")


def _display_path(path: Path) -> str:
    """Prefer repository-relative paths, but support isolated temp tests."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_scenario(*, source: Path, turtle: Path, live_neo4j: bool = False) -> dict[str, Any]:
    """Execute and assert the complete local Energy scenario."""
    create(source)
    service = EnergyDemoService(source_db=source)
    report = service.publication_report()
    _check(report.quarantined_count == 0, "valid source publication is quarantined")
    _check(report.published_triple_count == len(service.graph), "publication count is inconsistent")

    turtle.parent.mkdir(parents=True, exist_ok=True)
    turtle.write_text(service.export_turtle(), encoding="utf-8")
    reparsed = Graph().parse(turtle, format="turtle")
    _check(len(reparsed) == len(service.graph), "Turtle export cannot be reparsed losslessly")
    _check((ASSET["WT-01"], RDF.type, ENERGY.WindTurbine) in service.graph, "R2RML turbine typing is missing")

    maintenance = service.answer("maintenance_review", tenant=TENANT)
    historical = service.answer("historical_state", tenant=TENANT, as_of="2026-05-01T00:00:00Z")
    insufficient = service.answer("insufficient_evidence", tenant=TENANT)
    wrong_tenant = service.answer("maintenance_review", tenant="other-tenant")
    validation = service.validate_candidate()

    _check(maintenance["status"] == "advisory", "maintenance review did not produce an advisory")
    _check("WT-01" in maintenance["answer"], "WT-01 is missing from the advisory")
    _check(maintenance["answer_source"] == "version-controlled SPARQL query", "SPARQL provenance missing")
    _check(historical["authoritative_bulletin"] == "MFG-GBX-17-R1", "historical bulletin resolution failed")
    _check(insufficient["status"] == "insufficient_evidence", "no-evidence boundary was not preserved")
    _check(wrong_tenant["status"] == "not_found" and not wrong_tenant["evidence"], "tenant isolation failed")
    _check(validation["conforms"] is False and validation["violations"], "invalid RDF was not rejected by SHACL")

    dry_target = _RecordingProjectionTarget()
    projection = asyncio.run(project_to_neo4j(service.graph, dry_target, tenant=TENANT))
    _check(projection.node_count == len(dry_target.entities), "projection node count mismatch")
    _check(projection.relationship_count == len(dry_target.relationships), "projection edge count mismatch")
    _check(any(entity.type == "WIND_TURBINE" for entity in dry_target.entities), "Neo4j turbine label missing")
    _check(any(row["relation"] == "HAS_COMPONENT" for row in dry_target.relationships), "Neo4j topology edge missing")

    live_report: dict[str, Any] | None = None
    if live_neo4j:
        async def write_live() -> Any:
            try:
                return await project_to_neo4j(service.graph, get_neo4j(), tenant=TENANT)
            finally:
                await close_neo4j()

        live = asyncio.run(write_live())
        live_report = {
            "tenant": live.tenant,
            "nodes": live.node_count,
            "relationships": live.relationship_count,
        }

    return {
        "scenario": "energy-asset-intelligence/e2e-v2",
        "status": "passed",
        "source": {
            "kind": "synthetic SAP-shaped SQLite",
            "path": _display_path(source),
            "mapping": "ontology/mappings/energy-assets.r2rml.ttl",
            "rml_mapping": "ontology/mappings/energy-observations.rml.ttl",
        },
        "publication": {
            "version_id": report.version_id,
            "candidate_records": report.candidate_record_count,
            "published_triples": report.published_triple_count,
            "quarantined_records": report.quarantined_count,
            "shacl_invalid_probe": validation,
        },
        "rdf": {
            "turtle": _display_path(turtle),
            "triples": len(service.graph),
            "explicit_wind_turbine_type": True,
        },
        "answers": {
            "maintenance_review": maintenance,
            "historical_state": historical,
            "insufficient_evidence": insufficient,
            "wrong_tenant": wrong_tenant,
        },
        "neo4j_projection": {
            "mode": "live" if live_neo4j else "dry-run",
            "rdf_is_authoritative": True,
            "rebuildable_read_model": True,
            "nodes": projection.node_count,
            "relationships": projection.relationship_count,
            "live_result": live_report,
        },
        "capabilities_demonstrated": [
            "R2RML relational-to-RDF materialization",
            "RML telemetry-to-RDF materialization",
            "explicit RDF/OWL domain typing",
            "SHACL publication gate and invalid-record rejection",
            "version-controlled SPARQL advisory",
            "evidence-backed recommendation with provenance",
            "historical revision-aware reasoning",
            "no-evidence abstention boundary",
            "tenant isolation",
            "Turtle export and reparsing",
            "governed RDF-to-Neo4j GraphRAG projection",
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/energy-demo-e2e-sap.sqlite")
    parser.add_argument("--turtle", type=Path, default=ROOT / "artifacts/energy-demo-e2e.ttl")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--live-neo4j", action="store_true", help="Also write the projection to configured Neo4j")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_scenario(source=args.source, turtle=args.turtle, live_neo4j=args.live_neo4j)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote E2E report: {args.output}")
    print(rendered)


if __name__ == "__main__":
    main()
