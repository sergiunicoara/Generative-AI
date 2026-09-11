#!/usr/bin/env python
"""Run the client-demo E2E path: source export -> RDF -> SPARQL -> evidence response."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.domains.energy.demo import EnergyDemoService, TENANT  # noqa: E402
from scripts.create_energy_demo_sqlite import create  # noqa: E402


def main() -> None:
    source = ROOT / "artifacts/energy-demo-sap.sqlite"
    turtle = ROOT / "artifacts/energy-demo-e2e.ttl"
    create(source)
    service = EnergyDemoService(source_db=source)
    turtle.write_text(service.export_turtle(), encoding="utf-8")
    assessment = service.answer("maintenance_review", tenant=TENANT)
    historical = service.answer("historical_state", tenant=TENANT, as_of="2026-05-01T00:00:00Z")
    validation = service.validate_candidate()
    print(json.dumps({
        "flow": [
            "SAP-shaped SQLite source created",
            "R2RML mapping parsed and source materialized into RDF",
            "Version-controlled SPARQL query executed",
            "Evidence-backed advisory response returned",
            "SHACL validation executed",
        ],
        "assessment": assessment,
        "historical_state": historical,
        "validation": validation,
        "rdf_export": str(turtle.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
