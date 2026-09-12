from __future__ import annotations

from pathlib import Path

from scripts.run_energy_demo_e2e import run_scenario


def test_complete_energy_scenario(tmp_path: Path) -> None:
    result = run_scenario(
        source=tmp_path / "source.sqlite",
        turtle=tmp_path / "energy.ttl",
    )

    assert result["status"] == "passed"
    assert result["publication"]["quarantined_records"] == 0
    assert result["rdf"]["explicit_wind_turbine_type"] is True
    assert result["answers"]["maintenance_review"]["status"] == "advisory"
    assert result["answers"]["wrong_tenant"]["status"] == "not_found"
    assert result["neo4j_projection"]["mode"] == "dry-run"
    assert len(result["capabilities_demonstrated"]) >= 10
