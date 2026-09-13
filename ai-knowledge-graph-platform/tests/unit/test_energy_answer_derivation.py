"""Mutation regression: answers and citations must track the source data.

This is the executable refutation of the defect that motivated
``graphrag/domains/energy/answers.py``. The demo previously *looked* correct
because its prose was hand-written to match one fixture: it asserted a literal
``96°C``, a literal ``WO-9001`` and a literal ``WT-04 through WT-10``, picked
its bulletin with a hard-coded date branch, and returned three fixed evidence
rows for every question. Change the source records and none of it moved.

Each test here changes exactly one thing about the synthetic source -- the
affected turbine, the temperature, the threshold, the work-order status, the
bulletin revision, or the availability of evidence -- and asserts the answer,
the citations, and the abstention behaviour all move together. Every one of
these fails against the pre-change implementation.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from graphrag.domains.energy.demo import EnergyDemoService, TENANT
from graphrag.domains.energy.fixtures import create_sap_fixture_sqlite

ROOT = Path(__file__).resolve().parents[2]
OBSERVATIONS = ROOT / "data" / "energy_demo" / "snowflake_observations.json"


@pytest.fixture
def source(tmp_path) -> Path:
    database = tmp_path / "energy.sqlite"
    create_sap_fixture_sqlite(database)
    return database


@pytest.fixture
def observations(monkeypatch, tmp_path):
    """Rewrite the RML telemetry source for one test, then restore it.

    The mapping resolves its source relative to the repo root, so the file
    itself is swapped and put back rather than redirected -- restoring in a
    fixture teardown keeps a failing test from leaving the repo dirty.
    """
    original = OBSERVATIONS.read_text(encoding="utf-8")

    def _write(records: list[dict]) -> None:
        OBSERVATIONS.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    yield _write
    OBSERVATIONS.write_text(original, encoding="utf-8")


def _answer(source_db: Path, question_id: str, **kwargs) -> dict:
    return EnergyDemoService(source_db=source_db).answer(question_id, tenant=TENANT, **kwargs)


def _evidence_ids(result: dict) -> set[str]:
    return {item["source_id"] for item in result["evidence"]}


def _baseline(records: list[dict]) -> list[dict]:
    return [dict(record) for record in records]


_BASE_OBSERVATIONS = [
    {"observation_id": "SNOW-OBS-WT-01", "asset_id": "WT-01", "metric": "temperature_c",
     "value": 96.0, "unit": "C", "observed_at": "2026-08-28T08:00:00Z",
     "recorded_at": "2026-08-28T08:00:00Z"},
    {"observation_id": "SNOW-OBS-WT-02", "asset_id": "WT-02", "metric": "vibration_mm_s",
     "value": 12.4, "unit": "mm/s", "observed_at": "2026-08-28T08:05:00Z",
     "recorded_at": "2026-08-28T08:05:00Z"},
    {"observation_id": "SNOW-OBS-WT-03", "asset_id": "WT-03", "metric": "temperature_c",
     "value": 72.0, "unit": "C", "observed_at": "2026-08-28T08:10:00Z",
     "recorded_at": "2026-08-28T08:10:00Z"},
]


class TestAnswersTrackChangedSourceData:
    def test_baseline_names_the_turbine_reading_threshold_and_work_order(self, source):
        """With no instant given, "now" is the latest recorded time in the
        graph -- so the advisory reflects the corrected reading (91.5), not
        the original 96.0 it superseded. See TestBitemporalViews for the
        original still being what a pre-correction view returns."""
        result = _answer(source, "maintenance_review")
        assert result["status"] == "advisory"
        for stated in ("WT-01", "91.5", "85.0", "MFG-GBX-17-R2", "WO-9001"):
            assert stated in result["answer"]
        assert _evidence_ids(result) == {"SNOW-OBS-WT-01-R2", "WO-9001", "MFG-GBX-17-R2"}

    def test_changing_the_affected_turbine_moves_the_answer_and_the_citation(
        self, source, observations,
    ):
        """Move the over-threshold reading from WT-01 to WT-03. The advisory
        must now name WT-03 and cite WT-03's observation -- the old
        implementation kept saying WT-01 because the turbine was only ever
        read from the query for a name list, while the sentence was literal.

        Both facts have to move: an advisory needs a reading over the
        threshold AND an open work order, and WT-03's WO-9003 ships closed.
        Raising only the temperature correctly produces no advisory, which is
        itself the review condition being genuinely evaluated rather than
        decorated -- see the assertion on `partial` below.
        """
        records = _baseline(_BASE_OBSERVATIONS)
        records[0]["value"] = 70.0            # WT-01 drops below threshold
        records[2]["value"] = 97.5            # WT-03 rises above it
        observations(records)

        partial = _answer(source, "maintenance_review")
        assert partial["query_rows"] == []    # over threshold, but no open work order

        with sqlite3.connect(source) as db:
            db.execute("UPDATE sap_work_orders SET status = 'open' WHERE work_order_id = 'WO-9003'")
        result = _answer(source, "maintenance_review")

        assert "WT-03" in result["answer"]
        assert "WT-01" not in result["answer"]
        assert "97.5" in result["answer"]
        assert "96.0" not in result["answer"]
        assert "WO-9003" in result["answer"]
        assert "SNOW-OBS-WT-03" in _evidence_ids(result)
        assert "SNOW-OBS-WT-01" not in _evidence_ids(result)
        assert "WO-9003" in _evidence_ids(result)
        assert "WO-9001" not in _evidence_ids(result)

    def test_changing_the_temperature_changes_the_stated_reading(self, source, observations):
        records = _baseline(_BASE_OBSERVATIONS)
        records[0]["value"] = 101.25
        observations(records)

        result = _answer(source, "maintenance_review")

        assert "101.25" in result["answer"]
        assert "96" not in result["answer"]
        cited = [item for item in result["evidence"] if item["source_id"] == "SNOW-OBS-WT-01"]
        assert cited and cited[0]["value"] == "101.25 C"

    def test_dropping_every_reading_below_the_threshold_withdraws_the_advisory(
        self, source, observations,
    ):
        records = _baseline(_BASE_OBSERVATIONS)
        records[0]["value"] = 40.0
        observations(records)

        result = _answer(source, "maintenance_review")

        assert result["query_rows"] == []
        assert "WT-01" not in result["answer"]
        assert "No asset exceeds" in result["answer"]

    def test_closing_the_work_order_withdraws_the_advisory_and_its_citation(self, source):
        """The advisory requires an *open* work order. Closing WO-9001 must
        remove WT-01 from the review and stop citing that work order."""
        with sqlite3.connect(source) as db:
            db.execute("UPDATE sap_work_orders SET status = 'closed' WHERE work_order_id = 'WO-9001'")

        result = _answer(source, "maintenance_review")

        assert result["query_rows"] == []
        assert "WT-01" not in result["answer"]
        assert "WO-9001" not in _evidence_ids(result)

    def test_open_work_orders_answer_follows_the_status_column(self, source):
        before = _answer(source, "open_work_orders")
        assert "WO-9001" in before["answer"] and "WO-9002" in before["answer"]

        with sqlite3.connect(source) as db:
            db.execute("UPDATE sap_work_orders SET status = 'closed' WHERE work_order_id = 'WO-9002'")
            db.execute("UPDATE sap_work_orders SET status = 'open' WHERE work_order_id = 'WO-9003'")

        after = _answer(source, "open_work_orders")

        assert "WO-9002" not in after["answer"]
        assert "WO-9003" in after["answer"]
        assert "WO-9002" not in _evidence_ids(after)
        assert "WO-9003" in _evidence_ids(after)


class TestBulletinSelectionIsDerivedFromStoredValidity:
    def test_historical_instant_resolves_to_the_earlier_revision(self, source):
        """Selection comes from energy:validFrom/validTo, not a hard-coded
        `effective >= 2026-06-01` branch."""
        historical = _answer(source, "historical_state", as_of="2026-05-01T00:00:00Z")

        assert historical["authoritative_bulletin"] == "MFG-GBX-17-R1"
        assert "90.0" in historical["answer"]
        assert "85.0" not in historical["answer"]
        assert _evidence_ids(historical) == {"MFG-GBX-17-R1"}

    def test_instant_before_any_guidance_abstains_instead_of_guessing(self, source):
        result = _answer(source, "maintenance_review", as_of="2025-01-01T00:00:00Z")

        assert result["status"] == "insufficient_evidence"
        assert result["authoritative_bulletin"] == ""
        assert result["evidence"] == []
        # The abstention now names *which* guidance problem blocked it --
        # missing vs expired vs conflicting are distinguishable outcomes.
        # See tests/unit/test_energy_temporal_correctness.py for each case.
        assert result["guidance"]["status"] == "missing"
        assert "no guidance revision recorded" in result["answer"]

    def test_the_threshold_in_force_changes_which_assets_are_reviewed(self, source, observations):
        """A reading of 92 is over R1's 90 threshold but under nothing at
        R2's 85 -- so the same reading must be reviewable under both, while a
        reading of 87 is reviewable only under R2. This is the threshold
        genuinely driving the outcome rather than decorating the sentence."""
        records = _baseline(_BASE_OBSERVATIONS)
        records[0]["value"] = 87.0
        observations(records)

        current = _answer(source, "maintenance_review")
        historical = _answer(source, "maintenance_review", as_of="2026-05-01T00:00:00Z")

        assert "WT-01" in current["answer"]           # 87 > 85 (R2)
        assert current["authoritative_bulletin"] == "MFG-GBX-17-R2"
        assert historical["query_rows"] == []          # 87 < 90 (R1)
        assert historical["authoritative_bulletin"] == "MFG-GBX-17-R1"

    def test_revision_change_reports_the_actual_supersession(self, source):
        result = _answer(source, "revision_change")

        assert "MFG-GBX-17-R2" in result["answer"]
        assert "MFG-GBX-17-R1" in result["answer"]
        assert "90.0" in result["answer"] and "85.0" in result["answer"]

    def test_revision_change_at_the_earliest_revision_says_nothing_was_superseded(self, source):
        result = _answer(source, "revision_change", as_of="2026-05-01T00:00:00Z")

        assert result["query_rows"] == []
        assert "supersedes no earlier revision" in result["answer"]


class TestAbstentionIsQueryDerived:
    def test_insufficient_evidence_names_the_assets_the_data_actually_lacks(self, source):
        """The old answer asserted "WT-04 through WT-10" -- seven assets, and
        wrong: WT-02 has only vibration telemetry, so it cannot be assessed
        for a gearbox *temperature* review either."""
        result = _answer(source, "insufficient_evidence")

        assert result["status"] == "insufficient_evidence"
        named = {row["asset"].rsplit("/", 1)[-1] for row in result["query_rows"]}
        assert named == {"WT-02", "WT-04", "WT-05", "WT-06", "WT-07", "WT-08", "WT-09", "WT-10"}
        assert "WT-02" in result["answer"]
        assert "WT-01" not in result["answer"]
        assert "WT-03" not in result["answer"]

    def test_supplying_the_missing_evidence_removes_an_asset_from_the_gap(
        self, source, observations,
    ):
        records = _baseline(_BASE_OBSERVATIONS)
        records.append({
            "observation_id": "SNOW-OBS-WT-04", "asset_id": "WT-04", "metric": "temperature_c",
            "value": 55.0, "unit": "C", "observed_at": "2026-08-28T08:20:00Z",
            "recorded_at": "2026-08-28T08:20:00Z",
        })
        observations(records)
        with sqlite3.connect(source) as db:
            db.execute(
                "INSERT INTO sap_work_orders VALUES ('WO-9004', 'WT-04', 'open',"
                " '2026-08-20T09:00:00Z', '2026-08-20T09:00:00Z')"
            )

        result = _answer(source, "insufficient_evidence")

        named = {row["asset"].rsplit("/", 1)[-1] for row in result["query_rows"]}
        assert "WT-04" not in named
        assert "WT-05" in named


class TestCitationsAreReadFromTheGraph:
    def test_every_citation_carries_real_provenance_and_both_time_axes(self, source):
        result = _answer(source, "maintenance_review")

        assert result["evidence"]
        for item in result["evidence"]:
            assert item["source_document"].startswith("urn:synthetic:")
            assert item["recorded_at"] and item["valid_from"]
            assert item["access_scope"] == TENANT
            # Derived from the record's own prov:wasDerivedFrom IRI.
            assert item["source_type"] in {
                "sap_work_orders", "snowflake_telemetry", "sharepoint_technical_guidance",
            }

    def test_evidence_differs_per_question_rather_than_being_one_fixed_list(self, source):
        review = _evidence_ids(_answer(source, "maintenance_review"))
        historical = _evidence_ids(_answer(source, "historical_state", as_of="2026-05-01T00:00:00Z"))

        assert review != historical
        assert "SNOW-OBS-WT-01-R2" in review and "SNOW-OBS-WT-01-R2" not in historical
        assert historical == {"MFG-GBX-17-R1"}
