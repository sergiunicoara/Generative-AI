"""Unit tests for scripts/ingest_r2rml.py — the first real caller of
r2rml_to_mapping()/FederatedOBDAIngestor at runtime (previously exercised
only by tests/unit/test_r2rml_obda.py)."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Allow importing from scripts/ — same convention as test_export_rdf.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import ingest_r2rml  # noqa: E402

from graphrag.ingestion.relational import (  # noqa: E402
    ExcelWorkbookConnector,
    PostgreSQLSourceConnector,
    SQLiteSourceConnector,
)


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        mapping="ontology/mappings/supply-chain.r2rml.ttl",
        mapping_id="", version="1.0.0", tenant="sustainability", source_id="supplier-db",
        sqlite=None, postgres_url=None, excel=None, validate_only=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _sqlite_db(tmp_path) -> str:
    path = tmp_path / "source.db"
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE suppliers (id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT);
        CREATE TABLE materials (id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT);
        CREATE TABLE supplies (supplier_id TEXT NOT NULL, material_id TEXT NOT NULL, confidence REAL);
        INSERT INTO suppliers VALUES ('s1', 'Supplier One', 'x');
        INSERT INTO materials VALUES ('m1', 'Material One', 'x');
        INSERT INTO supplies VALUES ('s1', 'm1', 0.9);
        """)
    return str(path)


class TestBuildConnector:
    def test_exactly_one_source_flag_is_required(self):
        with pytest.raises(SystemExit):
            ingest_r2rml._build_connector(_args())

    def test_two_source_flags_at_once_is_rejected(self):
        with pytest.raises(SystemExit):
            ingest_r2rml._build_connector(_args(sqlite="a.db", excel="b.xlsx"))

    def test_sqlite_flag_builds_a_sqlite_connector(self, tmp_path):
        connector = ingest_r2rml._build_connector(_args(sqlite=_sqlite_db(tmp_path)))
        assert isinstance(connector, SQLiteSourceConnector)

    def test_postgres_url_flag_builds_a_postgres_connector(self):
        connector = ingest_r2rml._build_connector(
            _args(postgres_url="postgresql+asyncpg://u:p@localhost/db")
        )
        assert isinstance(connector, PostgreSQLSourceConnector)

    def test_excel_flag_builds_an_excel_connector(self, tmp_path):
        connector = ingest_r2rml._build_connector(_args(excel=str(tmp_path / "sheet.xlsx")))
        assert isinstance(connector, ExcelWorkbookConnector)


class TestMainFlow:
    async def test_validate_only_does_not_call_ingest(self, tmp_path):
        args = _args(sqlite=_sqlite_db(tmp_path), validate_only=True)
        with patch("ingest_r2rml.GraphWriter"):
            with patch.object(
                ingest_r2rml.RelationalGraphIngestor, "ingest", new_callable=AsyncMock,
            ) as mock_ingest:
                await ingest_r2rml.main(args)
        mock_ingest.assert_not_awaited()

    async def test_validation_failure_exits_before_ingest(self, tmp_path):
        # An empty required-field row must fail validate() and never reach
        # ingest() -- same fail-closed guarantee RelationalGraphIngestor
        # already gives direct callers.
        db_path = tmp_path / "bad.db"
        with sqlite3.connect(db_path) as db:
            db.executescript("""
            CREATE TABLE suppliers (id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT);
            CREATE TABLE materials (id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT);
            CREATE TABLE supplies (supplier_id TEXT NOT NULL, material_id TEXT NOT NULL, confidence REAL);
            INSERT INTO suppliers VALUES ('s1', '', 'missing name');
            """)
        args = _args(sqlite=str(db_path))
        with patch("ingest_r2rml.GraphWriter"):
            with patch.object(
                ingest_r2rml.RelationalGraphIngestor, "ingest", new_callable=AsyncMock,
            ) as mock_ingest:
                with pytest.raises(SystemExit):
                    await ingest_r2rml.main(args)
        mock_ingest.assert_not_awaited()

    async def test_malformed_r2rml_mapping_exits_before_touching_the_connector(self, tmp_path):
        bad_ttl = tmp_path / "bad.ttl"
        bad_ttl.write_text("@prefix rr: <http://www.w3.org/ns/r2rml#> .\n")  # no TriplesMap
        args = _args(mapping=str(bad_ttl), sqlite=_sqlite_db(tmp_path))
        with pytest.raises(SystemExit):
            await ingest_r2rml.main(args)

    async def test_real_supply_chain_mapping_validates_against_a_matching_sqlite_source(self, tmp_path):
        # Exercises the actual shipped mapping file end to end (minus the
        # Neo4j write) -- this is the regression guard proving the mapping
        # file and the adapter still agree with each other.
        args = _args(
            mapping="ontology/mappings/supply-chain.r2rml.ttl",
            sqlite=_sqlite_db(tmp_path), validate_only=True,
        )
        with patch("ingest_r2rml.GraphWriter"):
            await ingest_r2rml.main(args)  # must not raise
