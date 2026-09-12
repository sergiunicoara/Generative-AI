"""Synthetic SAP-shaped fixture data shared by the Energy demo and its CLI.

Moved out of ``scripts/create_energy_demo_sqlite.py`` so
``graphrag/domains/energy/demo.py`` (library code) can build the identical
fixture into an ephemeral SQLite file without importing from ``scripts/``
-- CLI entry points aren't meant to be import targets for library code.
The script is now a thin wrapper around this, matching this repo's
existing script/library split (e.g. ``scripts/materialize_r2rml_rdf.py``
around ``graphrag/ingestion/r2rml_rdf.py``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def create_sap_fixture_sqlite(path: Path) -> None:
    """Create the synthetic SAP-shaped SQLite export used by the Energy demo.

    Ten wind turbines (``sap_assets``) and three work orders
    (``sap_work_orders``), read for real by
    ``graphrag/ingestion/r2rml_rdf.py``'s ``materialize_r2rml()`` against
    ``ontology/mappings/energy-assets.r2rml.ttl``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript("""
            DROP TABLE IF EXISTS sap_assets;
            DROP TABLE IF EXISTS sap_work_orders;
            CREATE TABLE sap_assets (asset_id TEXT PRIMARY KEY, asset_name TEXT NOT NULL);
            CREATE TABLE sap_work_orders (work_order_id TEXT PRIMARY KEY, asset_id TEXT NOT NULL, status TEXT NOT NULL);
        """)
        conn.executemany(
            "INSERT INTO sap_assets VALUES (?, ?)",
            [(f"WT-{index:02d}", f"Wind turbine WT-{index:02d}") for index in range(1, 11)],
        )
        conn.executemany(
            "INSERT INTO sap_work_orders VALUES (?, ?, ?)",
            [("WO-9001", "WT-01", "open"), ("WO-9002", "WT-02", "open"), ("WO-9003", "WT-03", "closed")],
        )
        conn.commit()
    finally:
        conn.close()


__all__ = ["create_sap_fixture_sqlite"]
