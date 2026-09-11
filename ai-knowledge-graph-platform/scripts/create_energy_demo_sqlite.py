#!/usr/bin/env python
"""Create the synthetic SAP-shaped SQLite export used by the energy R2RML demo."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def create(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            DROP TABLE IF EXISTS sap_assets;
            DROP TABLE IF EXISTS sap_work_orders;
            CREATE TABLE sap_assets (asset_id TEXT PRIMARY KEY, asset_name TEXT NOT NULL);
            CREATE TABLE sap_work_orders (work_order_id TEXT PRIMARY KEY, asset_id TEXT NOT NULL, status TEXT NOT NULL);
        """)
        connection.executemany(
            "INSERT INTO sap_assets VALUES (?, ?)",
            [(f"WT-{index:02d}", f"Wind turbine WT-{index:02d}") for index in range(1, 11)],
        )
        connection.executemany(
            "INSERT INTO sap_work_orders VALUES (?, ?, ?)",
            [("WO-9001", "WT-01", "open"), ("WO-9002", "WT-02", "open"), ("WO-9003", "WT-03", "closed")],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/energy-demo-sap.sqlite"))
    args = parser.parse_args()
    create(args.output)
    print(f"Created synthetic SAP-shaped SQLite export: {args.output}")


if __name__ == "__main__":
    main()
