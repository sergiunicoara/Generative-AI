"""Every schema entry point must apply every statement in schema.cypher."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from graphrag.graph.schema_statements import (
    SCHEMA_PATH,
    load_schema_statements,
    parse_schema_statements,
)

ROOT = Path(__file__).parents[2]
ENTRY_POINTS = [
    "workers/ingestion_worker.py",
    "workers/combined_worker.py",
    "scripts/init_schema_only.py",
    "scripts/init_neo4j.py",
    "scripts/ingest_corpus.py",
    "graphrag/graph/neo4j_client.py",
]


def test_statement_after_comment_is_kept():
    text = "-- header\nCREATE CONSTRAINT a IF NOT EXISTS FOR (n:A) REQUIRE n.id IS UNIQUE;\n"
    assert len(parse_schema_statements(text)) == 1


def test_semicolon_inside_comment_does_not_create_fragment():
    text = "-- note; more\nCREATE INDEX i IF NOT EXISTS FOR (n:A) ON (n.x);"
    assert parse_schema_statements(text) == ["CREATE INDEX i IF NOT EXISTS FOR (n:A) ON (n.x)"]


def test_real_schema_loads_every_ddl_statement():
    stmts = load_schema_statements()
    raw = SCHEMA_PATH.read_text(encoding="utf-8")
    expected = len(re.findall(r"^\s*(?:CREATE|DROP)\b", raw, flags=re.MULTILINE))
    assert expected > 60
    assert len(stmts) == expected
    assert all(s.startswith(("CREATE", "DROP")) for s in stmts)


def test_previously_skipped_constraints_are_present():
    names = "\n".join(load_schema_statements())
    for name in ("doc_id", "entity_name_type_tenant", "conflict_id"):
        assert name in names


def test_every_entry_point_uses_shared_loader():
    for rel in ENTRY_POINTS:
        src = (ROOT / rel).read_text(encoding="utf-8")
        ast.parse(src)
        assert "load_schema_statements" in src, rel
        assert 'split(";")' not in src, f"{rel} still splits schema.cypher itself"
