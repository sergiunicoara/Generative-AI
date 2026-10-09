"""Single loader for ``schema.cypher``.

Stdlib only, so bootstrap scripts can load it by file path without importing the
``graphrag`` package. Every schema entry point (client, workers, scripts) must
use this so none of them skip statements that follow a comment line.
"""
from __future__ import annotations

from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.cypher")


def parse_schema_statements(text: str) -> list[str]:
    """Split Cypher DDL text into statements.

    Comment lines are removed *before* splitting on ``;`` so a statement that
    follows a ``--`` comment is kept and a ``;`` inside a comment cannot create
    a bogus fragment.
    """
    code = "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("--")
    )
    return [s for s in (frag.strip() for frag in code.split(";")) if s]


def load_schema_statements(path: Path | None = None) -> list[str]:
    return parse_schema_statements((path or SCHEMA_PATH).read_text(encoding="utf-8"))
