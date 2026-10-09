"""Graph-context validation queries. Read-only by construction.

Every query passes ``assert_read_only`` before it is sent, and is executed in a
READ-access session when the client offers ``run_read``. A validation query can
therefore never mutate the published graph.
"""
from __future__ import annotations

import re

from graphrag.graph.validation.report import ValidationReport, Violation

_WRITE_CLAUSE = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|LOAD\s+CSV|FOREACH)\b|\bCALL\s*\{|\bCALL\s+(db|dbms|apoc)\.",
    re.IGNORECASE,
)


class WriteQueryRejected(ValueError):
    pass


def assert_read_only(cypher: str) -> None:
    stripped = re.sub(r"//[^\n]*", "", cypher)
    stripped = re.sub(r"'[^']*'|\"[^\"]*\"", "''", stripped)
    m = _WRITE_CLAUSE.search(stripped)
    if m:
        raise WriteQueryRejected(f"validation query contains a write/procedure clause: {m.group(0)!r}")


async def run_read_only(neo4j, cypher: str, **params) -> list[dict]:
    assert_read_only(cypher)
    runner = getattr(neo4j, "run_read", None) or neo4j.run
    return await runner(cypher, **params)


EXISTING_DOCUMENTS = """
UNWIND $ids AS id
MATCH (d:Document {id: id, tenant: $tenant})
RETURN d.id AS id
"""


async def check_supersedes(neo4j, *, tenant: str, document_key: str, supersedes: list[str],
                           source: str) -> ValidationReport:
    """Dangling SUPERSEDES references (target not in this tenant).

    Only the caller's tenant is queried: whether the id exists in another
    tenant is deliberately not checked, so the report cannot leak it.
    """
    report = ValidationReport(tenant=tenant, source=source)
    ids = sorted({s for s in supersedes if s})
    if not ids:
        return report
    report.records_checked = len(ids)
    rows = await run_read_only(neo4j, EXISTING_DOCUMENTS, ids=ids, tenant=tenant)
    found = {r["id"] for r in rows}
    for missing in ids:
        if missing not in found:
            report.violations.append(Violation(
                "DOC-SUPERSEDES-001", "document", document_key, f"supersedes {missing}"))
    return report
