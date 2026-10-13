"""Auditor sign-off on IATF clause evidence, stored as a tamper-evident hash chain per tenant.

Each sign-off records who decided what about which clause, when, and a digest of the exact evidence they saw
(``audit.evidence_digest``). Entries are append-only; each one's ``hash`` covers its own fields plus the previous
entry's hash, so editing or deleting any stored entry breaks ``verify_chain`` from that point on.

Concurrency: the append is one Cypher statement that only succeeds when the tenant's chain head still equals the hash
the writer read. A concurrent append makes it return nothing and the caller retries with the new head, so the chain
never forks.

This is tamper-*evident*, not tamper-*proof*: anyone with database write access can rewrite the whole chain. For an
external guarantee, periodically publish the head hash somewhere the database admin cannot change (see docs).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

DECISIONS = ("accepted", "rejected", "needs_more_evidence")
GENESIS = "0" * 64
_FIELDS = ("id", "tenant", "seq", "clause", "decision", "reviewer", "note", "evidence_digest", "evidence_as_of",
           "signed_at", "prev_hash")

APPEND = """
MERGE (h:IatfSignoffHead {tenant: $tenant})
  ON CREATE SET h.hash = $genesis, h.seq = 0
// Take the head's write lock BEFORE comparing: a concurrent appender blocks here and then sees the committed head,
// so it fails the check below instead of forking the chain.
SET h.locked_at = datetime()
WITH h
WHERE h.hash = $prev_hash AND h.seq = $prev_seq
CREATE (s:IatfSignoff {id: $id, tenant: $tenant, seq: $seq, clause: $clause, decision: $decision,
                       reviewer: $reviewer, note: $note, evidence_digest: $evidence_digest,
                       evidence_as_of: $evidence_as_of, signed_at: $signed_at, prev_hash: $prev_hash, hash: $hash})
SET h.hash = $hash, h.seq = $seq
RETURN s.id AS id
"""

SCHEMA = (
    "CREATE CONSTRAINT iatf_signoff_seq IF NOT EXISTS FOR (s:IatfSignoff) REQUIRE (s.tenant, s.seq) IS UNIQUE",
    "CREATE CONSTRAINT iatf_signoff_head IF NOT EXISTS FOR (h:IatfSignoffHead) REQUIRE h.tenant IS UNIQUE",
)

HEAD = """
OPTIONAL MATCH (h:IatfSignoffHead {tenant: $tenant})
RETURN coalesce(h.hash, $genesis) AS hash, coalesce(h.seq, 0) AS seq
"""

LIST = """
MATCH (s:IatfSignoff {tenant: $tenant})
WHERE $clause IS NULL OR s.clause = $clause
RETURN s {.*} AS entry
ORDER BY s.seq
"""


class ChainConflict(RuntimeError):
    """The chain head moved while appending, more times than the retry budget allows."""


def entry_hash(entry: dict[str, Any]) -> str:
    material = {k: entry.get(k) for k in _FIELDS}
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def verify_chain(entries: list[dict[str, Any]], head_hash: str | None = None) -> dict[str, Any]:
    """Recompute every hash in sequence order. Reports the first broken link, if any."""
    prev, expected_seq = GENESIS, 1
    for entry in sorted(entries, key=lambda e: int(e.get("seq") or 0)):
        seq = int(entry.get("seq") or 0)
        if seq != expected_seq:
            return {"valid": False, "entries": len(entries), "broken_at_seq": expected_seq,
                    "reason": f"sequence gap: expected {expected_seq}, found {seq}"}
        if entry.get("prev_hash") != prev:
            return {"valid": False, "entries": len(entries), "broken_at_seq": seq, "reason": "prev_hash mismatch"}
        if entry_hash(entry) != entry.get("hash"):
            return {"valid": False, "entries": len(entries), "broken_at_seq": seq, "reason": "content changed"}
        prev, expected_seq = entry["hash"], seq + 1
    if head_hash is not None and head_hash != prev:
        return {"valid": False, "entries": len(entries), "broken_at_seq": expected_seq,
                "reason": "head does not match the last entry (entries removed from the end?)"}
    return {"valid": True, "entries": len(entries), "head": prev}


class SignoffStore:
    _schema_ready = False

    def __init__(self, neo4j, *, max_retries: int = 5):
        self._neo4j = neo4j
        self._max_retries = max_retries

    async def ensure_schema(self) -> None:
        for statement in SCHEMA:
            await self._neo4j.run(statement)
        SignoffStore._schema_ready = True

    async def append(self, *, tenant: str, clause: str, decision: str, reviewer: str, evidence_digest: str,
                     evidence_as_of: str | None = None, note: str = "") -> dict[str, Any]:
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        if not reviewer:
            raise ValueError("an authenticated reviewer identity is required")
        if not tenant or not clause or not evidence_digest:
            raise ValueError("tenant, clause and evidence_digest are required")
        if not SignoffStore._schema_ready:
            await self.ensure_schema()
        for _ in range(self._max_retries):
            head = (await self._neo4j.run(HEAD, tenant=tenant, genesis=GENESIS))[0]
            entry = {"id": str(uuid4()), "tenant": tenant, "seq": int(head["seq"]) + 1, "clause": clause,
                     "decision": decision, "reviewer": reviewer, "note": note[:2000],
                     "evidence_digest": evidence_digest, "evidence_as_of": evidence_as_of,
                     "signed_at": datetime.now(timezone.utc).isoformat(), "prev_hash": head["hash"]}
            entry["hash"] = entry_hash(entry)
            rows = await self._neo4j.run(APPEND, genesis=GENESIS, prev_seq=int(head["seq"]), **entry)
            if rows:
                return entry
        raise ChainConflict("sign-off chain head kept moving; retry later")

    async def entries(self, tenant: str, clause: str | None = None) -> list[dict[str, Any]]:
        rows = await self._neo4j.run(LIST, tenant=tenant, clause=clause)
        return [dict(r["entry"]) for r in rows]

    async def verify(self, tenant: str) -> dict[str, Any]:
        head = (await self._neo4j.run(HEAD, tenant=tenant, genesis=GENESIS))[0]
        return verify_chain(await self.entries(tenant), head_hash=head["hash"])

    async def latest_by_clause(self, tenant: str) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for entry in await self.entries(tenant):
            latest[entry["clause"]] = entry
        return latest


__all__ = ["DECISIONS", "GENESIS", "ChainConflict", "SignoffStore", "entry_hash", "verify_chain", "SCHEMA"]
