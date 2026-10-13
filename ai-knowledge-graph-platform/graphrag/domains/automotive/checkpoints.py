"""Signed checkpoints of the IATF sign-off chain head, kept outside Neo4j.

The hash chain (``signoff.py``) detects edits, but someone with database write access can rewrite the whole chain and
recompute every hash. A checkpoint records ``(tenant, seq, head)`` at a point in time, signed with an HMAC key the
database does not hold, in an append-only JSON-lines file. Verification then also requires that the chain still has
the checkpointed hash at the checkpointed sequence number, so a rewrite of history before the latest checkpoint is
detected unless the attacker also holds the key *and* can rewrite the checkpoint file.

The key comes from the environment (``IATF_CHECKPOINT_KEY``), never from config or the repository. Point the
checkpoint path at storage the database administrator cannot write (a separate host, a WORM/object-lock bucket
mount); on the same disk as the database it only raises the bar.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KEY_ENV = "IATF_CHECKPOINT_KEY"
_SIGNED = ("tenant", "seq", "head", "created_at", "key_id")


def _key(key: bytes | None) -> bytes:
    key = key if key is not None else os.environ.get(KEY_ENV, "").encode("utf-8")
    if len(key) < 32:
        raise ValueError(f"{KEY_ENV} must be set to at least 32 bytes")
    return key


def key_id(key: bytes) -> str:
    """Public identifier of a key (so a rotated key is visible in old checkpoints), not the key."""
    return hashlib.sha256(b"iatf-checkpoint-key-id:" + key).hexdigest()[:16]


def _signature(record: dict[str, Any], key: bytes) -> str:
    material = json.dumps({k: record.get(k) for k in _SIGNED}, sort_keys=True).encode("utf-8")
    return hmac.new(key, material, hashlib.sha256).hexdigest()


def make_checkpoint(*, tenant: str, seq: int, head: str, key: bytes | None = None,
                    now: datetime | None = None) -> dict[str, Any]:
    k = _key(key)
    record = {"tenant": tenant, "seq": int(seq), "head": head,
              "created_at": (now or datetime.now(timezone.utc)).isoformat(), "key_id": key_id(k)}
    record["signature"] = _signature(record, k)
    return record


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:          # append-only by convention; enforce with storage policy
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def read_checkpoints(path: Path, tenant: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("tenant") == tenant:
                out.append(record)
    return out


def verify_against_checkpoints(entries: list[dict[str, Any]], checkpoints: list[dict[str, Any]], *,
                               key: bytes | None = None) -> dict[str, Any]:
    """Every checkpoint must carry a valid signature, and the chain must still hold its head at its seq."""
    k = _key(key)
    by_seq = {int(e["seq"]): e for e in entries}
    problems = []
    for cp in checkpoints:
        if not hmac.compare_digest(_signature(cp, k), str(cp.get("signature", ""))):
            problems.append({"seq": cp.get("seq"), "problem": "bad signature (wrong key or edited checkpoint)"})
            continue
        if int(cp["seq"]) == 0:
            continue
        entry = by_seq.get(int(cp["seq"]))
        if entry is None:
            problems.append({"seq": cp["seq"], "problem": "chain no longer has this sequence number"})
        elif entry.get("hash") != cp["head"]:
            problems.append({"seq": cp["seq"], "problem": "chain hash differs from the checkpoint (history rewritten)"})
    latest = max((int(c["seq"]) for c in checkpoints), default=None)
    return {"valid": not problems, "checkpoints": len(checkpoints), "latest_checkpoint_seq": latest,
            "problems": problems}


__all__ = ["KEY_ENV", "append_checkpoint", "key_id", "make_checkpoint", "read_checkpoints",
           "verify_against_checkpoints"]
