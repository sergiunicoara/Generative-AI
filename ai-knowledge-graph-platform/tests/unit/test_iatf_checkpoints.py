"""Signed checkpoints anchoring the IATF sign-off chain head outside the database."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from graphrag.domains.automotive import checkpoints as cp
from graphrag.domains.automotive import signoff as so

KEY = b"k" * 32
NOW = datetime(2026, 10, 13, tzinfo=timezone.utc)


def _chain(n, decision="accepted"):
    entries, prev = [], so.GENESIS
    for seq in range(1, n + 1):
        e = {"id": f"id{seq}", "tenant": "t", "seq": seq, "clause": "8.4", "decision": decision, "reviewer": "r",
             "note": "", "evidence_digest": "d", "evidence_as_of": None, "signed_at": "s", "prev_hash": prev}
        e["hash"] = so.entry_hash(e)
        entries.append(e)
        prev = e["hash"]
    return entries


def test_key_must_be_long_enough_and_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv(cp.KEY_ENV, raising=False)
    with pytest.raises(ValueError, match=cp.KEY_ENV):
        cp.make_checkpoint(tenant="t", seq=1, head="h")
    monkeypatch.setenv(cp.KEY_ENV, "x" * 32)
    assert cp.make_checkpoint(tenant="t", seq=1, head="h", now=NOW)["key_id"] == cp.key_id(b"x" * 32)


def test_intact_chain_passes_and_checkpoints_are_per_tenant(tmp_path):
    chain = _chain(3)
    path = tmp_path / "cp.jsonl"
    cp.append_checkpoint(path, cp.make_checkpoint(tenant="t", seq=2, head=chain[1]["hash"], key=KEY, now=NOW))
    cp.append_checkpoint(path, cp.make_checkpoint(tenant="other", seq=9, head="x", key=KEY, now=NOW))
    records = cp.read_checkpoints(path, "t")
    assert len(records) == 1
    result = cp.verify_against_checkpoints(chain, records, key=KEY)
    assert result == {"valid": True, "checkpoints": 1, "latest_checkpoint_seq": 2, "problems": []}


def test_a_fully_rehashed_rewrite_is_caught_by_the_checkpoint():
    """The attack the chain alone cannot see: edit an entry and recompute every later hash."""
    original = _chain(3)
    checkpoint = cp.make_checkpoint(tenant="t", seq=3, head=original[-1]["hash"], key=KEY, now=NOW)
    rewritten = _chain(3, decision="rejected")
    assert so.verify_chain(rewritten)["valid"] is True
    result = cp.verify_against_checkpoints(rewritten, [checkpoint], key=KEY)
    assert result["valid"] is False and "rewritten" in result["problems"][0]["problem"]


def test_truncation_below_a_checkpoint_and_edited_or_foreign_key_checkpoints_fail():
    chain = _chain(3)
    good = cp.make_checkpoint(tenant="t", seq=3, head=chain[-1]["hash"], key=KEY, now=NOW)
    assert "no longer has" in cp.verify_against_checkpoints(chain[:2], [good], key=KEY)["problems"][0]["problem"]
    edited = {**good, "head": "f" * 64}
    assert "bad signature" in cp.verify_against_checkpoints(chain, [edited], key=KEY)["problems"][0]["problem"]
    other_key = cp.make_checkpoint(tenant="t", seq=3, head=chain[-1]["hash"], key=b"z" * 32, now=NOW)
    assert cp.verify_against_checkpoints(chain, [other_key], key=KEY)["valid"] is False


def test_empty_chain_checkpoint_and_no_checkpoints():
    zero = cp.make_checkpoint(tenant="t", seq=0, head=so.GENESIS, key=KEY, now=NOW)
    assert cp.verify_against_checkpoints([], [zero], key=KEY)["valid"] is True
    assert cp.verify_against_checkpoints(_chain(1), [], key=KEY)["latest_checkpoint_seq"] is None


async def test_script_refuses_to_checkpoint_a_broken_chain_and_verifies_a_good_one(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import iatf_checkpoint as script

    monkeypatch.setenv(cp.KEY_ENV, KEY.decode())
    chain = _chain(2)

    class Neo:
        def __init__(self, entries):
            self.entries = entries

        async def run(self, cypher, **p):
            if cypher is so.HEAD:
                last = self.entries[-1] if self.entries else None
                return [{"hash": last["hash"] if last else so.GENESIS, "seq": last["seq"] if last else 0}]
            if cypher is so.LIST:
                return [{"entry": e} for e in self.entries]
            raise AssertionError(cypher)

    path = tmp_path / "cp.jsonl"
    code, report = await script.run("write", "t", path, Neo(chain))
    assert code == 0 and report["checkpoint"]["seq"] == 2
    code, report = await script.run("verify", "t", path, Neo(chain))
    assert code == 0 and report["checkpoints"]["valid"] is True
    broken = [dict(chain[0], decision="rejected"), chain[1]]
    code, report = await script.run("write", "t", path, Neo(broken))
    assert code == 1 and report["written"] is False
    code, _ = await script.run("verify", "t", path, Neo(_chain(2, decision="rejected")))
    assert code == 1
