"""Ingestion-gap detection: pure diff, read-only query, tenant scoping."""
from __future__ import annotations

import pytest

from graphrag.enterprise.gaps import BUCKETS, GRAPH_DOCUMENTS, diff_inventory, find_ingestion_gaps, normalise_inventory
from graphrag.graph.validation.graph_checks import assert_read_only


def _row(ext, version="v1", deleted=False, status="done", chunks=3):
    return {"external_id": ext, "source_version": version, "is_deleted": deleted, "status": status, "chunks": chunks}


def test_each_gap_kind_is_detected_and_only_those():
    inventory = {"ok": "v1", "new": "v1", "changed": "v2", "back": "v1", "half": "v1", "unk": "v1"}
    graph = [_row("ok"), _row("changed", "v1"), _row("back", deleted=True), _row("half", chunks=0),
             _row("unk", version=""), _row("gone")]
    report = diff_inventory(inventory, graph)
    assert report["counts"] == {"missing": 1, "stale": 1, "restored_in_source": 1, "not_ingested": 1,
                                "duplicate_in_graph": 0, "unverifiable_version": 1, "would_tombstone": 1}
    assert report["samples"]["missing"] == ["new"] and report["samples"]["stale"] == ["changed"]
    assert report["repair_ids"] == ["back", "changed", "half", "new"]
    assert report["in_sync"] == 2 and report["coverage"] == pytest.approx(2 / 6, abs=1e-4)
    assert set(report["counts"]) == set(BUCKETS)


def test_status_not_done_counts_as_not_ingested_even_with_chunks():
    report = diff_inventory({"a": "v1"}, [_row("a", status="processing", chunks=5)])
    assert report["counts"]["not_ingested"] == 1 and report["coverage"] == 0.0


def test_bare_id_list_cannot_judge_staleness_but_still_finds_missing():
    report = diff_inventory(["a", "b"], [_row("a", version="anything")])
    assert report["counts"]["stale"] == 0 and report["counts"]["missing"] == 1 and report["in_sync"] == 1


def test_duplicate_graph_documents_are_flagged_and_a_live_one_is_enough():
    report = diff_inventory({"a": "v1"}, [_row("a", deleted=True), _row("a")])
    assert report["counts"]["duplicate_in_graph"] == 1
    assert report["counts"]["restored_in_source"] == 0 and report["in_sync"] == 1


def test_deleted_graph_documents_absent_from_source_are_not_tombstone_candidates():
    report = diff_inventory({"a": "v1"}, [_row("a"), _row("old", deleted=True)])
    assert report["counts"]["would_tombstone"] == 0


def test_empty_inventory_and_empty_graph():
    assert diff_inventory({}, [])["coverage"] is None
    report = diff_inventory({}, [_row("x")])
    assert report["counts"]["would_tombstone"] == 1 and report["repair_ids"] == []


def test_samples_are_capped_but_repair_ids_are_complete():
    inventory = {f"id{i:03d}": "v1" for i in range(60)}
    report = diff_inventory(inventory, [], sample=10)
    assert len(report["samples"]["missing"]) == 10 and len(report["repair_ids"]) == 60


def test_normalise_inventory_drops_blank_ids_and_none_versions():
    assert normalise_inventory({"a": None, "": "x"}) == {"a": ""}
    assert normalise_inventory(["a", "", "b"]) == {"a": "", "b": ""}


def test_graph_query_is_read_only_and_tenant_scoped():
    assert_read_only(GRAPH_DOCUMENTS)
    assert "tenant: $tenant, source_id: $source_id" in GRAPH_DOCUMENTS
    assert "Chunk {tenant: $tenant}" in GRAPH_DOCUMENTS


class _FakeNeo4j:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    async def run_read(self, cypher, **params):
        self.calls.append((cypher, params))
        return self.rows


async def test_find_ingestion_gaps_runs_one_read_query_with_the_tenant():
    neo = _FakeNeo4j([_row("a")])
    report = await find_ingestion_gaps(neo, tenant="t1", source_id="s1", inventory={"a": "v1", "b": "v1"})
    assert len(neo.calls) == 1 and neo.calls[0][1]["tenant"] == "t1" and neo.calls[0][1]["source_id"] == "s1"
    assert report["counts"]["missing"] == 1 and report["tenant"] == "t1" and report["source_id"] == "s1"
