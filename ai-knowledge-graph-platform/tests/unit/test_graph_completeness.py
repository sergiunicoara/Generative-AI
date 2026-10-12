"""Graph completeness report: measures, thresholds, read-only tenant-scoped queries."""
from __future__ import annotations

import pytest

from graphrag.graph.completeness import MEASURES, QUERIES, graph_completeness_report, summarize
from graphrag.graph.validation.graph_checks import assert_read_only

FULL = {
    "documents": {"documents": 10, "ingested": 10, "with_chunks": 10, "with_entities": 10, "with_external_id": 10,
                  "with_version": 10, "with_ingested_at": 10},
    "chunks": {"chunks": 40, "embedded": 40, "with_text": 40, "with_entities": 40},
    "entities": {"entities": 20, "with_description": 20, "embedded": 20, "typed": 20, "with_evidence": 20,
                 "connected": 20},
    "relations": {"relations": 30, "with_relation": 30, "with_confidence": 30, "with_source": 30,
                  "with_valid_from": 30},
}


def test_complete_graph_has_every_share_at_one_and_nothing_incomplete():
    report = summarize(FULL)
    assert all(m["share"] == 1.0 for m in report["measures"].values())
    assert report["incomplete"] == [] and report["sizes"] == {"documents": 10, "chunks": 40, "entities": 20,
                                                              "relations": 30}


def test_incomplete_measures_are_listed_lowest_first_with_numerators():
    counts = {k: dict(v) for k, v in FULL.items()}
    counts["chunks"]["embedded"] = 10             # 0.25
    counts["entities"]["with_description"] = 15   # 0.75
    counts["relations"]["with_source"] = 29       # 0.9667, above the 0.95 default
    report = summarize(counts)
    assert [m["measure"] for m in report["incomplete"]] == ["chunks_embedded", "entities_described"]
    assert report["measures"]["chunks_embedded"] == {"share": 0.25, "numerator": 10, "denominator": 40}
    assert summarize(counts, incomplete_below=0.99)["incomplete"][-1]["measure"] == "relations_sourced"


def test_zero_denominators_give_none_not_a_misleading_one():
    report = summarize({})
    assert all(m["share"] is None for m in report["measures"].values()) and report["incomplete"] == []


def test_every_measure_names_a_section_and_keys_that_the_queries_return():
    for section, name, num_key, den_key in MEASURES:
        cypher = QUERIES[section]
        assert f"AS {num_key}" in cypher and f"AS {den_key}" in cypher, name


@pytest.mark.parametrize("section", sorted(QUERIES))
def test_queries_are_read_only_and_tenant_scoped(section):
    assert_read_only(QUERIES[section])
    assert "$tenant" in QUERIES[section]


def test_quarantined_entities_and_deleted_documents_are_excluded():
    assert "quarantined" in QUERIES["entities"]
    assert "is_deleted" in QUERIES["documents"] and "is_deleted" in QUERIES["chunks"]
    assert "confidence_state" in QUERIES["relations"]      # current-edge predicate, shared with correctness


class _FakeNeo4j:
    def __init__(self):
        self.calls = []

    async def run_read(self, cypher, **params):
        self.calls.append(params)
        for section, query in QUERIES.items():
            if query == cypher:
                return [FULL[section]]
        raise AssertionError("unexpected query")


async def test_report_runs_four_read_queries_and_passes_tenant_and_source():
    neo = _FakeNeo4j()
    report = await graph_completeness_report(neo, "t1", "src")
    assert len(neo.calls) == 4 and all(c["tenant"] == "t1" and c["source_id"] == "src" for c in neo.calls)
    assert report["tenant"] == "t1" and report["source_id"] == "src" and report["incomplete"] == []
