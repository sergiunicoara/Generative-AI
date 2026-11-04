"""Data expectations on relational builds, per-build lineage, and the Atlas /
OpenLineage catalog export (offline: SQLite fixtures, fakes, mocked HTTP)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import sqlite3
import uuid

import httpx
import jsonschema
import pytest

from graphrag.context_graph.definitions import BusinessDefinition
from graphrag.enterprise.catalog_export import (
    ATLAS_TERM_SCHEMA,
    OPENLINEAGE_RUN_EVENT_SCHEMA,
    AtlasClient,
    AtlasConfig,
    OpenLineageClient,
    atlas_terms,
    run_event,
)
from graphrag.enterprise.models import AccessContext, ACLState, DocumentAccessPolicy
from graphrag.ingestion.build_ledger import BuildLedger, BuildRecord, table_digest
from graphrag.ingestion.expectations import Expectation, ExpectationsFailed, check_expectations
from graphrag.ingestion.relational import RelationalGraphIngestor, SQLiteSourceConnector
from tests.unit.test_relational_incremental import _AssertNotCalledWriter, _db, _mapping, _RecordingWriter

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
FORMATS = jsonschema.FormatChecker()


# -- expectations ----------------------------------------------------------------------------


def _rows():
    return [{"id": "s1", "name": "One", "updated": "2026-11-04T08:00:00Z"},
            {"id": "s2", "name": "", "updated": "2026-11-01T08:00:00Z"},
            {"id": "s2", "name": "Two", "updated": None}]


@pytest.mark.parametrize("expectation,passed,observed", [
    ({"kind": "schema", "columns": ["id", "name"]}, True, []),
    ({"kind": "schema", "columns": ["id", "country"]}, False, ["country"]),
    ({"kind": "not_null", "column": "name"}, False, 1),
    ({"kind": "unique", "column": "id"}, False, 1),
    ({"kind": "unique", "column": "name"}, True, 0),
    ({"kind": "freshness", "column": "updated", "max_age_hours": 6}, True, 4.0),
    ({"kind": "freshness", "column": "updated", "max_age_hours": 2}, False, 4.0),
    ({"kind": "row_count", "min_rows": 1, "max_rows": 2}, False, 3),
    ({"kind": "row_count", "min_rows": 3}, True, 3),
])
def test_each_expectation_kind(expectation, passed, observed):
    [result] = check_expectations([Expectation(id="e1", table="suppliers", **expectation)],
                                  {"suppliers": _rows()}, now=NOW)
    assert (result.passed, result.observed) == (passed, observed), result.message


def test_an_unread_table_and_malformed_declarations_fail():
    [result] = check_expectations([Expectation(id="e", table="nope", kind="row_count", min_rows=1)], {}, now=NOW)
    assert not result.passed and "not read" in result.message
    with pytest.raises(ValueError):
        Expectation(id="e", table="t", kind="freshness", column="updated")
    with pytest.raises(ValueError):
        Expectation(id="e", table="t", kind="row_count", min_rows=5, max_rows=1)


# -- builds --------------------------------------------------------------------------------


class _Ledger:
    def __init__(self):
        self.records: list[dict] = []

    async def record(self, build):
        self.records.append(build.as_dict())


def _expecting(*expectations):
    return _mapping().model_copy(update={"expectations": [Expectation(**e) for e in expectations]})


async def test_a_failed_expectation_blocks_the_build_before_any_write(tmp_path):
    ledger = _Ledger()
    ingestor = RelationalGraphIngestor(SQLiteSourceConnector(_db(tmp_path)), _AssertNotCalledWriter(),
                                       build_ledger=ledger, code_version="abc123")
    mapping = _expecting({"id": "two-suppliers", "table": "suppliers", "kind": "row_count", "min_rows": 2},
                         {"id": "ids-unique", "table": "suppliers", "kind": "unique", "column": "id"})
    with pytest.raises(ExpectationsFailed, match="two-suppliers") as raised:
        await ingestor.ingest(mapping)
    assert [r.passed for r in raised.value.results] == [False, True]
    [record] = ledger.records
    assert record["outcome"] == "blocked" and record["code_version"] == "abc123"
    assert {i["table"] for i in record["inputs"]} == {"suppliers", "materials", "supplies"}
    assert record["build_id"] == raised.value.build_id and record["document_id"] == ""


async def test_a_passing_build_records_inputs_results_and_outputs(tmp_path):
    ledger = _Ledger()
    writer = _RecordingWriter()
    ingestor = RelationalGraphIngestor(SQLiteSourceConnector(_db(tmp_path)), writer, build_ledger=ledger,
                                       code_version="abc123")
    report = await ingestor.ingest(_expecting({"id": "ids", "table": "suppliers", "kind": "not_null", "column": "id"}))
    [record] = ledger.records
    assert record["outcome"] == "published" and record["document_id"] == writer.documents[0].id
    assert record["outputs"] == {"entities": 2, "relations": 1}
    assert report.build_id == record["build_id"] and report.expectations[0]["passed"] is True
    supplies = next(i for i in record["inputs"] if i["table"] == "supplies")
    assert supplies["rows"] == 1 and len(supplies["sha256"]) == 64
    again = await ingestor.ingest(_mapping(), previous_hash=report.content_hash)
    assert again.skipped and ledger.records[-1]["outcome"] == "skipped"


async def test_a_failure_after_the_expectations_is_recorded_as_failed(tmp_path):
    ledger = _Ledger()
    path = _db(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE supplies SET confidence = 1.5")
    with pytest.raises(ValueError, match="SHACL"):
        await RelationalGraphIngestor(SQLiteSourceConnector(path), _AssertNotCalledWriter(),
                                      build_ledger=ledger).ingest(_mapping())
    assert ledger.records[0]["outcome"] == "failed" and "SHACL" in ledger.records[0]["error"]


async def test_without_a_ledger_expectations_still_apply(tmp_path):
    ingestor = RelationalGraphIngestor(SQLiteSourceConnector(_db(tmp_path)), _AssertNotCalledWriter())
    with pytest.raises(ExpectationsFailed):
        await ingestor.ingest(_expecting({"id": "big", "table": "materials", "kind": "row_count", "min_rows": 10}))


async def test_the_ledger_links_published_builds_and_resolves_lineage():
    calls = []

    class Neo4j:
        async def run(self, cypher, **params):
            calls.append((cypher, params))
            if "RETURN DISTINCT b.id" in cypher:
                return [{"build_id": "b-1", "source_id": "supplier-db", "mapping_version": "1.0.0",
                         "code_version": "abc", "finished_at": "2026-11-04T10:00:00+00:00",
                         "inputs_json": '[{"table": "suppliers", "rows": 1}]', "document_id": "d-1"}]
            return []

    ledger = BuildLedger(Neo4j())
    build = BuildRecord.start(tenant="acme", source_id="supplier-db", source_uri="sqlite://x", mapping_id="m",
                              mapping_version="1.0.0", version="abc").finish("published", document_id="d-1")
    await ledger.record(build)
    assert "MERGE (b)-[:PUBLISHED]->(d)" in calls[0][0] and calls[0][1]["outcome"] == "published"
    lineage = await ledger.lineage("acme", "e-1")
    assert lineage["build_id"] == "b-1" and lineage["inputs"] == [{"table": "suppliers", "rows": 1}]
    assert all(params["tenant"] == "acme" for _, params in calls)
    assert "MENTIONS" in calls[1][0] and "PART_OF" in calls[1][0] and "PUBLISHED" in calls[1][0]


def test_table_digest_ignores_row_order():
    rows = [{"a": 1}, {"a": 2}]
    assert table_digest(rows) == table_digest(list(reversed(rows))) != table_digest([{"a": 3}])


# -- OpenLineage -------------------------------------------------------------------------------


def _build(outcome="published"):
    build = BuildRecord.start(tenant="acme", source_id="supplier-db", source_uri="sqlite:///supply.db",
                              mapping_id="supply-chain", mapping_version="1.0.0", version="abc")
    build.read({"suppliers": [{"id": "s1"}], "supplies": [{"supplier_id": "s1"}]})
    build.expectations = [{"id": "ids", "table": "suppliers", "kind": "not_null", "passed": outcome != "blocked",
                           "observed": 0, "message": "ok", "column": "id"}]
    return build.finish(outcome, outputs={"entities": 2, "relations": 1} if outcome == "published" else {},
                        error="" if outcome == "published" else "data expectations failed: ids")


def test_a_run_event_names_exactly_what_the_build_read_and_wrote():
    event = run_event(_build())
    jsonschema.validate(event, OPENLINEAGE_RUN_EVENT_SCHEMA, format_checker=FORMATS)
    assert uuid.UUID(event["run"]["runId"])
    assert event["eventType"] == "COMPLETE"
    assert [(d["namespace"], d["name"]) for d in event["inputs"]] == [("sqlite:///supply.db", "suppliers"),
                                                                     ("sqlite:///supply.db", "supplies")]
    assert [(d["namespace"], d["name"]) for d in event["outputs"]] == [("graphrag://acme", "graph/supplier-db")]
    assertions = event["inputs"][0]["inputFacets"]["dataQualityAssertions"]["assertions"]
    assert assertions == [{"assertion": "not_null:ids", "success": True, "column": "id"}]


def test_a_blocked_build_is_a_failed_run_without_outputs():
    event = run_event(_build("blocked"))
    jsonschema.validate(event, OPENLINEAGE_RUN_EVENT_SCHEMA, format_checker=FORMATS)
    assert event["eventType"] == "FAIL" and event["outputs"] == []
    assert "expectations failed" in event["run"]["facets"]["errorMessage"]["message"]


async def test_the_openlineage_client_posts_the_event():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201)

    client = OpenLineageClient("https://marquez.example/api/v1/lineage", api_key="k",
                               client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await client.emit(run_event(_build()))
    assert seen[0].headers["authorization"] == "Bearer k" and json.loads(seen[0].content)["eventType"] == "COMPLETE"


# -- Atlas ------------------------------------------------------------------------------------


def _definition(id_, version, term, *, status="published", access=None, depends_on=()):
    return BusinessDefinition.model_validate({
        "id": id_, "version": version, "tenant": "acme", "term": term, "owner": "quality-office",
        "definition": f"{term} as agreed (v{version}).", "precision": "loose", "status": status,
        "depends_on": list(depends_on),
        "sources": [{"source_id": "doc-1", "locator": "p1"}],
        "recorded_at": f"2026-11-0{version}T09:00:00+00:00",
        **({"access": access} if access else {}),
    })


def _definitions():
    restricted = DocumentAccessPolicy(mode="restricted", state=ACLState.KNOWN, allow_principals=["group:legal"])
    return [
        _definition("DEF-ACTIVE-SUPPLIER", 1, "Active supplier"),
        _definition("DEF-ACTIVE-SUPPLIER", 2, "Active supplier"),
        _definition("DEF-LATE", 1, "Late delivery", depends_on=("DEF-ACTIVE-SUPPLIER",)),
        _definition("DEF-PENALTY", 1, "Contract penalty", access=restricted.model_dump()),
        _definition("DEF-DRAFT", 1, "Draft term", status="draft"),
    ]


def test_terms_cover_visible_published_latest_definitions_only():
    plain = atlas_terms(_definitions(), tenant="acme", access=AccessContext.from_claims({"sub": "u"}),
                        glossary_guid="g-1")
    for term in plain:
        jsonschema.validate(term, ATLAS_TERM_SCHEMA)
    assert sorted(t["name"] for t in plain) == ["Active supplier", "Late delivery"]
    active = next(t for t in plain if t["name"] == "Active supplier")
    assert active["additionalAttributes"]["graphrag.version"] == "2"
    legal = atlas_terms(_definitions(), tenant="acme", access=AccessContext.from_claims(
        {"sub": "u", "groups": ["legal"]}), glossary_guid="g-1")
    assert "Contract penalty" in {t["name"] for t in legal}
    assert atlas_terms(_definitions(), tenant="other", access=AccessContext.from_claims({"sub": "u"}),
                       glossary_guid="g-1") == []


async def test_a_new_version_updates_the_existing_term_and_dependencies_become_links():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        if request.method == "GET":
            return httpx.Response(200, json=[{"guid": "t-active", "name": "Active supplier"}])
        if request.method == "POST":
            return httpx.Response(200, json={"guid": "t-late"})
        return httpx.Response(200, json={"guid": request.url.path.rsplit("/", 1)[-1]})

    terms = atlas_terms(_definitions(), tenant="acme", access=AccessContext.from_claims({"sub": "u"}),
                        glossary_guid="g-1")
    client = AtlasClient(AtlasConfig("https://atlas.example", "svc", "pw", "g-1"),
                         httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    result = await client.upsert_terms(terms)
    assert result == {"created": ["Late delivery"], "updated": ["Active supplier"], "linked": ["Late delivery"]}
    assert ("PUT", "/api/atlas/v2/glossary/term/t-active") in [(m, p) for m, p, _ in calls]
    assert [m for m, *_ in calls].count("POST") == 1, "no duplicate term for a new version"
    link = calls[-1]
    assert link[1] == "/api/atlas/v2/glossary/term/t-late" and link[2]["seeAlso"] == [{"termGuid": "t-active"}]


def test_unsafe_urls_are_refused():
    with pytest.raises(ValueError):
        AtlasConfig("file:///etc", "u", "p", "g")
    with pytest.raises(ValueError):
        OpenLineageClient("ftp://x")


def test_builds_and_events_use_utc_iso_times():
    build = _build()
    assert datetime.fromisoformat(build.finished_at).tzinfo is not None
    assert datetime.fromisoformat(build.finished_at) - datetime.fromisoformat(build.started_at) < timedelta(seconds=5)
