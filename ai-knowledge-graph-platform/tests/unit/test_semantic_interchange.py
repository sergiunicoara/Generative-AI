"""Regression tests for JSON-LD, Excel mappings, SharePoint and LLM SHACL gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openpyxl import Workbook
from rdflib import Graph

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from graphrag.enterprise.models import ACLState
from graphrag.enterprise.sharepoint import SharePointSourceConfig, SharePointSyncConnector, _access_policy
from graphrag.ingestion.document_loader import load_document_content
from graphrag.ingestion.relational import (
    EntityTableMapping,
    ExcelWorkbookConnector,
    RelationalGraphIngestor,
    RelationalGraphMapping,
)


@pytest.mark.asyncio
async def test_excel_uses_existing_semantic_mapping_contract(tmp_path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "parties"
    sheet.append(["id", "name", "description"])
    sheet.append(["p-1", "Acme Ltd", "Contract party"])
    path = tmp_path / "contracts.xlsx"
    workbook.save(path)

    connector = ExcelWorkbookConnector(path)
    mapping = RelationalGraphMapping(
        id="contract-parties", version="1.0.0", source_id="contracts-xlsx", tenant="legal",
        entities=[EntityTableMapping(table="parties", entity_type="LEGAL_PARTY", id_column="id", name_column="name")],
    )

    assert await connector.read_table("parties") == [{"id": "p-1", "name": "Acme Ltd", "description": "Contract party"}]
    report = await RelationalGraphIngestor(connector, object()).validate(mapping)
    assert report.valid is True


def test_json_and_excel_content_are_readable_without_local_temp_files():
    assert '"agreement"' in load_document_content("contract.json", json.dumps({"agreement": "A-1"}).encode())
    workbook = Workbook()
    workbook.active.append(["id", "name"])
    workbook.active.append(["1", "Acme"])
    from io import BytesIO
    data = BytesIO()
    workbook.save(data)
    assert "Acme" in load_document_content("parties.xlsx", data.getvalue())


@pytest.mark.asyncio
async def test_jsonld_export_is_parseable_by_standard_rdflib(tmp_path):
    from export_rdf import export

    output = tmp_path / "graph.jsonld"
    neo4j = MagicMock()
    neo4j.run = AsyncMock(side_effect=[[], [], [], [], []])
    neo4j.close = AsyncMock()
    with patch("graphrag.graph.neo4j_client.get_neo4j", return_value=neo4j):
        await export(tenant="legal", output=output, limit=10, rdf_format="json-ld")
    parsed = Graph().parse(output, format="json-ld")
    assert len(parsed) > 0


@pytest.mark.asyncio
async def test_sharepoint_delta_maps_permissions_and_persists_cursor():
    class GraphClient:
        async def delta(self, cursor):
            assert cursor == "old-delta"
            return ([
                {"id": "item-1", "name": "agreement.txt", "webUrl": "https://sharepoint/item-1", "eTag": "v2", "file": {"mimeType": "text/plain"}},
                {"id": "item-2", "deleted": {}},
            ], "new-delta")

        async def content(self, item_id):
            assert item_id == "item-1"
            return b"Agreement text"

        async def permissions(self, item_id):
            return [{"grantedToV2": {"group": {"id": "legal-team"}}}]

    sync = MagicMock()
    sync.current_cursor = AsyncMock(return_value="old-delta")
    sync.apply_changes = AsyncMock(return_value={"queued": 1, "tombstoned": 1, "cursor": "new-delta"})
    config = SharePointSourceConfig("legal-sharepoint", "directory", "client", "SECRET_ENV", "site", "drive", "legal")

    result = await SharePointSyncConnector(config, graph_client=GraphClient(), sync_service=sync).sync_once()

    assert result["received"] == 2
    changes = sync.apply_changes.await_args.args[1]
    assert changes[0].access_policy.allow_principals == ["group:legal-team"]
    assert changes[1].change_type.value == "delete"
    assert sync.apply_changes.await_args.kwargs["cursor"] == "new-delta"


def test_sharepoint_unresolvable_links_fail_closed():
    policy = _access_policy([{"link": {"scope": "anonymous"}}])
    assert policy.state == ACLState.UNKNOWN


@pytest.mark.asyncio
async def test_llm_extraction_is_stopped_by_the_prewrite_shacl_gate():
    from graphrag.core.models import Chunk
    from graphrag.ingestion.extractor import Extractor

    extractor = Extractor.__new__(Extractor)
    extractor._model_name = "test-model"
    extractor._entity_types = ["ORG"]
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=json.dumps({
        "entities": [{"name": "Acme", "type": "ORG"}], "relations": [],
    }))
    registry = MagicMock()
    registry.is_loaded = True
    registry.validate_extraction.return_value = {
        "rejected_entity_ids": [], "rejected_relation_ids": [], "drift_detected": False,
    }
    registry.record_schema_event = AsyncMock()
    report = SimpleNamespace(conforms=False, text="missing tenant", counts={"violations": 1})

    with (
        patch("graphrag.ingestion.extractor.get_llm", return_value=llm),
        patch("graphrag.graph.ontology_registry.get_ontology_registry", return_value=registry),
        patch("graphrag.graph.shacl_validator.SHACLValidator.validate_relational_batch_report", return_value=report),
    ):
        entities, relations = await extractor.extract(Chunk(document_id="doc-1", text="Acme", chunk_index=0))

    assert entities == [] and relations == []
    registry.record_schema_event.assert_awaited_once()
    assert registry.record_schema_event.await_args.kwargs["event_type"] == "extraction_shacl_rejected"
