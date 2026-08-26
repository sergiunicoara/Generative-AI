"""Regression tests for source-grounded intelligence ingestion additions."""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from graphrag.core.models import (
    Chunk,
    IngestionRunManifest,
    IntelligenceArtifact,
    StructuredTable,
)
from graphrag.ingestion.document_loader import load_structured_tables
from graphrag.ingestion.intelligence import (
    IntelligenceArtifactExtractor,
    expand_temporal_query,
    mine_explicit_aliases,
    temporal_periods,
)


def test_explicit_alias_mining_requires_both_extracted_names():
    aliases = mine_explicit_aliases(
        "MOUNJARO (tirzepatide) was approved. See Table 2 for dosage.",
        ["MOUNJARO", "tirzepatide"],
    )
    assert aliases == [("MOUNJARO", "tirzepatide", "parenthetical", "MOUNJARO (tirzepatide)")]
    assert mine_explicit_aliases("See Table (2).", ["Table"]) == []


def test_temporal_periods_build_calendar_parent_chain():
    periods = {item["value"]: item for item in temporal_periods("The event occurred on 2024-01-15 in January 2024.")}
    assert periods["2024-01-15"]["parent"] == "2024-01"
    assert periods["2024-01"]["parent"] == "2024-Q1"
    assert periods["2024-Q1"]["parent"] == "2024"
    assert periods["2024"]["kind"] == "year"
    assert expand_temporal_query("What changed in January 2024?").endswith("2024-Q1")


@pytest.mark.asyncio
async def test_intelligence_extractor_rejects_non_verbatim_evidence():
    chunk = Chunk(document_id="doc-1", id="chunk-1", chunk_index=0, tenant="acme", text="Acme acquired Beta on 2024-01-15.")
    payload = {
        "artifacts": [
            {
                "type": "EVENT",
                "text": "Acme acquired Beta.",
                "evidence_quote": "Acme acquired Beta on 2024-01-15.",
                "confidence": 0.99,
                "entity_names": ["Acme", "Beta"],
                "event_start": "2024-01-15",
                "event_end": "",
            },
            {
                "type": "FINDING",
                "text": "The deal will dominate the market.",
                "evidence_quote": "The deal will dominate the market.",
                "confidence": 0.9,
                "entity_names": ["Acme"],
                "event_start": "",
                "event_end": "",
            },
        ]
    }
    fake_llm = type("LLM", (), {"generate": AsyncMock(return_value=json.dumps(payload))})()
    with patch("graphrag.ingestion.intelligence.get_llm", return_value=fake_llm):
        artifacts = await IntelligenceArtifactExtractor("test-model").extract(chunk, ["Acme", "Beta"])

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.artifact_type == "EVENT"
    assert artifact.entity_names == ["Acme", "Beta"]
    assert artifact.event_start == datetime(2024, 1, 15, tzinfo=timezone.utc)


def test_ingestion_manifest_hash_changes_for_material_stage_change():
    manifest = IngestionRunManifest(job_id="job", tenant="acme", filename="a.txt", content_hash="abc")
    first = manifest.compute_integrity_hash()
    manifest.stage_metrics = {"extraction": {"items": 3, "cost_usd": None}}
    assert manifest.compute_integrity_hash() != first


def test_structured_table_jsonld_and_excel_loader():
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Revenue"
    sheet.append(["Quarter", "Revenue"])
    sheet.append(["Q1", 100])
    buffer = io.BytesIO()
    workbook.save(buffer)

    tables = load_structured_tables("report.xlsx", buffer.getvalue(), document_id="doc-1", tenant="acme")
    assert len(tables) == 1
    assert tables[0].columns == ["Quarter", "Revenue"]
    assert tables[0].as_jsonld()["@type"] == "schema:Table"


@pytest.mark.asyncio
async def test_writer_persists_new_artifact_table_and_manifest_contracts():
    from graphrag.ingestion.graph_writer import GraphWriter

    client = type("Client", (), {
        "merge_intelligence_artifacts": AsyncMock(),
        "merge_structured_tables": AsyncMock(),
        "upsert_ingestion_manifest": AsyncMock(),
        "merge_temporal_periods": AsyncMock(),
    })()
    writer = GraphWriter.__new__(GraphWriter)
    writer._neo4j = client
    writer._ensure_registry = AsyncMock()
    writer._get_registry = lambda _tenant: type(
        "Registry", (), {"resolve": staticmethod(lambda name: ("Acme Corporation", "ORG") if name == "Acme" else None)}
    )()

    chunk = Chunk(id="chunk", document_id="doc", text="source", chunk_index=0, tenant="acme")
    artifact = IntelligenceArtifact(
        artifact_type="OBSERVATION", text="source", evidence_quote="source",
        source_chunk_id="chunk", source_doc_id="doc", tenant="acme", entity_names=["Acme"],
    )
    table = StructuredTable(document_id="doc", table_index=0, columns=["a"], rows=[["b"]], tenant="acme")
    manifest = IngestionRunManifest(job_id="job", tenant="acme", filename="a.txt", document_id="doc")

    await writer.write_intelligence_artifacts([artifact], chunk)
    await writer.write_structured_tables([table], "acme")
    await writer.write_ingestion_manifest(manifest)
    await writer.write_temporal_periods(chunk, [{"value": "2024", "kind": "year", "parent": ""}])

    client.merge_intelligence_artifacts.assert_awaited_once_with([artifact], tenant="acme")
    assert artifact.entity_names == ["Acme Corporation"]
    client.merge_structured_tables.assert_awaited_once_with([table], tenant="acme")
    client.upsert_ingestion_manifest.assert_awaited_once_with(manifest)
    client.merge_temporal_periods.assert_awaited_once()
