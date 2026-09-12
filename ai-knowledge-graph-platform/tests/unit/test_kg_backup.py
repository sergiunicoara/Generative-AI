"""Storage URI routing for the graph backup CLI."""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location("kg_backup", ROOT / "scripts" / "kg_backup.py")
assert SPEC and SPEC.loader
kg_backup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kg_backup)


def test_gcs_uri_is_parsed_and_routed_as_remote():
    uri = "gs://graphrag-backups/tenant-a/graph.ndjson"
    assert kg_backup._is_gcs(uri) is True
    assert kg_backup._is_remote(uri) is True
    assert kg_backup._parse_gcs(uri) == ("graphrag-backups", "tenant-a/graph.ndjson")


def test_s3_and_local_uri_routing_is_unchanged():
    assert kg_backup._is_remote("s3://bucket/graph.ndjson") is True
    assert kg_backup._is_remote("backups/graph.ndjson") is False
    assert kg_backup._parse_s3("s3://bucket/graph.ndjson") == ("bucket", "graph.ndjson")


def test_gcs_writes_use_an_in_memory_buffer_before_upload():
    buf = kg_backup._open_write("gs://bucket/graph.ndjson")
    kg_backup._write_ndjson_line(buf, {"_type": "meta", "tenant": "tenant-a"})
    assert '"tenant": "tenant-a"' in buf.getvalue()


class _RecordingNeo4j:
    """Small deterministic stand-in for the backup tool's three read queries."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def run(self, cypher: str, **params):
        self.calls.append((cypher, params))
        if "MATCH (e:Entity" in cypher:
            return [{
                "name": "WT-01", "type": "WIND_TURBINE", "tenant": "tenant-a",
                "description": "demo", "confidence": 0.9,
                "valid_from": "2026-01-01T00:00:00Z",
                "valid_to": "2026-12-31T00:00:00Z", "wikidata_qid": None,
                "quarantined": True,
            }]
        if "MATCH (s:Entity" in cypher:
            return [{
                "src": "WT-01", "src_type": "WIND_TURBINE", "tgt": "WO-9001",
                "tgt_type": "WORK_ORDER", "relation": "HAS_OPEN_WORK_ORDER",
                "confidence": 0.95, "source_doc_ids": ["SAP-WO-9001"],
                "valid_from": "2026-08-28T00:00:00Z",
                "valid_to": "2026-12-31T00:00:00Z",
                "source_type": "synthetic_sap_export", "tenant": "tenant-a",
            }]
        if "MATCH (c:Chunk" in cypher:
            return []
        return []


async def test_backup_restore_preserves_exported_temporal_and_quarantine_fields(tmp_path, monkeypatch):
    """Regression guard for fields that were exported but silently dropped on restore."""
    from graphrag.graph import neo4j_client

    client = _RecordingNeo4j()
    monkeypatch.setattr(neo4j_client, "get_neo4j", lambda: client)
    backup = tmp_path / "backup.ndjson"

    await kg_backup.do_backup(Namespace(tenant="tenant-a", output=str(backup)))
    await kg_backup.do_restore(Namespace(input=str(backup), tenant=""))

    entity_restore = next(params for cypher, params in client.calls if "MERGE (e:Entity" in cypher)
    relation_restore = next(params for cypher, params in client.calls if "MERGE (s)-[r:RELATES_TO" in cypher)
    assert entity_restore["valid_to"] == "2026-12-31T00:00:00Z"
    assert entity_restore["quarantined"] is True
    assert relation_restore["valid_to"] == "2026-12-31T00:00:00Z"
    assert relation_restore["source_doc_ids"] == ["SAP-WO-9001"]
