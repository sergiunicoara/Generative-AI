"""Unit tests for the GLEIF LEI federation stub — see
ontology/mappings/gleif-lei.r2rml.ttl's header comment for scope.

Follows tests/unit/test_r2rml_obda.py's exact conventions.
"""

from pathlib import Path

import pytest

from graphrag.ingestion.r2rml import (
    FederatedOBDAIngestor, FederatedOBDASource, r2rml_to_mapping,
)

_GLEIF_MAPPING = Path(__file__).resolve().parents[2] / "ontology/mappings/gleif-lei.r2rml.ttl"
_SUPPLY_CHAIN_MAPPING = Path(__file__).resolve().parents[2] / "ontology/mappings/supply-chain.r2rml.ttl"


def test_gleif_mapping_parses_legal_entity_record():
    mapping = r2rml_to_mapping(
        _GLEIF_MAPPING, mapping_id="gleif-lei", version="1", source_id="gleif", tenant="acme",
    )

    assert [(item.table, item.entity_type) for item in mapping.entities] == [
        ("gleif_lei_extract", "LEGALENTITYRECORD"),
    ]
    assert mapping.entities[0].id_column == "lei_code"
    assert mapping.entities[0].name_column == "legal_name"
    # No R2RML-native join by design -- cross-source reconciliation is an
    # entity-resolution-time concern, not a mapping-time one. See the
    # mapping file's header comment.
    assert mapping.relations == []


class _Ingestor:
    """Fake ingestor stub -- mirrors test_r2rml_obda.py's federation test.
    Proves FederatedOBDAIngestor drives both sources, without needing a
    real SQLite connector for either one."""

    def __init__(self):
        self.ingested = False

    async def validate(self, mapping):
        return type("Report", (), {"valid": True, "errors": []})()

    async def ingest(self, mapping):
        self.ingested = True


@pytest.mark.asyncio
async def test_gleif_and_supplier_sources_federate_under_one_tenant():
    supplier_mapping = r2rml_to_mapping(
        _SUPPLY_CHAIN_MAPPING, mapping_id="supply-chain", version="1",
        source_id="erp", tenant="acme",
    )
    gleif_mapping = r2rml_to_mapping(
        _GLEIF_MAPPING, mapping_id="gleif-lei", version="1",
        source_id="gleif", tenant="acme",
    )

    supplier_ingestor = _Ingestor()
    gleif_ingestor = _Ingestor()
    federation = FederatedOBDAIngestor([
        FederatedOBDASource("suppliers", supplier_ingestor, supplier_mapping),
        FederatedOBDASource("gleif", gleif_ingestor, gleif_mapping),
    ])

    reports = await federation.validate()
    assert all(report.valid for report in reports)

    await federation.ingest()
    assert supplier_ingestor.ingested
    assert gleif_ingestor.ingested


@pytest.mark.asyncio
async def test_gleif_source_rejects_cross_tenant_mixing():
    supplier_mapping = r2rml_to_mapping(
        _SUPPLY_CHAIN_MAPPING, mapping_id="supply-chain", version="1",
        source_id="erp", tenant="acme",
    )
    gleif_mapping = r2rml_to_mapping(
        _GLEIF_MAPPING, mapping_id="gleif-lei", version="1",
        source_id="gleif", tenant="other-tenant",
    )

    federation = FederatedOBDAIngestor([
        FederatedOBDASource("suppliers", _Ingestor(), supplier_mapping),
        FederatedOBDASource("gleif", _Ingestor(), gleif_mapping),
    ])

    with pytest.raises(ValueError, match="one tenant"):
        await federation.ingest()
