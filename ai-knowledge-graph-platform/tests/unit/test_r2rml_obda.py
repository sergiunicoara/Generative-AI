from pathlib import Path

import pytest

from graphrag.ingestion.r2rml import (
    FederatedOBDAIngestor, FederatedOBDASource, R2RMLMappingError, r2rml_to_mapping,
)
from graphrag.ingestion.relational import RelationalGraphMapping


def _ttl(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "mapping.ttl"
    path.write_text(text, encoding="utf-8")
    return path


def test_r2rml_adapter_materializes_subjects_and_parent_join(tmp_path):
    source = _ttl(tmp_path, """
        @prefix rr: <http://www.w3.org/ns/r2rml#> .
        @prefix ex: <https://example.test/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        ex:Supplier a rr:TriplesMap; rr:logicalTable [rr:tableName "suppliers"];
          rr:subjectMap [rr:template "https://x/s/{id}"; rr:class ex:Supplier];
          rr:predicateObjectMap [rr:predicate rdfs:label; rr:objectMap [rr:column "name"]].
        ex:Order a rr:TriplesMap; rr:logicalTable [rr:tableName "orders"];
          rr:subjectMap [rr:template "https://x/o/{id}"; rr:class ex:Order];
          rr:predicateObjectMap [rr:predicate rdfs:label; rr:objectMap [rr:column "name"]];
          rr:predicateObjectMap [rr:predicate ex:orderedFrom; rr:objectMap [
            rr:parentTriplesMap ex:Supplier; rr:joinCondition [rr:child "supplier_id"; rr:parent "id"]]].
    """)

    mapping = r2rml_to_mapping(source, mapping_id="r2rml", version="1", source_id="erp", tenant="acme")

    assert [(item.table, item.entity_type) for item in mapping.entities] == [("suppliers", "SUPPLIER"), ("orders", "ORDER")]
    assert mapping.relations[0].source_table == "orders"
    assert mapping.relations[0].target_table == "suppliers"
    assert mapping.relations[0].relation == "ORDEREDFROM"


def test_r2rml_adapter_rejects_unlabelled_subjects(tmp_path):
    source = _ttl(tmp_path, """
        @prefix rr: <http://www.w3.org/ns/r2rml#> . @prefix ex: <https://example.test/> .
        ex:X a rr:TriplesMap; rr:logicalTable [rr:tableName "x"];
        rr:subjectMap [rr:template "https://x/{id}"; rr:class ex:X].
    """)
    with pytest.raises(R2RMLMappingError, match="rdfs:label"):
        r2rml_to_mapping(source, mapping_id="x", version="1", source_id="x", tenant="acme")


@pytest.mark.asyncio
async def test_federation_preflights_all_sources_before_any_ingest():
    valid = RelationalGraphMapping(
        id="a", version="1", source_id="a", tenant="acme",
        entities=[{"table": "items", "entity_type": "ITEM", "id_column": "id", "name_column": "name"}],
    )
    invalid = valid.model_copy(update={"id": "b", "source_id": "b", "tenant": "other"})

    class Ingestor:
        async def validate(self, mapping):
            return type("Report", (), {"valid": True, "errors": []})()

        async def ingest(self, mapping):
            raise AssertionError("must not ingest an invalid federation")

    federation = FederatedOBDAIngestor([
        FederatedOBDASource("one", Ingestor(), valid),
        FederatedOBDASource("two", Ingestor(), invalid),
    ])
    with pytest.raises(ValueError, match="one tenant"):
        await federation.ingest()
