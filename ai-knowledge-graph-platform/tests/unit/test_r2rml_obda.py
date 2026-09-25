from pathlib import Path

import pytest

from rdflib import Graph, Literal, URIRef

from graphrag.ingestion.r2rml import (
    FederatedOBDAIngestor, FederatedOBDASource, R2RMLMappingError, _identifier_from_template,
    _local_name, _one, r2rml_to_mapping,
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

    mapping = r2rml_to_mapping(
        source, mapping_id="r2rml", version="1", source_id="erp", tenant="acme",
        ontology_version="r2rml/v2",
    )

    assert [(item.table, item.entity_type) for item in mapping.entities] == [("suppliers", "SUPPLIER"), ("orders", "ORDER")]
    supplier, order = mapping.entities
    assert (supplier.id_column, supplier.name_column) == ("id", "name")
    assert (order.id_column, order.name_column) == ("id", "name")
    assert mapping.relations[0].source_table == "orders"
    assert mapping.relations[0].target_table == "suppliers"
    assert mapping.relations[0].source_column == "id"      # orders.id_column
    assert mapping.relations[0].target_column == "supplier_id"  # rr:child
    assert mapping.relations[0].relation == "ORDEREDFROM"
    assert (mapping.id, mapping.version, mapping.source_id, mapping.tenant) == ("r2rml", "1", "erp", "acme")
    assert mapping.ontology_version == "r2rml/v2"


def test_r2rml_adapter_defaults_ontology_version_when_not_given(tmp_path):
    source = _ttl(tmp_path, """
        @prefix rr: <http://www.w3.org/ns/r2rml#> . @prefix ex: <https://example.test/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        ex:X a rr:TriplesMap; rr:logicalTable [rr:tableName "x"];
          rr:subjectMap [rr:template "https://x/{id}"; rr:class ex:X];
          rr:predicateObjectMap [rr:predicate rdfs:label; rr:objectMap [rr:column "name"]].
    """)

    mapping = r2rml_to_mapping(source, mapping_id="x", version="1", source_id="x", tenant="acme")

    assert mapping.ontology_version == "r2rml/v1"


def test_r2rml_adapter_rejects_a_source_with_no_triples_map(tmp_path):
    source = _ttl(tmp_path, "@prefix ex: <https://example.test/> . ex:X ex:unrelated ex:Y .")

    with pytest.raises(R2RMLMappingError, match="no rr:TriplesMap found"):
        r2rml_to_mapping(source, mapping_id="x", version="1", source_id="x", tenant="acme")


def test_r2rml_adapter_rejects_a_relation_with_more_than_one_predicate(tmp_path):
    source = _ttl(tmp_path, """
        @prefix rr: <http://www.w3.org/ns/r2rml#> . @prefix ex: <https://example.test/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        ex:Supplier a rr:TriplesMap; rr:logicalTable [rr:tableName "suppliers"];
          rr:subjectMap [rr:template "https://x/s/{id}"; rr:class ex:Supplier];
          rr:predicateObjectMap [rr:predicate rdfs:label; rr:objectMap [rr:column "name"]].
        ex:Order a rr:TriplesMap; rr:logicalTable [rr:tableName "orders"];
          rr:subjectMap [rr:template "https://x/o/{id}"; rr:class ex:Order];
          rr:predicateObjectMap [rr:predicate rdfs:label; rr:objectMap [rr:column "name"]];
          rr:predicateObjectMap [rr:predicate ex:orderedFrom, ex:alsoOrderedFrom; rr:objectMap [
            rr:parentTriplesMap ex:Supplier; rr:joinCondition [rr:child "supplier_id"; rr:parent "id"]]].
    """)

    with pytest.raises(R2RMLMappingError, match="one predicate and one object map"):
        r2rml_to_mapping(source, mapping_id="x", version="1", source_id="x", tenant="acme")


def test_r2rml_adapter_rejects_a_parent_triples_map_not_defined_locally(tmp_path):
    source = _ttl(tmp_path, """
        @prefix rr: <http://www.w3.org/ns/r2rml#> . @prefix ex: <https://example.test/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        ex:Order a rr:TriplesMap; rr:logicalTable [rr:tableName "orders"];
          rr:subjectMap [rr:template "https://x/o/{id}"; rr:class ex:Order];
          rr:predicateObjectMap [rr:predicate rdfs:label; rr:objectMap [rr:column "name"]];
          rr:predicateObjectMap [rr:predicate ex:orderedFrom; rr:objectMap [
            rr:parentTriplesMap ex:NotDefinedHere; rr:joinCondition [rr:child "supplier_id"; rr:parent "id"]]].
    """)

    with pytest.raises(R2RMLMappingError, match="must reference a local rr:TriplesMap"):
        r2rml_to_mapping(source, mapping_id="x", version="1", source_id="x", tenant="acme")


def test_r2rml_adapter_rejects_a_join_whose_parent_column_is_not_the_identifier(tmp_path):
    source = _ttl(tmp_path, """
        @prefix rr: <http://www.w3.org/ns/r2rml#> . @prefix ex: <https://example.test/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        ex:Supplier a rr:TriplesMap; rr:logicalTable [rr:tableName "suppliers"];
          rr:subjectMap [rr:template "https://x/s/{id}"; rr:class ex:Supplier];
          rr:predicateObjectMap [rr:predicate rdfs:label; rr:objectMap [rr:column "name"]].
        ex:Order a rr:TriplesMap; rr:logicalTable [rr:tableName "orders"];
          rr:subjectMap [rr:template "https://x/o/{id}"; rr:class ex:Order];
          rr:predicateObjectMap [rr:predicate rdfs:label; rr:objectMap [rr:column "name"]];
          rr:predicateObjectMap [rr:predicate ex:orderedFrom; rr:objectMap [
            rr:parentTriplesMap ex:Supplier; rr:joinCondition [rr:child "supplier_id"; rr:parent "not_id"]]].
    """)

    with pytest.raises(R2RMLMappingError, match="must be the parent subject identifier"):
        r2rml_to_mapping(source, mapping_id="x", version="1", source_id="x", tenant="acme")


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


def test_federated_obda_ingestor_requires_at_least_one_source():
    with pytest.raises(ValueError, match="at least one federated OBDA source is required"):
        FederatedOBDAIngestor([])


@pytest.mark.asyncio
async def test_federated_obda_ingestor_rejects_duplicate_source_ids():
    mapping = RelationalGraphMapping(
        id="a", version="1", source_id="dup", tenant="acme",
        entities=[{"table": "items", "entity_type": "ITEM", "id_column": "id", "name_column": "name"}],
    )

    class Ingestor:
        async def validate(self, mapping):
            return type("Report", (), {"valid": True, "errors": []})()

    federation = FederatedOBDAIngestor([
        FederatedOBDASource("one", Ingestor(), mapping),
        FederatedOBDASource("two", Ingestor(), mapping),
    ])
    with pytest.raises(ValueError, match="source_id values must be unique"):
        await federation.validate()


@pytest.mark.asyncio
async def test_federated_obda_ingestor_surfaces_validation_report_errors():
    mapping = RelationalGraphMapping(
        id="a", version="1", source_id="a", tenant="acme",
        entities=[{"table": "items", "entity_type": "ITEM", "id_column": "id", "name_column": "name"}],
    )

    class Ingestor:
        async def validate(self, mapping):
            return type("Report", (), {"valid": False, "errors": ["column x missing"]})()

    federation = FederatedOBDAIngestor([FederatedOBDASource("one", Ingestor(), mapping)])
    with pytest.raises(ValueError, match="column x missing"):
        await federation.validate()


class TestSmallR2RMLHelpers:
    """Direct unit coverage for r2rml.py's small internal parsing helpers —
    otherwise only exercised indirectly through full-mapping fixtures above,
    which tolerate many mutations of these functions without ever hitting
    their edge cases (see docs/REMAINING_AUDIT_BACKLOG.md item #8's
    mutation-campaign follow-up)."""

    def test_one_returns_the_single_value(self):
        g = Graph()
        s, p = URIRef("urn:s"), URIRef("urn:p")
        g.add((s, p, Literal("only")))

        assert _one(g, s, p, "thing") == Literal("only")

    def test_one_raises_when_no_value(self):
        g = Graph()
        s, p = URIRef("urn:s"), URIRef("urn:p")

        with pytest.raises(R2RMLMappingError, match="thing must occur exactly once"):
            _one(g, s, p, "thing")

    def test_one_raises_when_multiple_values(self):
        g = Graph()
        s, p = URIRef("urn:s"), URIRef("urn:p")
        g.add((s, p, Literal("a")))
        g.add((s, p, Literal("b")))

        with pytest.raises(R2RMLMappingError, match="thing must occur exactly once"):
            _one(g, s, p, "thing")

    def test_local_name_takes_the_last_path_segment(self):
        assert _local_name(URIRef("https://example.test/ns/Widget")) == "WIDGET"

    def test_local_name_takes_the_fragment_when_present(self):
        assert _local_name(URIRef("https://example.test/ns#Widget")) == "WIDGET"

    def test_local_name_takes_the_last_slash_segment_within_the_fragment(self):
        assert _local_name(URIRef("https://example.test/ns#group/Widget")) == "WIDGET"

    def test_local_name_normalizes_non_alnum_runs_to_underscore(self):
        assert _local_name(URIRef("https://example.test/ns#Work Order-2")) == "WORK_ORDER_2"

    def test_local_name_raises_when_nothing_survives_normalization(self):
        with pytest.raises(R2RMLMappingError, match="cannot derive a safe local name"):
            _local_name(URIRef("https://example.test/ns#---"))

    def test_identifier_from_template_extracts_the_column_name(self):
        assert _identifier_from_template("ex:item/{item_id}", "subject template") == "item_id"

    def test_identifier_from_template_rejects_a_template_without_a_placeholder(self):
        with pytest.raises(R2RMLMappingError, match="exactly one identifier template"):
            _identifier_from_template("ex:item/static", "subject template")
