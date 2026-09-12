"""Regression tests proving graphrag/ingestion/rml_rdf.py's materialize_rml()
executes a real RML mapping (JSON source) into real RDF triples.

Before this module, ontology/mappings/energy-observations.rml.ttl's own
comment admitted nothing executed it: it had no predicate-object maps at
all (subject map only), and no RML processor or JSONPath library exists
anywhere in this repo. These tests exercise the completed, shipped mapping
end to end against its real JSON fixture, plus fail-closed coverage for
every construct this deliberately narrow executor does not support.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdflib import Literal, URIRef
from rdflib.namespace import RDF, XSD

from graphrag.ingestion.rml_rdf import RMLMappingError, materialize_rml

ROOT = Path(__file__).resolve().parents[2]
OBSERVATIONS_MAPPING = ROOT / "ontology" / "mappings" / "energy-observations.rml.ttl"

ENERGY_NS = "https://example.energy.demo/ontology#"
PROV_NS = "http://www.w3.org/ns/prov#"


class TestMaterializeEnergyObservations:
    async def test_observation_triples_are_materialized_from_the_real_fixture(self):
        graph = await materialize_rml(OBSERVATIONS_MAPPING, ROOT)

        obs = URIRef("https://example.energy.demo/record/SNOW-OBS-WT-01")
        assert (obs, RDF.type, URIRef(ENERGY_NS + "Observation")) in graph
        assert (obs, URIRef(ENERGY_NS + "observedAsset"), URIRef("https://example.energy.demo/asset/WT-01")) in graph
        assert (obs, URIRef(ENERGY_NS + "metric"), Literal("temperature_c")) in graph
        assert (obs, URIRef(ENERGY_NS + "value"), Literal(96.0, datatype=XSD.decimal)) in graph
        assert (obs, URIRef(ENERGY_NS + "unit"), Literal("C")) in graph
        assert (obs, URIRef(ENERGY_NS + "observedAt"), Literal("2026-08-28T08:00:00Z", datatype=XSD.dateTime)) in graph
        assert (obs, URIRef(PROV_NS + "wasDerivedFrom"), URIRef("urn:synthetic:snowflake:telemetry")) in graph

    async def test_every_fixture_record_is_materialized(self):
        graph = await materialize_rml(OBSERVATIONS_MAPPING, ROOT)
        observations = set(graph.subjects(RDF.type, URIRef(ENERGY_NS + "Observation")))
        assert observations == {
            URIRef("https://example.energy.demo/record/SNOW-OBS-WT-01"),
            URIRef("https://example.energy.demo/record/SNOW-OBS-WT-02"),
            URIRef("https://example.energy.demo/record/SNOW-OBS-WT-03"),
        }

    async def test_templated_object_map_builds_the_same_iri_shape_the_r2rml_mapping_uses(self):
        """energy:observedAsset must resolve to the identical asset IRI
        pattern materialize_r2rml() builds from energy-assets.r2rml.ttl, so
        a SPARQL join between the two graphs actually connects."""
        graph = await materialize_rml(OBSERVATIONS_MAPPING, ROOT)
        obs = URIRef("https://example.energy.demo/record/SNOW-OBS-WT-02")
        assert (obs, URIRef(ENERGY_NS + "observedAsset"), URIRef("https://example.energy.demo/asset/WT-02")) in graph


class TestUnsupportedConstructsFailClosed:
    @pytest.fixture
    def json_fixture(self, tmp_path: Path) -> Path:
        source = tmp_path / "records.json"
        source.write_text(json.dumps([{"id": "1", "name": "x"}]), encoding="utf-8")
        return source

    def _write_mapping(self, tmp_path: Path, body: str) -> Path:
        mapping = tmp_path / "bad.rml.ttl"
        mapping.write_text(
            "@prefix rml: <http://semweb.mmlab.be/ns/rml#> .\n"
            "@prefix ql: <http://semweb.mmlab.be/ns/ql#> .\n"
            "@prefix rr: <http://www.w3.org/ns/r2rml#> .\n"
            "@prefix energy: <https://example.energy.demo/ontology#> .\n"
            f"{body}\n",
            encoding="utf-8",
        )
        return mapping

    async def test_non_jsonpath_formulation_is_rejected(self, tmp_path, json_fixture):
        mapping = self._write_mapping(tmp_path, """
            energy:Map a rr:TriplesMap;
              rml:logicalSource [ rml:source "records.json"; rml:referenceFormulation rml:XPath; rml:iterator "$[*]" ];
              rr:subjectMap [ rr:template "https://example.org/{id}"; rr:class energy:Thing ].
        """)
        with pytest.raises(RMLMappingError, match="ql:JSONPath"):
            await materialize_rml(mapping, tmp_path)

    async def test_unsupported_iterator_is_rejected(self, tmp_path, json_fixture):
        mapping = self._write_mapping(tmp_path, """
            energy:Map a rr:TriplesMap;
              rml:logicalSource [ rml:source "records.json"; rml:referenceFormulation ql:JSONPath; rml:iterator "$.items[*]" ];
              rr:subjectMap [ rr:template "https://example.org/{id}"; rr:class energy:Thing ].
        """)
        with pytest.raises(RMLMappingError, match="iterator"):
            await materialize_rml(mapping, tmp_path)

    async def test_missing_json_source_is_rejected(self, tmp_path):
        mapping = self._write_mapping(tmp_path, """
            energy:Map a rr:TriplesMap;
              rml:logicalSource [ rml:source "does-not-exist.json"; rml:referenceFormulation ql:JSONPath; rml:iterator "$[*]" ];
              rr:subjectMap [ rr:template "https://example.org/{id}"; rr:class energy:Thing ].
        """)
        with pytest.raises(RMLMappingError, match="not found"):
            await materialize_rml(mapping, tmp_path)

    async def test_rr_column_is_rejected_as_a_tabular_only_construct(self, tmp_path, json_fixture):
        mapping = self._write_mapping(tmp_path, """
            energy:Map a rr:TriplesMap;
              rml:logicalSource [ rml:source "records.json"; rml:referenceFormulation ql:JSONPath; rml:iterator "$[*]" ];
              rr:subjectMap [ rr:template "https://example.org/{id}"; rr:class energy:Thing ];
              rr:predicateObjectMap [ rr:predicate energy:name; rr:objectMap [ rr:column "name" ] ].
        """)
        with pytest.raises(RMLMappingError, match="rr:column"):
            await materialize_rml(mapping, tmp_path)

    async def test_parent_triples_map_join_is_rejected(self, tmp_path, json_fixture):
        mapping = self._write_mapping(tmp_path, """
            energy:Parent a rr:TriplesMap;
              rml:logicalSource [ rml:source "records.json"; rml:referenceFormulation ql:JSONPath; rml:iterator "$[*]" ];
              rr:subjectMap [ rr:template "https://example.org/{id}"; rr:class energy:Thing ].
            energy:Map a rr:TriplesMap;
              rml:logicalSource [ rml:source "records.json"; rml:referenceFormulation ql:JSONPath; rml:iterator "$[*]" ];
              rr:subjectMap [ rr:template "https://example.org/child/{id}"; rr:class energy:Thing ];
              rr:predicateObjectMap [
                rr:predicate energy:parent;
                rr:objectMap [ rr:parentTriplesMap energy:Parent; rr:joinCondition [ rr:child "id"; rr:parent "id" ] ]
              ].
        """)
        with pytest.raises(RMLMappingError, match="rr:parentTriplesMap"):
            await materialize_rml(mapping, tmp_path)

    async def test_object_map_with_no_recognized_shape_is_rejected(self, tmp_path, json_fixture):
        mapping = self._write_mapping(tmp_path, """
            energy:Map a rr:TriplesMap;
              rml:logicalSource [ rml:source "records.json"; rml:referenceFormulation ql:JSONPath; rml:iterator "$[*]" ];
              rr:subjectMap [ rr:template "https://example.org/{id}"; rr:class energy:Thing ];
              rr:predicateObjectMap [ rr:predicate energy:name; rr:objectMap [ rr:language "en" ] ].
        """)
        with pytest.raises(RMLMappingError):
            await materialize_rml(mapping, tmp_path)

    async def test_non_array_json_source_is_rejected(self, tmp_path):
        source = tmp_path / "not-an-array.json"
        source.write_text(json.dumps({"id": "1"}), encoding="utf-8")
        mapping = self._write_mapping(tmp_path, """
            energy:Map a rr:TriplesMap;
              rml:logicalSource [ rml:source "not-an-array.json"; rml:referenceFormulation ql:JSONPath; rml:iterator "$[*]" ];
              rr:subjectMap [ rr:template "https://example.org/{id}"; rr:class energy:Thing ].
        """)
        with pytest.raises(RMLMappingError, match="array"):
            await materialize_rml(mapping, tmp_path)

    async def test_multi_column_template_is_rejected(self, tmp_path, json_fixture):
        mapping = self._write_mapping(tmp_path, """
            energy:Map a rr:TriplesMap;
              rml:logicalSource [ rml:source "records.json"; rml:referenceFormulation ql:JSONPath; rml:iterator "$[*]" ];
              rr:subjectMap [ rr:template "https://example.org/{id}/{name}"; rr:class energy:Thing ].
        """)
        with pytest.raises(RMLMappingError, match="exactly one"):
            await materialize_rml(mapping, tmp_path)

    async def test_item_missing_the_identifier_key_is_rejected(self, tmp_path):
        source = tmp_path / "records.json"
        source.write_text(json.dumps([{"other": "x"}]), encoding="utf-8")
        mapping = self._write_mapping(tmp_path, """
            energy:Map a rr:TriplesMap;
              rml:logicalSource [ rml:source "records.json"; rml:referenceFormulation ql:JSONPath; rml:iterator "$[*]" ];
              rr:subjectMap [ rr:template "https://example.org/{id}"; rr:class energy:Thing ].
        """)
        with pytest.raises(RMLMappingError, match="missing identifier"):
            await materialize_rml(mapping, tmp_path)


class TestSupportedObjectMapShapes:
    async def test_reference_template_and_constant_all_work_together(self, tmp_path):
        source = tmp_path / "records.json"
        source.write_text(json.dumps([{"id": "1", "parent_id": "p1", "name": "Alice"}]), encoding="utf-8")
        mapping = tmp_path / "ok.rml.ttl"
        mapping.write_text(
            "@prefix rml: <http://semweb.mmlab.be/ns/rml#> .\n"
            "@prefix ql: <http://semweb.mmlab.be/ns/ql#> .\n"
            "@prefix rr: <http://www.w3.org/ns/r2rml#> .\n"
            "@prefix energy: <https://example.energy.demo/ontology#> .\n"
            "energy:Map a rr:TriplesMap;\n"
            '  rml:logicalSource [ rml:source "records.json"; rml:referenceFormulation ql:JSONPath; rml:iterator "$[*]" ];\n'
            '  rr:subjectMap [ rr:template "https://example.org/{id}"; rr:class energy:Thing ];\n'
            '  rr:predicateObjectMap [ rr:predicate energy:name; rr:objectMap [ rml:reference "name" ] ];\n'
            '  rr:predicateObjectMap [ rr:predicate energy:parent; rr:objectMap [ rr:template "https://example.org/{parent_id}" ] ];\n'
            "  rr:predicateObjectMap [ rr:predicate energy:source; rr:objectMap [ rr:constant <urn:fixed:source> ] ].\n",
            encoding="utf-8",
        )
        graph = await materialize_rml(mapping, tmp_path)
        subject = URIRef("https://example.org/1")
        assert (subject, URIRef(ENERGY_NS + "name"), Literal("Alice")) in graph
        assert (subject, URIRef(ENERGY_NS + "parent"), URIRef("https://example.org/p1")) in graph
        assert (subject, URIRef(ENERGY_NS + "source"), URIRef("urn:fixed:source")) in graph

    async def test_reference_without_datatype_produces_a_plain_literal(self, tmp_path):
        source = tmp_path / "records.json"
        source.write_text(json.dumps([{"id": "1", "name": "Alice"}]), encoding="utf-8")
        mapping = tmp_path / "ok.rml.ttl"
        mapping.write_text(
            "@prefix rml: <http://semweb.mmlab.be/ns/rml#> .\n"
            "@prefix ql: <http://semweb.mmlab.be/ns/ql#> .\n"
            "@prefix rr: <http://www.w3.org/ns/r2rml#> .\n"
            "@prefix energy: <https://example.energy.demo/ontology#> .\n"
            "energy:Map a rr:TriplesMap;\n"
            '  rml:logicalSource [ rml:source "records.json"; rml:referenceFormulation ql:JSONPath; rml:iterator "$[*]" ];\n'
            '  rr:subjectMap [ rr:template "https://example.org/{id}"; rr:class energy:Thing ];\n'
            '  rr:predicateObjectMap [ rr:predicate energy:name; rr:objectMap [ rml:reference "name" ] ].\n',
            encoding="utf-8",
        )
        graph = await materialize_rml(mapping, tmp_path)
        assert (URIRef("https://example.org/1"), URIRef(ENERGY_NS + "name"), Literal("Alice")) in graph

    async def test_a_missing_reference_key_on_one_item_is_skipped_not_fatal(self, tmp_path):
        source = tmp_path / "records.json"
        source.write_text(json.dumps([{"id": "1"}, {"id": "2", "name": "Bob"}]), encoding="utf-8")
        mapping = tmp_path / "ok.rml.ttl"
        mapping.write_text(
            "@prefix rml: <http://semweb.mmlab.be/ns/rml#> .\n"
            "@prefix ql: <http://semweb.mmlab.be/ns/ql#> .\n"
            "@prefix rr: <http://www.w3.org/ns/r2rml#> .\n"
            "@prefix energy: <https://example.energy.demo/ontology#> .\n"
            "energy:Map a rr:TriplesMap;\n"
            '  rml:logicalSource [ rml:source "records.json"; rml:referenceFormulation ql:JSONPath; rml:iterator "$[*]" ];\n'
            '  rr:subjectMap [ rr:template "https://example.org/{id}"; rr:class energy:Thing ];\n'
            '  rr:predicateObjectMap [ rr:predicate energy:name; rr:objectMap [ rml:reference "name" ] ].\n',
            encoding="utf-8",
        )
        graph = await materialize_rml(mapping, tmp_path)
        assert (URIRef("https://example.org/1"), URIRef(ENERGY_NS + "name"), None) not in graph
        assert (URIRef("https://example.org/2"), URIRef(ENERGY_NS + "name"), Literal("Bob")) in graph
        # Both items still get their type triple even when a reference is absent.
        assert (URIRef("https://example.org/1"), RDF.type, URIRef(ENERGY_NS + "Thing")) in graph
