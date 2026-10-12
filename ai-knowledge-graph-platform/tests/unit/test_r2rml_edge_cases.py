"""Edge-case coverage for the R2RML adapter and the federated OBDA runner.

Added after a mutation campaign (docs/REMAINING_AUDIT_BACKLOG.md #8) showed the adapter's mapping
fields, error messages, name normalisation and federation preflight largely unpinned.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from rdflib import URIRef

from graphrag.ingestion import r2rml
from graphrag.ingestion.r2rml import (
    FederatedOBDAIngestor, FederatedOBDASource, R2RMLMappingError, r2rml_to_mapping,
)
from graphrag.ingestion.relational import RelationalGraphMapping

PREFIXES = """
@prefix rr: <http://www.w3.org/ns/r2rml#> .
@prefix ex: <https://example.test/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
"""

SUPPLIER = """
ex:Supplier a rr:TriplesMap; rr:logicalTable [rr:tableName "suppliers"];
  rr:subjectMap [rr:template "https://x/s/{sid}"; rr:class ex:Supplier];
  rr:predicateObjectMap [rr:predicate rdfs:label; rr:objectMap [rr:column "sname"]].
"""

ORDER = """
ex:Order a rr:TriplesMap; rr:logicalTable [rr:tableName "orders"];
  rr:subjectMap [rr:template "https://x/o/{oid}"; rr:class ex:Order];
  rr:predicateObjectMap [rr:predicate rdfs:label; rr:objectMap [rr:column "oname"]];
  rr:predicateObjectMap [rr:predicate ex:orderedFrom; rr:objectMap [
    rr:parentTriplesMap ex:Supplier; rr:joinCondition [rr:child "supplier_id"; rr:parent "sid"]]].
"""


def _load(tmp_path: Path, body: str, **kw):
    path = tmp_path / "m.ttl"
    path.write_text(PREFIXES + body, encoding="utf-8")
    args = dict(mapping_id="m1", version="3", source_id="erp", tenant="acme")
    args.update(kw)
    return r2rml_to_mapping(path, **args)


def _fails(tmp_path, body, message, **kw):
    with pytest.raises(R2RMLMappingError, match=re.escape(message)):
        _load(tmp_path, body, **kw)


def test_mapping_carries_ids_columns_and_default_ontology_version(tmp_path):
    mapping = _load(tmp_path, SUPPLIER + ORDER)
    assert (mapping.id, mapping.version, mapping.source_id, mapping.tenant) == ("m1", "3", "erp", "acme")
    assert mapping.ontology_version == "r2rml/v1"
    suppliers, orders = mapping.entities
    assert (suppliers.table, suppliers.entity_type, suppliers.id_column, suppliers.name_column) == \
        ("suppliers", "SUPPLIER", "sid", "sname")
    assert (orders.table, orders.entity_type, orders.id_column, orders.name_column) == \
        ("orders", "ORDER", "oid", "oname")
    (relation,) = mapping.relations
    assert (relation.table, relation.source_table, relation.target_table) == ("orders", "orders", "suppliers")
    assert (relation.source_column, relation.target_column, relation.relation) == ("oid", "supplier_id", "ORDEREDFROM")


def test_explicit_ontology_version_is_kept(tmp_path):
    assert _load(tmp_path, SUPPLIER, ontology_version="custom/v9").ontology_version == "custom/v9"


def test_a_map_with_no_relations_yields_none(tmp_path):
    assert _load(tmp_path, SUPPLIER).relations == []


def test_no_triples_map_is_rejected(tmp_path):
    _fails(tmp_path, "ex:a ex:b ex:c .", "no rr:TriplesMap found")


@pytest.mark.parametrize("body,label", [
    ('ex:X a rr:TriplesMap; rr:subjectMap [rr:template "https://x/{id}"; rr:class ex:X].', "rr:logicalTable"),
    ('ex:X a rr:TriplesMap; rr:logicalTable [rr:sqlQuery "select 1"]; '
     'rr:subjectMap [rr:template "https://x/{id}"; rr:class ex:X].', "rr:tableName"),
    ('ex:X a rr:TriplesMap; rr:logicalTable [rr:tableName "x"].', "rr:subjectMap"),
    ('ex:X a rr:TriplesMap; rr:logicalTable [rr:tableName "x"]; rr:subjectMap [rr:class ex:X].',
     "rr:subjectMap/rr:template"),
    ('ex:X a rr:TriplesMap; rr:logicalTable [rr:tableName "x"]; rr:subjectMap [rr:template "https://x/{id}"].',
     "rr:subjectMap/rr:class"),
])
def test_required_properties_must_occur_exactly_once(tmp_path, body, label):
    _fails(tmp_path, body, f"{label} must occur exactly once")


def test_duplicate_property_is_rejected(tmp_path):
    body = ('ex:X a rr:TriplesMap; rr:logicalTable [rr:tableName "x"], [rr:tableName "y"]; '
            'rr:subjectMap [rr:template "https://x/{id}"; rr:class ex:X].')
    _fails(tmp_path, body, "rr:logicalTable must occur exactly once")


@pytest.mark.parametrize("template", ["https://x/static", "https://x/{a}/{b}x{", "{1bad}", "https://x/{}"])
def test_subject_template_needs_one_identifier_column(tmp_path, template):
    body = ('ex:X a rr:TriplesMap; rr:logicalTable [rr:tableName "x"]; '
            f'rr:subjectMap [rr:template "{template}"; rr:class ex:X]; '
            'rr:predicateObjectMap [rr:predicate rdfs:label; rr:objectMap [rr:column "n"]].')
    if template == "https://x/{a}/{b}x{":
        # greedy match picks the last complete {identifier}; only the malformed ones are rejected
        assert _load(tmp_path, body).entities[0].id_column == "b"
    else:
        _fails(tmp_path, body, "rr:subjectMap/rr:template must contain exactly one identifier template {column}")


def test_exactly_one_label_column_is_required(tmp_path):
    two = ('ex:X a rr:TriplesMap; rr:logicalTable [rr:tableName "x"]; '
           'rr:subjectMap [rr:template "https://x/{id}"; rr:class ex:X]; '
           'rr:predicateObjectMap [rr:predicate rdfs:label; rr:objectMap [rr:column "a"]]; '
           'rr:predicateObjectMap [rr:predicate rdfs:label; rr:objectMap [rr:column "b"]].')
    _fails(tmp_path, two, "x: exactly one rdfs:label rr:column is required")


def test_label_object_map_needs_a_single_column(tmp_path):
    body = ('ex:X a rr:TriplesMap; rr:logicalTable [rr:tableName "x"]; '
            'rr:subjectMap [rr:template "https://x/{id}"; rr:class ex:X]; '
            'rr:predicateObjectMap [rr:predicate rdfs:label; rr:objectMap [rr:template "t/{a}"]].')
    _fails(tmp_path, body, "rdfs:label rr:column must occur exactly once")


def test_non_label_non_join_maps_are_ignored(tmp_path):
    extra = SUPPLIER.rstrip(".\n") + ('; rr:predicateObjectMap [rr:predicate ex:note; '
                                      'rr:objectMap [rr:column "note"]].')
    mapping = _load(tmp_path, extra)
    assert mapping.relations == [] and mapping.entities[0].name_column == "sname"


def test_parent_map_must_be_local(tmp_path):
    body = SUPPLIER + ORDER.replace("ex:Supplier;", "ex:Missing;")
    _fails(tmp_path, body, "rr:parentTriplesMap must reference a local rr:TriplesMap")


def test_join_parent_must_be_the_parent_identifier(tmp_path):
    body = SUPPLIER + ORDER.replace('rr:parent "sid"', 'rr:parent "other"')
    _fails(tmp_path, body, "join parent 'other' must be the parent subject identifier 'sid'")


def test_relation_needs_one_predicate_and_one_object_map(tmp_path):
    two_predicates = ORDER.replace("rr:predicate ex:orderedFrom;", "rr:predicate ex:orderedFrom, ex:boughtFrom;")
    _fails(tmp_path, SUPPLIER + two_predicates,
           "a parent-triples-map relation needs one predicate and one object map")


def test_join_condition_and_columns_must_be_present(tmp_path):
    no_join = ORDER.replace('rr:joinCondition [rr:child "supplier_id"; rr:parent "sid"]', "")
    _fails(tmp_path, SUPPLIER + no_join, "rr:joinCondition must occur exactly once")
    no_child = ORDER.replace('rr:child "supplier_id"; ', "")
    _fails(tmp_path, SUPPLIER + no_child, "rr:child must occur exactly once")
    no_parent = ORDER.replace('; rr:parent "sid"', "")
    _fails(tmp_path, SUPPLIER + no_parent, "rr:parent must occur exactly once")


@pytest.mark.parametrize("iri,expected", [
    ("https://example.test/ns#orderedFrom", "ORDEREDFROM"),
    ("https://example.test/path/Some-Name", "SOME_NAME"),
    ("https://example.test/a__b  c", "A_B_C"),
    ("https://example.test/v2.Thing", "V2_THING"),
    ("https://example.test/ns#--x--", "X"),
])
def test_local_name_is_a_safe_upper_snake_identifier(iri, expected):
    assert r2rml._local_name(URIRef(iri)) == expected


def test_local_name_rejects_unusable_iris():
    with pytest.raises(R2RMLMappingError, match=re.escape("cannot derive a safe local name from https://x/#---")):
        r2rml._local_name(URIRef("https://x/#---"))


def test_identifier_extraction():
    assert r2rml._identifier_from_template("https://x/{abc_1}", "t") == "abc_1"
    assert r2rml._identifier_from_template("{id}/suffix", "t") == "id"
    with pytest.raises(R2RMLMappingError, match=re.escape("lbl must contain exactly one identifier template {column}")):
        r2rml._identifier_from_template("https://x/none", "lbl")


# ── federation ────────────────────────────────────────────────────────────────

def _mapping(source_id="a", tenant="acme"):
    return RelationalGraphMapping(
        id=source_id, version="1", source_id=source_id, tenant=tenant,
        entities=[{"table": "items", "entity_type": "ITEM", "id_column": "id", "name_column": "name"}],
    )


class _Ingestor:
    def __init__(self, errors=(), result="done"):
        self.errors, self.result, self.validated, self.ingested = list(errors), result, [], []

    async def validate(self, mapping):
        self.validated.append(mapping.source_id)
        return type("Report", (), {"valid": not self.errors, "errors": self.errors})()

    async def ingest(self, mapping):
        self.ingested.append(mapping.source_id)
        return f"{self.result}:{mapping.source_id}"


def test_federation_needs_at_least_one_source():
    with pytest.raises(ValueError, match="at least one federated OBDA source is required"):
        FederatedOBDAIngestor([])


@pytest.mark.asyncio
async def test_source_ids_must_be_unique():
    one, two = _Ingestor(), _Ingestor()
    fed = FederatedOBDAIngestor([FederatedOBDASource("x", one, _mapping("same")),
                                 FederatedOBDASource("y", two, _mapping("same"))])
    with pytest.raises(ValueError, match="federated OBDA source_id values must be unique"):
        await fed.validate()
    assert one.validated == [] and two.validated == []        # nothing is validated or read first


@pytest.mark.asyncio
async def test_all_sources_must_target_one_tenant():
    fed = FederatedOBDAIngestor([FederatedOBDASource("x", _Ingestor(), _mapping("a", "acme")),
                                 FederatedOBDASource("y", _Ingestor(), _mapping("b", "other"))])
    with pytest.raises(ValueError, match="all federated OBDA sources must target one tenant"):
        await fed.validate()


@pytest.mark.asyncio
async def test_validation_failures_are_reported_together_and_block_ingest():
    good, bad1, bad2 = _Ingestor(), _Ingestor(errors=["e1", "e2"]), _Ingestor(errors=["e3"])
    fed = FederatedOBDAIngestor([FederatedOBDASource("g", good, _mapping("a")),
                                 FederatedOBDASource("b1", bad1, _mapping("b")),
                                 FederatedOBDASource("b2", bad2, _mapping("c"))])
    with pytest.raises(ValueError, match=re.escape("federated OBDA validation failed: e1; e2; e3")):
        await fed.ingest()
    assert good.ingested == bad1.ingested == bad2.ingested == []


@pytest.mark.asyncio
async def test_valid_federation_validates_then_ingests_every_source_in_order():
    one, two = _Ingestor(result="r"), _Ingestor(result="r")
    fed = FederatedOBDAIngestor([FederatedOBDASource("x", one, _mapping("a")),
                                 FederatedOBDASource("y", two, _mapping("b"))])
    reports = await fed.validate()
    assert [r.valid for r in reports] == [True, True]
    assert await fed.ingest() == ["r:a", "r:b"]
    assert one.validated == ["a", "a"] and one.ingested == ["a"] and two.ingested == ["b"]


def test_exports_and_regex():
    assert r2rml.__all__ == ["FederatedOBDAIngestor", "FederatedOBDASource", "R2RMLMappingError", "r2rml_to_mapping"]
    assert str(r2rml.RR["class"]) == "http://www.w3.org/ns/r2rml#class"
    assert issubclass(R2RMLMappingError, ValueError)


# ── branch conditions inside r2rml_to_mapping ─────────────────────────────────

def test_predicate_without_object_map_is_ignored_not_rejected(tmp_path):
    body = SUPPLIER.rstrip(".\n") + "; rr:predicateObjectMap [rr:predicate ex:dangling]."
    mapping = _load(tmp_path, body)
    assert mapping.relations == [] and len(mapping.entities) == 1


_ORDER_HEAD = """
ex:Order a rr:TriplesMap; rr:logicalTable [rr:tableName "orders"];
  rr:subjectMap [rr:template "https://x/o/{oid}"; rr:class ex:Order];
  rr:predicateObjectMap [rr:predicate rdfs:label; rr:objectMap [rr:column "oname"]];
  rr:predicateObjectMap [rr:predicate ex:orderedFrom; rr:objectMap %s].
"""
_PARENT_MAP = '[rr:parentTriplesMap ex:Supplier; rr:joinCondition [rr:child "supplier_id"; rr:parent "sid"]]'


def test_two_object_maps_on_a_relation_are_rejected(tmp_path):
    two = _ORDER_HEAD % (_PARENT_MAP + ", " + _PARENT_MAP.replace("supplier_id", "alt_id"))
    _fails(tmp_path, SUPPLIER + two, "a parent-triples-map relation needs one predicate and one object map")


def test_one_parent_object_map_among_several_still_marks_a_relation(tmp_path):
    mixed = _ORDER_HEAD % (_PARENT_MAP + ', [rr:column "plain"]')
    _fails(tmp_path, SUPPLIER + mixed, "a parent-triples-map relation needs one predicate and one object map")


def test_no_label_at_all_names_the_table(tmp_path):
    body = ('ex:X a rr:TriplesMap; rr:logicalTable [rr:tableName "things"]; '
            'rr:subjectMap [rr:template "https://x/{id}"; rr:class ex:X].')
    _fails(tmp_path, body, "things: exactly one rdfs:label rr:column is required")


def test_several_relations_keep_their_own_endpoints(tmp_path):
    third = """
ex:Part a rr:TriplesMap; rr:logicalTable [rr:tableName "parts"];
  rr:subjectMap [rr:template "https://x/p/{pid}"; rr:class ex:Part];
  rr:predicateObjectMap [rr:predicate rdfs:label; rr:objectMap [rr:column "pname"]];
  rr:predicateObjectMap [rr:predicate ex:madeBy; rr:objectMap [
    rr:parentTriplesMap ex:Supplier; rr:joinCondition [rr:child "maker_id"; rr:parent "sid"]]].
"""
    mapping = _load(tmp_path, SUPPLIER + ORDER + third)
    rels = {r.relation: r for r in mapping.relations}
    assert set(rels) == {"ORDEREDFROM", "MADEBY"}
    assert (rels["MADEBY"].source_table, rels["MADEBY"].target_table) == ("parts", "suppliers")
    assert (rels["MADEBY"].source_column, rels["MADEBY"].target_column) == ("pid", "maker_id")
    assert (rels["ORDEREDFROM"].source_column, rels["ORDEREDFROM"].target_column) == ("oid", "supplier_id")


def test_source_may_be_a_string_path(tmp_path):
    path = tmp_path / "m.ttl"
    path.write_text(PREFIXES + SUPPLIER, encoding="utf-8")
    mapping = r2rml_to_mapping(str(path), mapping_id="m", version="1", source_id="s", tenant="t")
    assert mapping.entities[0].table == "suppliers"


@pytest.mark.parametrize("iri,expected", [
    ("http://x/ns#a/b", "B"),            # '#' is split first, then '/'
    ("http://x/a/b", "B"),
    ("http://x#a#b", "B"),               # only the last '#'
    ("http://x/__Mixed__Case__", "MIXED_CASE"),
    ("http://x/a1b2", "A1B2"),
])
def test_local_name_splitting_rules(iri, expected):
    assert r2rml._local_name(URIRef(iri)) == expected


def test_federation_reports_validation_errors_in_source_order():
    a, b = _Ingestor(errors=["first"]), _Ingestor(errors=["second"])
    fed = FederatedOBDAIngestor([FederatedOBDASource("a", a, _mapping("a")), FederatedOBDASource("b", b, _mapping("b"))])
    with pytest.raises(ValueError) as exc:
        import asyncio
        asyncio.run(fed.validate())
    assert str(exc.value).endswith("first; second")
    assert fed.sources == tuple(fed.sources) and len(fed.sources) == 2
