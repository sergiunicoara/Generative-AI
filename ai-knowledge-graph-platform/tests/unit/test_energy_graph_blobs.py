"""Content-addressed storage for published Energy graphs.

The properties that matter are the ones the governance store depends on:
the hash identifies content and nothing else, storing the same content twice
is a no-op, and a stored version reloads to an equal graph with its prefixes
intact.
"""

from __future__ import annotations

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

from graphrag.domains.energy.graph_blobs import (
    canonical_bytes,
    content_hash,
    prefixes_of,
    read_graph,
    write_graph,
)

EX = Namespace("https://example.energy.demo/ontology#")
REC = Namespace("https://example.energy.demo/record/")


def _graph(value: str = "96.0") -> Graph:
    graph = Graph()
    graph.bind("energy", EX)
    graph.add((REC["obs-1"], RDF.type, EX.Observation))
    graph.add((REC["obs-1"], EX.value, Literal(value, datatype=XSD.decimal)))
    graph.add((REC["obs-1"], EX.unit, Literal("C")))
    return graph


class TestContentIdentity:
    def test_insertion_order_does_not_change_the_hash(self):
        """The hash must identify content, not the order rdflib happened to
        serialise it in -- otherwise publish-idempotency silently fails."""
        first = Graph()
        second = Graph()
        triples = [
            (REC["a"], RDF.type, EX.Observation),
            (REC["b"], RDF.type, EX.Observation),
            (REC["c"], EX.unit, Literal("C")),
        ]
        for triple in triples:
            first.add(triple)
        for triple in reversed(triples):
            second.add(triple)

        assert content_hash(first) == content_hash(second)

    def test_a_changed_value_changes_the_hash(self):
        assert content_hash(_graph("96.0")) != content_hash(_graph("91.5"))

    def test_canonical_bytes_are_sorted_n_triples(self):
        payload = canonical_bytes(_graph()).decode("utf-8")
        lines = [line for line in payload.splitlines() if line]

        assert lines == sorted(lines)
        assert all(line.endswith(" .") for line in lines)
        # N-Triples, not Turtle: no prefix declarations, every term absolute.
        assert "@prefix" not in payload


class TestStorage:
    def test_writing_returns_a_path_named_by_the_content(self, tmp_path):
        graph = _graph()
        digest, path, size = write_graph(graph, tmp_path)

        assert digest == content_hash(graph)
        assert path.name == f"{digest}.nt"
        assert path.parent.name == digest[:2]
        assert size == len(canonical_bytes(graph))
        assert path.read_bytes() == canonical_bytes(graph)

    def test_storing_the_same_content_twice_is_a_no_op(self, tmp_path):
        """Two workers publishing identical content must not conflict --
        the path is derived from the bytes, so the second write finds its
        own result already there."""
        first_digest, first_path, _ = write_graph(_graph(), tmp_path)
        before = first_path.stat().st_mtime_ns

        second_digest, second_path, _ = write_graph(_graph(), tmp_path)

        assert (second_digest, second_path) == (first_digest, first_path)
        assert second_path.stat().st_mtime_ns == before

    def test_a_stored_graph_reloads_equal_with_its_prefixes(self, tmp_path):
        graph = _graph()
        _, path, _ = write_graph(graph, tmp_path)

        restored = read_graph(path, prefixes_of(graph))

        assert set(restored) == set(graph)
        assert dict(prefixes_of(restored))["energy"] == str(EX)
        # And the round trip is itself content-stable.
        assert content_hash(restored) == content_hash(graph)

    def test_no_partial_file_is_left_behind_on_success(self, tmp_path):
        write_graph(_graph(), tmp_path)

        leftovers = [p for p in tmp_path.rglob("*") if p.is_file() and p.suffix == ".tmp"]
        assert leftovers == []

    def test_typed_literals_survive_the_round_trip(self, tmp_path):
        graph = _graph()
        _, path, _ = write_graph(graph, tmp_path)

        restored = read_graph(path)
        value = restored.value(REC["obs-1"], EX.value)

        assert isinstance(value, Literal)
        assert value.datatype == URIRef(str(XSD.decimal))
