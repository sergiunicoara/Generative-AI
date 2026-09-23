"""Unit tests for the SOSA vocabulary annotation layer over Energy RDF
observations — see graphrag/provenance/sosa.py's module docstring."""

from __future__ import annotations

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from graphrag.domains.energy.vocabulary import ASSET, ENERGY, REC
from graphrag.provenance.sosa import SOSA, annotate_sosa_observations


def _observation_graph() -> tuple[Graph, URIRef, URIRef]:
    graph = Graph()
    obs = REC["obs-1"]
    asset = ASSET["WT-01"]
    graph.add((obs, RDF.type, ENERGY.Observation))
    graph.add((obs, ENERGY.observedAsset, asset))
    graph.add((obs, ENERGY.metric, Literal("vibration_rms")))
    graph.add((obs, ENERGY.value, Literal("0.82", datatype=XSD.decimal)))
    graph.add((obs, ENERGY.unit, Literal("mm/s")))
    graph.add((obs, ENERGY.observedAt, Literal("2026-01-01T00:00:00+00:00", datatype=XSD.dateTime)))
    graph.add((obs, ENERGY.recordedAt, Literal("2026-01-01T00:05:00+00:00", datatype=XSD.dateTime)))
    return graph, obs, asset


def test_observation_gets_sosa_type_and_feature_of_interest():
    graph, obs, asset = _observation_graph()
    annotate_sosa_observations(graph)

    assert (obs, RDF.type, SOSA.Observation) in graph
    assert (asset, RDF.type, SOSA.FeatureOfInterest) in graph
    assert (obs, SOSA.hasFeatureOfInterest, asset) in graph


def test_observation_gets_result_and_times():
    graph, obs, _ = _observation_graph()
    annotate_sosa_observations(graph)

    assert (obs, SOSA.hasSimpleResult, Literal("0.82", datatype=XSD.decimal)) in graph
    assert (obs, SOSA.resultTime, Literal("2026-01-01T00:05:00+00:00", datatype=XSD.dateTime)) in graph
    assert (obs, SOSA.phenomenonTime, Literal("2026-01-01T00:00:00+00:00", datatype=XSD.dateTime)) in graph


def test_observation_gets_observed_property_from_metric():
    graph, obs, _ = _observation_graph()
    annotate_sosa_observations(graph)

    prop = graph.value(obs, SOSA.observedProperty)
    assert prop is not None
    assert "vibration_rms" in str(prop)


def test_missing_optional_field_does_not_crash_or_fabricate():
    graph, obs, _ = _observation_graph()
    graph.remove((obs, ENERGY.unit, None))
    annotate_sosa_observations(graph)

    # No crash, and the still-present value is annotated correctly even
    # though an optional sibling field (unit) is absent.
    assert (obs, SOSA.hasSimpleResult, Literal("0.82", datatype=XSD.decimal)) in graph


def test_energy_triples_are_untouched():
    graph, obs, asset = _observation_graph()
    before = set(graph)
    annotate_sosa_observations(graph)

    assert before.issubset(set(graph))


def test_annotation_is_idempotent():
    graph, obs, _ = _observation_graph()
    annotate_sosa_observations(graph)
    once = set(graph)
    annotate_sosa_observations(graph)

    assert set(graph) == once
