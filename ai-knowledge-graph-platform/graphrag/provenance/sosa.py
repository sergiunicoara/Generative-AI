"""SOSA (W3C Semantic Sensor Network) vocabulary annotation for Energy RDF.

The operational graph remains a Neo4j property graph and the Energy domain's
own `energy:Observation` vocabulary (see `ontology/models/energy-asset-
intelligence.yaml`) stays the authoritative shape. This module only *adds*
SOSA typing/predicates onto observation subjects an RML mapping already
materialized — the same "annotate what's already there, add types/predicates,
never touch the compiler" pattern `graphrag/provenance/prov_o.py` uses for
PROV-O.

Simplification, stated rather than hidden: `sosa:hasSimpleResult` carries the
raw value literal without a formal units vocabulary (e.g. QUDT) attached to
it — the existing `energy:unit` triple already covers the unit; out of scope
for a vocabulary-typing layer to build a full SOSA Result/QUDT structure.
"""

from __future__ import annotations

from urllib.parse import quote

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF

from graphrag.domains.energy.vocabulary import ENERGY

SOSA = Namespace("http://www.w3.org/ns/sosa/")


def annotate_sosa_observations(graph: Graph) -> Graph:
    """Add SOSA vocabulary typing to every `energy:Observation` in `graph`.

    Additive only: reads the energy:* triples an RML mapping already put on
    each Observation subject and adds SOSA types/predicates beside them.
    Never removes, replaces, or re-derives an energy:* triple. Safe to call
    more than once — every added triple is idempotent (same subject/
    predicate/object each time), so rdflib's set semantics prevent
    duplication.
    """
    for obs in list(graph.subjects(RDF.type, ENERGY.Observation)):
        graph.add((obs, RDF.type, SOSA.Observation))

        asset = graph.value(obs, ENERGY.observedAsset)
        if asset is not None:
            graph.add((asset, RDF.type, SOSA.FeatureOfInterest))
            graph.add((obs, SOSA.hasFeatureOfInterest, asset))

        metric = graph.value(obs, ENERGY.metric)
        if metric is not None:
            graph.add((obs, SOSA.observedProperty, _property_uri(str(metric))))

        value = graph.value(obs, ENERGY.value)
        if value is not None:
            # sosa:hasSimpleResult (not sosa:hasResult + a child Result node):
            # the value is already a plain literal, so SOSA's no-node shape is
            # both simpler and avoids inventing an untyped subject the LPG
            # projector's triple ledger has no representation for (see
            # graphrag/domains/energy/lpg_projection.py's triple_ledger()).
            # The existing energy:unit triple already carries the unit;
            # not duplicated here.
            graph.add((obs, SOSA.hasSimpleResult, Literal(value)))

        recorded_at = graph.value(obs, ENERGY.recordedAt)
        if recorded_at is not None:
            graph.add((obs, SOSA.resultTime, Literal(recorded_at)))

        observed_at = graph.value(obs, ENERGY.observedAt)
        if observed_at is not None:
            graph.add((obs, SOSA.phenomenonTime, Literal(observed_at)))

    return graph


def _property_uri(metric: str) -> URIRef:
    return URIRef(f"https://example.energy.demo/observable-property/{quote(metric, safe='')}")


__all__ = ["SOSA", "annotate_sosa_observations"]
