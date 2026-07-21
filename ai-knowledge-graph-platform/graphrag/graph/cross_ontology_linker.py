"""Cross-ontology entity linking — resolve external RDF/OWL entities against
this platform's Neo4j-backed entity graph and emit owl:sameAs bridges.

Problem solved
--------------
A customer's own RDF/OWL ontology (exported from Stardog, Protege, another
platform) uses entity URIs and labels that don't line up with this
platform's ``https://graphrag.example.com/entity/...`` URIs, even when the
two graphs describe the same real-world entities. Without an identity
bridge, a SPARQL query spanning both graphs (a UNION, or a live SERVICE
federation) silently returns nothing for entities that are actually the
same thing under two different names.

Strategy — reuses the existing intra-corpus alias-resolution pipeline
(``AliasRegistry``: exact -> normalized -> fuzzy -> embedding -> human
review, see alias_registry.py) against a *second* data source instead of a
second document:

  1. Parse the external Turtle file; extract (uri, rdfs:label, rdf:type
     hint) triples.
  2. Resolve each label against this tenant's AliasRegistry (loaded from
     Neo4j) using the same exact/fuzzy bands already used for intra-corpus
     alias resolution, then fall back to embedding similarity for labels
     that have no exact/fuzzy candidate.
  3. High-confidence matches -> an ``owl:sameAs`` triple, emitted directly
     into an in-memory bridge graph.
  4. Ambiguous-band matches -> the same human ReviewQueueService used for
     intra-corpus dedup, tagged ``match_type="cross_ontology_fuzzy"`` /
     ``"cross_ontology_embedding"`` so a reviewer can tell a cross-ontology
     link decision apart from an intra-corpus one.
  5. No match at all -> left unlinked. The external entity still merges
     into the combined graph via ``merge_graphs()``, just without a
     same-as bridge — SPARQL queries simply won't join it to this
     platform's data.

Limitation
----------
Embedding-similarity matching filters candidates by entity type
(``find_duplicate_by_embedding`` / ``find_candidate_by_embedding`` both
require an exact ``entity_type`` match). The type hint extracted from the
external ontology's ``rdf:type`` is a best-effort guess (last URI path
segment, upper-cased) and will not always line up with this platform's
type vocabulary — a mismatch silently drops a candidate from the
embedding path rather than raising an error. Exact/fuzzy label matching is
unaffected by this (it is not type-filtered).

Usage::

    from graphrag.graph.cross_ontology_linker import CrossOntologyLinker

    linker = CrossOntologyLinker(neo4j_client, tenant="telecom")
    result = await linker.link("customer_ontology.ttl")
    print(result.summary())

    merged = CrossOntologyLinker.merge_graphs(
        "customer_ontology.ttl", "exports/graph_export.ttl",
        extra=result.same_as_graph,
    )
    merged.serialize("exports/merged.ttl", format="turtle")
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

import structlog
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from graphrag.graph.alias_registry import AmbiguousMatch, get_alias_registry

log = structlog.get_logger(__name__)

INST = Namespace("https://graphrag.example.com/entity/")


def _entity_uri(name: str, etype: str, tenant: str) -> URIRef:
    """Mirror scripts/export_rdf.py's _entity_uri — same tenant-scoped URI scheme,
    so same-as triples resolve to the exact URIs this platform's own export uses."""
    def _safe(s: str) -> str:
        return urllib.parse.quote(s, safe="")
    return INST[f"{_safe(tenant)}/{_safe(etype)}/{_safe(name)}"]


@dataclass
class LinkCandidate:
    external_uri: str
    label: str
    type_hint: str


@dataclass
class LinkResult:
    auto_linked: list[dict] = field(default_factory=list)
    queued_for_review: list[dict] = field(default_factory=list)
    unmatched: list[dict] = field(default_factory=list)
    same_as_graph: Graph = field(default_factory=Graph)

    def summary(self) -> str:
        total = len(self.auto_linked) + len(self.queued_for_review) + len(self.unmatched)
        return (
            f"Cross-ontology linking: {len(self.auto_linked)} auto-linked, "
            f"{len(self.queued_for_review)} queued for review, "
            f"{len(self.unmatched)} unmatched ({total} candidates total)"
        )


class CrossOntologyLinker:
    """Link an external RDF/OWL ontology's entities to this tenant's graph."""

    def __init__(self, neo4j_client=None, tenant: str = "default", embedder=None):
        self._tenant = tenant
        if neo4j_client is None:
            from graphrag.graph.neo4j_client import get_neo4j
            neo4j_client = get_neo4j()
        self._neo4j = neo4j_client
        self._registry = get_alias_registry(neo4j_client, tenant=tenant)
        # Optional — pass an Embedder to also attempt similarity matching for
        # labels with no exact/fuzzy candidate. Omitted by default so `link()`
        # works with zero extra config (exact/fuzzy only) unless opted in.
        self._embedder = embedder

    # ── Extraction ───────────────────────────────────────────────────────────

    def _extract_candidates(self, external_ttl_path: str | Path) -> list[LinkCandidate]:
        """Pull (uri, label, type_hint) triples out of the external Turtle file."""
        g = Graph()
        g.parse(str(external_ttl_path), format="turtle")

        candidates: list[LinkCandidate] = []
        seen_uris: set[str] = set()
        for subject, _, label in g.triples((None, RDFS.label, None)):
            uri = str(subject)
            if uri in seen_uris:
                continue
            seen_uris.add(uri)

            type_hint = "CONCEPT"
            types = [str(t) for t in g.objects(subject, RDF.type) if str(t) != str(OWL.NamedIndividual)]
            if types:
                type_hint = types[0].rstrip("/").split("/")[-1].split("#")[-1].upper()

            candidates.append(LinkCandidate(external_uri=uri, label=str(label), type_hint=type_hint))

        log.info("cross_ontology_linker.candidates_extracted",
                  path=str(external_ttl_path), count=len(candidates))
        return candidates

    # ── Linking ──────────────────────────────────────────────────────────────

    async def link(self, external_ttl_path: str | Path, source_label: str | None = None) -> LinkResult:
        """
        Resolve every rdfs:label'd entity in the external Turtle file against
        this tenant's graph. Returns a LinkResult with owl:sameAs triples for
        confident matches and human-review-queue entries for ambiguous ones.
        """
        await self._registry.load()
        candidates = self._extract_candidates(external_ttl_path)
        source_doc = source_label or Path(external_ttl_path).name

        result = LinkResult()
        result.same_as_graph.bind("owl", OWL)

        review_queue = None
        if self._review_queue_enabled():
            from graphrag.graph.review_queue import ReviewQueueService
            review_queue = ReviewQueueService(self._neo4j)

        for cand in candidates:
            match = self._registry.resolve(cand.label)

            if isinstance(match, tuple):
                name, etype = match
                self._add_same_as(result, cand.external_uri, name, etype)
                result.auto_linked.append({
                    "external_uri": cand.external_uri, "label": cand.label,
                    "internal_name": name, "internal_type": etype,
                    "match_type": "exact_or_fuzzy",
                })
                continue

            if isinstance(match, AmbiguousMatch):
                await self._enqueue_ambiguous(cand, match.candidate, match.score,
                                               f"cross_ontology_{match.match_type}",
                                               source_doc, review_queue, result)
                continue

            # No exact/fuzzy candidate — try embedding similarity if an
            # embedder was supplied (mirrors GraphWriter's soft-dup check).
            if not await self._try_embedding_match(cand, review_queue, source_doc, result):
                result.unmatched.append({"external_uri": cand.external_uri,
                                          "label": cand.label, "reason": "no_match"})

        log.info("cross_ontology_linker.link_complete",
                  tenant=self._tenant, source=source_doc,
                  auto_linked=len(result.auto_linked),
                  queued=len(result.queued_for_review),
                  unmatched=len(result.unmatched))
        return result

    async def _try_embedding_match(
        self, cand: LinkCandidate, review_queue, source_doc: str, result: LinkResult,
    ) -> bool:
        if self._embedder is None:
            return False
        try:
            embeddings = await self._embedder.embed_texts([cand.label])
            embedding = embeddings[0]
        except Exception as exc:
            log.debug("cross_ontology_linker.embed_failed", label=cand.label, error=str(exc)[:120])
            return False

        dup = await self._registry.find_duplicate_by_embedding(embedding, cand.type_hint)
        if dup:
            name, etype, score = dup
            self._add_same_as(result, cand.external_uri, name, etype)
            result.auto_linked.append({
                "external_uri": cand.external_uri, "label": cand.label,
                "internal_name": name, "internal_type": etype,
                "score": score, "match_type": "embedding",
            })
            return True

        soft = await self._registry.find_candidate_by_embedding(embedding, cand.type_hint)
        if soft:
            name, etype, score = soft
            await self._enqueue_ambiguous(cand, (name, etype), score,
                                           "cross_ontology_embedding",
                                           source_doc, review_queue, result)
            return True
        return False

    async def _enqueue_ambiguous(
        self, cand: LinkCandidate, candidate: tuple[str, str], score: float,
        match_type: str, source_doc: str, review_queue, result: LinkResult,
    ) -> None:
        if review_queue is None:
            result.unmatched.append({"external_uri": cand.external_uri, "label": cand.label,
                                      "reason": "review_queue_disabled"})
            return
        try:
            item_id = await review_queue.enqueue(
                raw_name=cand.label, raw_type=cand.type_hint,
                candidate_name=candidate[0], candidate_type=candidate[1],
                score=score, match_type=match_type,
                source_doc=source_doc, tenant=self._tenant,
            )
            result.queued_for_review.append({
                "external_uri": cand.external_uri, "label": cand.label,
                "candidate_name": candidate[0], "score": score, "item_id": item_id,
            })
        except Exception as exc:
            log.warning("cross_ontology_linker.enqueue_failed", label=cand.label, error=str(exc)[:120])
            result.unmatched.append({"external_uri": cand.external_uri, "label": cand.label,
                                      "reason": "enqueue_failed"})

    def _add_same_as(self, result: LinkResult, external_uri: str, internal_name: str, internal_type: str) -> None:
        internal_uri = _entity_uri(internal_name, internal_type, self._tenant)
        result.same_as_graph.add((URIRef(external_uri), OWL.sameAs, internal_uri))

    def _review_queue_enabled(self) -> bool:
        from graphrag.core.config import get_settings
        return bool(get_settings().ingestion.get("review_queue_enabled", True))

    # ── Merge ────────────────────────────────────────────────────────────────

    @staticmethod
    def merge_graphs(*ttl_paths: str | Path, extra: Graph | None = None) -> Graph:
        """
        Union multiple Turtle files (and an optional extra in-memory Graph,
        typically the ``same_as_graph`` produced by ``link()``) into one
        rdflib Graph, ready for ``SPARQLBridge`` queries spanning both
        ontologies via the owl:sameAs bridges.
        """
        merged = Graph()
        for path in ttl_paths:
            merged.parse(str(path), format="turtle")
        if extra is not None:
            for triple in extra:
                merged.add(triple)
        log.info("cross_ontology_linker.merged", sources=len(ttl_paths), triples=len(merged))
        return merged
