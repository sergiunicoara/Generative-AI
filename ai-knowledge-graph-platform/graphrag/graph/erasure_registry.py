"""Every place a fact can live after it is written, and what erasure does there.

An erasure or correction claim only holds for the whole system if each derived
store is accounted for.  This registry lists them with one of five coverages:

* ``erased``       an approved erasure removes the data (``handler``);
* ``redacted``     the record stays, the personal content is replaced;
* ``expires``      not purged, but gone after a stated time to live;
* ``immutable``    append-only by design, with the retention or correction
                   rule that applies instead (``rule``);
* ``not_covered``  a known gap, stated so it is not mistaken for coverage.

``tests/unit/test_privacy_controls.py`` checks that every ``erased`` and
``redacted`` handler exists and exercises the stores that run offline; the
Neo4j-backed stores are covered by mocked-driver tests only, not by a live
database run.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Literal

Coverage = Literal["erased", "redacted", "expires", "immutable", "not_covered"]


@dataclass(frozen=True)
class DerivedStore:
    name: str
    holds: str
    coverage: Coverage
    handler: str = ""      # "module:attribute.path" of the erasure entry point
    rule: str = ""


STORES: tuple[DerivedStore, ...] = (
    DerivedStore("neo4j.entity", "entity node, its edges, statements and change log", "erased",
                 "graphrag.graph.gdpr:GDPRService.forget_entity"),
    DerivedStore("neo4j.chunk_text", "chunk text mentioning the entity", "redacted",
                 "graphrag.graph.gdpr:GDPRService.forget_entity",
                 "exact-name matches become [REDACTED]; aliases and pronouns are not caught"),
    DerivedStore("neo4j.entity_embeddings", "entity vectors", "erased",
                 "graphrag.graph.gdpr:GDPRService.forget_entity", "removed with the entity node"),
    DerivedStore("neo4j.chunk_embeddings", "chunk vectors", "erased",
                 "graphrag.graph.gdpr:GDPRService.forget_entity",
                 "the vector of a redacted chunk is dropped; GDPRService.reembed_redacted recomputes it from the redacted text"),
    DerivedStore("neo4j.community_summaries", "community summaries, summary snapshots and their vectors", "redacted",
                 "graphrag.graph.gdpr:GDPRService.forget_entity",
                 "exact-name matches become [REDACTED] and the vectors are dropped until reembed_redacted"),
    DerivedStore("neo4j.aliases", "alias nodes and the alias fulltext index", "erased",
                 "graphrag.graph.gdpr:GDPRService.forget_entity",
                 "aliases pointing only at the entity are deleted; an alias shared with another entity stays"),
    DerivedStore("neo4j.media", "media attachments of the entity", "erased",
                 "graphrag.graph.multimodal:MultiModalEntityService.delete_entity_media",
                 "called from GDPRService._erase_derived"),
    DerivedStore("cache.embeddings", "in-process embedding cache", "erased",
                 "graphrag.graph.embedding_cache:EmbeddingCache.invalidate", "called from GDPRService._erase_derived"),
    DerivedStore("cache.answers", "query answer cache (raw chunk contexts)", "erased",
                 "graphrag.retrieval.query_cache:QueryCache.flush_tenant",
                 "flushed after erasure; the corpus revision bump makes old keys unreachable first"),
    DerivedStore("redis.result_store", "async query results by query id", "expires",
                 rule="time to live 3600 s; no tenant-wide purge"),
    DerivedStore("redis.sessions", "conversation sessions", "expires", rule="time to live 24 h"),
    DerivedStore("knowledge.decision_cache", "cached knowledge decisions", "erased",
                 "graphrag.context_graph.knowledge_service:KnowledgeEvaluationService.evict_tenant"),
    DerivedStore("knowledge.decision_log", "decision records with submitted facts and caller identity",
                 "redacted", "graphrag.context_graph.knowledge_service:KnowledgeEvaluationService.erase_subject",
                 "facts and identity are removed; the record of a decision stays and no longer replays"),
    DerivedStore("knowledge.trust_reports", "distrust reports with their bundles", "redacted",
                 "graphrag.domains.energy.governance_store:GovernanceStore.redact_knowledge_decisions",
                 "the reporter's identity and submitted facts are removed"),
    DerivedStore("knowledge.releases", "published knowledge releases", "immutable",
                 rule="curated knowledge, not personal data by contract; a correction is a new release, history is "
                      "kept for replay"),
    DerivedStore("energy.governance", "Energy publications, workflow and evidence history", "immutable",
                 rule="append-only governance history holding actor ids only"),
    DerivedStore("energy.rdf_blobs", "content-addressed RDF blobs", "immutable",
                 rule="shared by digest; a correction is a new publication"),
    DerivedStore("rdf.exports", "per-tenant Turtle exports", "erased",
                 "graphrag.graph.gdpr:GDPRService.forget_entity",
                 "the tenant's export file is removed; regenerate it with scripts/export_rdf.py"),
    DerivedStore("rdf.named_graphs", "triple-store named graphs (ADR-0013)", "erased",
                 "graphrag.graph.named_graphs:erase_resource", "per resource within one tenant; drop_tenant for all"),
    DerivedStore("cache.llm_extraction", "on-disk prompt/response cache (opt-in)", "erased",
                 "graphrag.core.llm_cache:LLMCache.purge", "no tenant key, so the purge empties the cache"),
    DerivedStore("logs.structured", "application logs", "redacted",
                 rule="the erasure log line records a digest, not the name; other lines follow log retention"),
    DerivedStore("audit.deletion", "DeletionAudit nodes", "immutable",
                 rule="keep the erasure's own audit record (who, when, counts); documented exception"),
    DerivedStore("audit.capabilities", "capability audit events", "immutable",
                 rule="arguments are stored as a hash only"),
    DerivedStore("context_graph.traces", "decision traces", "redacted",
                 "graphrag.context_graph.repository:ContextGraphRepository.redact_trace",
                 "appends a redaction marker; retention policy applies"),
)


def resolve(handler: str):
    """Import ``module:attr.path``; raises if the handler does not exist."""
    module, _, path = handler.partition(":")
    target = importlib.import_module(module)
    for part in path.split("."):
        target = getattr(target, part)
    return target


def erasure_table() -> str:
    """Markdown table of the registry, for documentation."""
    lines = ["| Store | Holds | Coverage | Handler or rule |", "| --- | --- | --- | --- |"]
    for store in STORES:
        detail = "; ".join(x for x in (f"`{store.handler}`" if store.handler else "", store.rule) if x)
        lines.append(f"| {store.name} | {store.holds} | {store.coverage} | {detail} |")
    return "\n".join(lines) + "\n"


__all__ = ["STORES", "DerivedStore", "erasure_table", "resolve"]
