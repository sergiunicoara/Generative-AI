"""Local search: vector ANN + BM25 → RRF → cross-encoder rerank → multi-hop → GNN scorer.

Full 6-step retrieval pipeline
-------------------------------
1. Vector ANN          — semantic similarity via Neo4j vector index
2. BM25 + RRF fusion   — lexical hybrid search; merges both ranked lists
3. Cross-encoder       — deep pairwise (query, chunk) relevance scoring
4. Multi-hop traversal — follow RELATES_TO edges to find bridging chunks
5. GNN scoring         — graph-propagated entity embeddings (GCN / GAT)
6. Entity context      — gather entity descriptions for LLM prompt

Additional features
-------------------
- Session context    — resolves ambiguous follow-up queries across turns
- Authority weights  — edges from superseded documents get confidence penalty
- Source type filter — inferred/LLM-generated edges scored separately
"""

from __future__ import annotations

import asyncio
import re

import structlog

from graphrag.core.config import get_settings
from graphrag.graph.document_authority import DocumentAuthorityService
from graphrag.graph.gnn_scorer import GNNScorer
from graphrag.graph.neo4j_client import get_neo4j
from graphrag.ingestion.embedder import Embedder
from graphrag.retrieval.bm25_search import HybridBM25Search
from graphrag.retrieval.reranker import CrossEncoderReranker
from graphrag.retrieval.session_context import get_session_context

log = structlog.get_logger(__name__)

# ── Query-adaptive weight routing ─────────────────────────────────────────────
_RELATIONAL_SIGNALS = [
    "relationship", "how did", "caused by", "connected", "led to",
    "between", "link", "related", "chain", "through", "via",
    "owned by", "founded by", "acquired", "impact of", "effect of",
]


def _adaptive_weights(question: str, alpha_default: float, beta_default: float):
    q = question.lower()
    if any(s in q for s in _RELATIONAL_SIGNALS):
        return 0.5, 0.5
    return alpha_default, beta_default


# ── Named-document boost ────────────────────────────────────────────────────
# Matches document-code-like tokens, e.g. "CSR-CLIENT-2023", "RFA-REG-01",
# "IL-INS-03" — uppercase letters followed by 1-4 dash-separated alphanumeric
# groups.
_DOC_CODE_RE = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+){1,4}\b")

# ── Source-document labeling trigger ────────────────────────────────────────
# Matches revision/version vocabulary (Romanian + English), e.g. "revizia",
# "rev.", "versiune", "current revision". Source labels help the LLM compare
# *which* document a chunk's content belongs to for cross-document revision
# questions, but for plain single-document lookups they can backfire: a model
# sees "Source: OTHER-DOC.txt" on the chunk holding the answer and concludes
# that fact isn't "conform <document named in the question>" even though the
# question didn't actually ask to restrict to that document's own text. Only
# attach source labels when the question is about revisions/versions, where
# that attribution is actually load-bearing.
_REVISION_QUERY_RE = re.compile(
    r"rev(?:izi[ae]|ision)?[.\s]|versiune|version|actual[ăa]|current",
    re.IGNORECASE,
)


def _needs_source_labels(question: str) -> bool:
    """True if source/provenance labels help more than they hurt here.

    Two cases where a chunk's `Source: <filename>` is load-bearing:
    - Revision/version comparison questions (_REVISION_QUERY_RE).
    - Questions that explicitly name 2+ documents (_DOC_CODE_RE) — the LLM
      needs to know which named document each chunk actually belongs to in
      order to scope its answer to the documents the question asked about,
      rather than picking up a same-topic fact from an unrelated document
      that happens to outrank the correct chunk.
    """
    if _REVISION_QUERY_RE.search(question):
        return True
    return len(set(_DOC_CODE_RE.findall(question))) >= 2


class LocalSearch:
    def __init__(self):
        self._cfg            = get_settings().retrieval
        self._neo4j          = get_neo4j()
        self._embedder       = Embedder()
        self._bm25           = HybridBM25Search()
        self._reranker       = CrossEncoderReranker(
            top_k=self._cfg.get("rerank_top_k", 5)
        )
        self._gnn = GNNScorer(
            gnn_type                  = self._cfg.get("gnn_type", "gat"),
            num_layers                = self._cfg.get("gnn_layers", 2),
            alpha                     = self._cfg.get("gnn_alpha", 0.9),
            beta                      = self._cfg.get("gnn_beta",  0.1),
            edge_confidence_threshold = self._cfg.get("gnn_edge_confidence_threshold", 0.7),
            confidence_half_life_days = self._cfg.get("gnn_confidence_half_life_days", 0),
        )
        self._adaptive_weights     = self._cfg.get("gnn_adaptive_weights", True)
        self._use_authority_weights = self._cfg.get("authority_weighting_enabled", True)
        self._authority_svc        = DocumentAuthorityService(self._neo4j)
        self._use_session_ctx      = self._cfg.get("session_context_enabled", True)
        self._session_ctx        = get_session_context() if self._use_session_ctx else None

    async def search(
        self,
        question: str,
        session_id: str = "",
        tenant: str = "default",
    ) -> dict:
        """Run the 6-step retrieval pipeline.

        Returns a dict with keys:
          chunks               — scored, sorted chunk list
          entities             — entity context rows
          referenced_entities  — entity names seen in this result set
          referenced_chunks    — chunk IDs in this result set

        Session recording is intentionally NOT done here.  The answer is only
        available after LLM generation; callers should use the returned
        referenced_* fields to call session_ctx.record_turn() with the real
        answer once it is known.
        """
        top_k      = self._cfg.get("local_top_k", 10)
        hops       = self._cfg.get("multihop_depth", 2)
        use_bm25   = self._cfg.get("bm25_enabled", True)
        use_rerank = self._cfg.get("reranker_enabled", True)
        use_gnn    = self._cfg.get("gnn_enabled", True)

        # ── Session context: resolve ambiguous follow-up queries ──────────────
        # Only runs when session_context_enabled=true in config AND a session_id
        # is provided.  Query enrichment uses prior turns; recording happens in
        # the caller (hybrid_retriever) once the real answer is available.
        enriched_question = question
        if self._use_session_ctx and self._session_ctx and session_id:
            enriched_question = await self._session_ctx.enrich_query(session_id, question)
            if enriched_question != question:
                log.info("local_search.query_enriched", session_id=session_id)

        # Step 1 — vector ANN (skipped when vector_search_enabled=false, e.g. OpenAI quota exhausted)
        use_vector = self._cfg.get("vector_search_enabled", True)
        embedding: list[float] | None = None
        if use_vector:
            embedding     = await self._embedder.embed_text(enriched_question)
            vector_chunks = await self._neo4j.vector_search_chunks(
                embedding,
                top_k=top_k,
                tenant=tenant,
            )
        else:
            log.info("local_search.vector_skipped", reason="vector_search_enabled=false")
            vector_chunks = []

        # Step 2 — BM25 + RRF fusion
        if use_bm25:
            fused_chunks = await self._bm25.search(
                query=enriched_question,
                vector_chunks=vector_chunks,
                top_k=top_k,
                tenant=tenant,
            )
        else:
            fused_chunks = vector_chunks

        # Step 3 — cross-encoder reranking
        if use_rerank and fused_chunks:
            seed_chunks = await self._reranker.rerank(enriched_question, fused_chunks)

            # RRF floor: the cross-encoder can drop a well-fused chunk
            # entirely — not just the #1 RRF chunk. Two known failure modes:
            # (a) non-English text (MS MARCO is English-trained) even when
            # it's a lexically-perfect match; (b) a chunk whose *content* is
            # a short numeric/list-style fact (e.g. "Quarterly media budget:
            # $2,400,000") scored low against a verbose neighboring section,
            # even in English, because the cross-encoder favors longer prose
            # matches over dense fact fragments. Guarantee the top
            # `rrf_floor_top_n` RRF-fused chunks a seed slot regardless of
            # what the cross-encoder did with them, dropping the weakest
            # reranked seeds to keep the seed count at rerank_top_k. This
            # runs BEFORE the lexical-diversity step below so a same-document
            # duplicate it inserts is naturally deduped by that step exactly
            # like any other same-document duplicate — inserting after would
            # need its own eviction logic that fights lexical-diversity's
            # document-coverage guarantee.
            floor_top_n = self._cfg.get("rrf_floor_top_n", 2)
            missing_floor = [
                c for c in fused_chunks[:floor_top_n]
                if seed_chunks and not any(s["chunk_id"] == c["chunk_id"] for s in seed_chunks)
            ]
            floored_ids: set[str] = {c["chunk_id"] for c in missing_floor}
            if missing_floor:
                floored = []
                for c in missing_floor:
                    c = dict(c)
                    c["rerank_score"] = c.get("score", 0.0)
                    floored.append(c)
                keep = max(0, len(seed_chunks) - len(floored))
                seed_chunks = floored + seed_chunks[:keep]

            # Preserve lexical evidence from distinct source documents. The
            # MS MARCO cross-encoder is English-trained and can demote strong
            # Romanian BM25 matches; repeated chunks from one long document
            # can then occupy nearly every seed slot. Keep the best RRF-fused
            # chunk from the configured number of distinct documents, then
            # fill remaining slots from the cross-encoder ranking.
            min_seed_docs = self._cfg.get("lexical_seed_min_documents", 0)
            if min_seed_docs > 1 and len(seed_chunks) > 1:
                fused_filenames = await self._neo4j.get_chunk_filenames(
                    [c["chunk_id"] for c in fused_chunks], tenant=tenant
                )
                lexical_diverse: list[dict] = []
                seen_docs: set[str] = set()
                for chunk in fused_chunks:
                    filename = fused_filenames.get(chunk["chunk_id"])
                    if not filename or filename in seen_docs:
                        continue
                    lexical_diverse.append(chunk)
                    seen_docs.add(filename)
                    if len(lexical_diverse) >= min_seed_docs:
                        break

                if len(lexical_diverse) > 1:
                    selected_ids = {c["chunk_id"] for c in lexical_diverse}
                    remaining = [
                        c for c in seed_chunks if c["chunk_id"] not in selected_ids
                    ]
                    seed_chunks = (lexical_diverse + remaining)[:len(seed_chunks)]

            # Promote any floored chunk that survived the steps above back to
            # the front (stable — relative order otherwise unchanged). This is
            # rank-only, no membership change, so it can't undo the
            # lexical-diversity document-coverage guarantee above: it only
            # reorders chunks that are already seed members. Needed because
            # the downstream GNN blend scores seed rank via 1.0-(rank/n_seed);
            # a floored chunk left mid-list by lexical-diversity's reordering
            # can score *worse* than a plain hop chunk's raw vector/BM25
            # score, silently undoing the floor's whole purpose.
            if floored_ids:
                promoted = [c for c in seed_chunks if c["chunk_id"] in floored_ids]
                rest = [c for c in seed_chunks if c["chunk_id"] not in floored_ids]
                seed_chunks = promoted + rest
        else:
            seed_chunks = fused_chunks

        # Named-document boost: if the question explicitly names a document
        # code (e.g. "RFA-REG-01"), guarantee that document's best-matching
        # chunk a seed slot. The cross-encoder can score a topically-correct
        # but lexically-dissimilar chunk as irrelevant even when the question
        # names its source document directly — trust the explicit reference
        # over the reranker in that case.
        if embedding is not None and seed_chunks:
            doc_codes = _DOC_CODE_RE.findall(enriched_question)
            if doc_codes:
                seed_id_set = {c["chunk_id"] for c in seed_chunks}
                filenames = await self._neo4j.get_document_filenames(tenant=tenant)

                # Prefer a chunk already proven relevant by BM25/RRF fusion
                # (fused_chunks) over a fresh whole-document cosine search:
                # the cross-encoder may have dropped it, but it was lexically
                # matched against the actual question, unlike a generic
                # "most similar chunk in this document" pick.
                non_seed_fused = [
                    c for c in fused_chunks if c["chunk_id"] not in seed_id_set
                ]
                fused_filenames = await self._neo4j.get_chunk_filenames(
                    [c["chunk_id"] for c in non_seed_fused], tenant=tenant
                )

                for code in doc_codes:
                    matched_filenames = [
                        f for f in filenames if f.upper().startswith(code)
                    ]
                    if not matched_filenames:
                        continue

                    best = None
                    for c in non_seed_fused:
                        if fused_filenames.get(c["chunk_id"]) in matched_filenames:
                            best = c
                            break

                    if best is None:
                        for filename in matched_filenames:
                            best = await self._neo4j.get_best_chunk_for_document(
                                filename, embedding, tenant=tenant
                            )
                            if best:
                                break

                    if best and best["chunk_id"] not in seed_id_set:
                        best = dict(best)
                        best["rerank_score"] = best.get("score", 0.0)
                        # Prepend (not append): _text_score is rank-based —
                        # appending puts it at the worst seed rank
                        # (text_score ≈ 0.2), which an explicit document
                        # reference shouldn't suffer.
                        seed_chunks = [best] + seed_chunks[:-1]
                        seed_id_set.add(best["chunk_id"])

        seed_ids = [c["chunk_id"] for c in seed_chunks]

        # Step 4 — multi-hop graph traversal
        # Semantic blend: rank hop chunks by (1-w)·path_score + w·cos(chunk, query)
        # BEFORE the multihop_top_k cap, so the cap keeps query-relevant chunks
        # rather than just topologically-cheap ones. Cosine runs inside Neo4j —
        # no embeddings cross the wire. Requires the query embedding, so it
        # degrades to pure path-score ranking when vector search is disabled.
        sem_weight = self._cfg.get("multihop_semantic_weight", 0.0)
        hop_chunks = await self._neo4j.get_multihop_chunks(
            seed_ids,
            hops=hops,
            tenant=tenant,
            query_embedding=embedding if sem_weight > 0 else None,
            semantic_weight=sem_weight,
        )

        # De-dupe by chunk_id as we go (not just against seed_ids) — multiple
        # entities/paths in the same hop-traversal frequently converge on the
        # same chunk, so get_multihop_chunks() can return one chunk_id dozens
        # of times. Without updating `seen` per accepted chunk, hop_top_k caps
        # on a pool that's mostly repeats of a handful of chunks, starving out
        # distinct-but-single-occurrence chunks that never get a seed slot.
        hop_top_k = self._cfg.get("multihop_top_k", 50)
        seen: set[str] = set(seed_ids)
        extra_chunks: list[dict] = []
        for c in hop_chunks:
            if c["chunk_id"] in seen:
                continue
            seen.add(c["chunk_id"])
            extra_chunks.append(c)
            if len(extra_chunks) >= hop_top_k:  # cap before GNN — prevents full-corpus scoring
                break
        all_chunks   = seed_chunks + extra_chunks
        all_ids      = [c["chunk_id"] for c in all_chunks]

        # Step 5 — GNN scoring with authority-weighted edges (requires query embedding)
        if use_gnn and all_chunks and embedding is not None:
            alpha, beta = _adaptive_weights(
                enriched_question,
                self._cfg.get("gnn_alpha", 0.9),
                self._cfg.get("gnn_beta",  0.1),
            ) if self._adaptive_weights else (
                self._cfg.get("gnn_alpha", 0.9),
                self._cfg.get("gnn_beta",  0.1),
            )

            chunk_entities, entity_edges = await asyncio.gather(
                self._neo4j.get_chunk_entity_embeddings(all_ids),
                _fetch_subgraph_edges(self._neo4j, all_ids, tenant),
            )

            # Apply document authority weights to edge confidence (config-gated)
            if self._use_authority_weights:
                entity_edges = await self._authority_svc.apply_authority_weights(
                    entity_edges
                )

            loop = asyncio.get_running_loop()
            all_chunks = await loop.run_in_executor(
                None,
                lambda: self._gnn.score(
                    query_vec      = embedding,
                    chunks         = all_chunks,
                    chunk_entities = chunk_entities,
                    entity_edges   = entity_edges,
                    alpha          = alpha,
                    beta           = beta,
                ),
            )

        # Attach source document filenames so the LLM can attribute claims to
        # a specific document/revision — needed for cross-document questions
        # (e.g. "which revision of X is referenced in Y, and is it current?")
        # where the answer hinges on knowing which chunk came from which doc.
        chunk_filenames = await self._neo4j.get_chunk_filenames(all_ids, tenant=tenant)
        needs_source_labels = _needs_source_labels(enriched_question)
        for chunk in all_chunks:
            filename = chunk_filenames.get(chunk["chunk_id"])
            if filename:
                chunk["_doc_name"] = filename.replace(".txt", "")
                if needs_source_labels:
                    chunk["source"] = filename

        # Step 6 — entity context
        entities = await self._neo4j.get_entity_neighbors(all_ids, tenant=tenant)

        # Collect the entity/chunk references so the caller can record the
        # session turn once the real LLM answer is available.
        referenced_entities = list({
            e.get("entity", "") for e in entities if e.get("entity")
        })

        log.info(
            "local_search.done",
            vector_chunks=len(vector_chunks),
            fused_chunks=len(fused_chunks),
            reranked_chunks=len(seed_chunks),
            hop_chunks=len(extra_chunks),
            total_chunks=len(all_chunks),
            entities=len(entities),
            hops=hops,
            bm25=use_bm25,
            reranker=use_rerank,
            gnn=use_gnn,
            session_ctx=self._use_session_ctx,
            session_id=session_id or "none",
        )
        return {
            "chunks": all_chunks,
            "entities": entities,
            # Returned so hybrid_retriever can record the turn with the real answer
            "referenced_entities": referenced_entities,
            "referenced_chunks": all_ids,
        }


# ── helpers ───────────────────────────────────────────────────────────────────

async def _fetch_subgraph_edges(
    neo4j,
    chunk_ids: list[str],
    tenant: str = "default",
) -> list[dict]:
    """Helper: get entity (name, type) pairs from chunks then fetch edges.

    Passing full (name, type) pairs instead of names alone prevents ambiguous
    MATCH results when the same tenant has two entities with identical names
    but different types (e.g. "Apple" as ORG vs. PRODUCT).
    """
    rows = await neo4j.get_chunk_entity_embeddings(chunk_ids)
    seen: set[tuple[str, str]] = set()
    entities: list[dict] = []
    for r in rows:
        key = (r["entity_name"], r["entity_type"])
        if key not in seen:
            seen.add(key)
            entities.append({"name": r["entity_name"], "type": r["entity_type"]})
    if not entities:
        return []
    return await neo4j.get_entity_relations_subgraph(entities, tenant=tenant)
