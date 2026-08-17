"""Assemble final context string from local + global retrieval results."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_NEAR_DUPLICATE_RATIO = 0.85


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _is_near_duplicate(text: str, seen_texts: list[str]) -> bool:
    """True if `text` is near-identical to a chunk already selected.

    Corpora with repeated boilerplate sections (e.g. a clause restated
    verbatim in several places in the same document) can produce several
    distinct chunk_ids whose text is effectively the same passage. Letting
    all of them occupy top_k slots wastes context budget on redundant
    information and crowds out a genuinely different document's chunk.
    """
    norm = _normalize(text)
    for other in seen_texts:
        if SequenceMatcher(None, norm[:300], other[:300]).ratio() >= _NEAR_DUPLICATE_RATIO:
            return True
    return False


class ContextBuilder:
    def build(
        self,
        local_results: dict,
        global_results: dict,
        weights: tuple[float, float] = (0.6, 0.4),
        top_k: int = 5,
        conflicts: list[dict] | None = None,
    ) -> tuple[str, list[str]]:
        sections: list[str] = []
        citations: list[str] = []

        # Local: top-k chunks, ranked by the GNN-blended final_score (falls back
        # to rerank_score / path_score for chunks the GNN scorer didn't touch).
        chunks = local_results.get("chunks", [])
        local_weight, global_weight = weights
        chunks_sorted = sorted(
            chunks,
            key=lambda c: c.get("final_score", c.get("rerank_score", c.get("score", 0))) * local_weight,
            reverse=True,
        )

        # Multi-hop traversal can reach the same chunk via multiple entity
        # paths, producing duplicate entries. Drop duplicates by chunk_id
        # (keeping the highest-ranked occurrence) BEFORE the top_k slice, so
        # a repeated chunk doesn't crowd out a distinct document's chunk.
        # Also drop near-duplicate TEXT across different chunk_ids — corpora
        # with a clause repeated verbatim in several places can otherwise
        # fill every top_k slot with the same passage.
        seen_chunk_ids: set[str] = set()
        seen_texts: list[str] = []
        deduped: list[dict] = []
        for chunk in chunks_sorted:
            if chunk["chunk_id"] in seen_chunk_ids:
                continue
            if _is_near_duplicate(chunk["text"], seen_texts):
                continue
            seen_chunk_ids.add(chunk["chunk_id"])
            seen_texts.append(_normalize(chunk["text"])[:300])
            deduped.append(chunk)
            if len(deduped) >= top_k:
                break

        for chunk in deduped:
            source = chunk.get("source")
            header = f"[Chunk {chunk['chunk_id']} | Source: {source}]" if source else f"[Chunk {chunk['chunk_id']}]"
            sections.append(f"{header}\n{chunk['text']}")
            doc_name = chunk.get("_doc_name") or (source.replace(".txt", "") if source else None)
            citations.append(doc_name if doc_name else chunk["chunk_id"])

        # Local: entity context
        entities = local_results.get("entities", [])
        if entities:
            entity_lines = []
            for e in entities[:5]:  # limit to 5 entities
                neighbors = ", ".join(e.get("neighbors", [])[:3])
                entity_lines.append(
                    f"{e['entity']} ({e['type']}): {e['description']}. Related: {neighbors}"
                )
            sections.append("Entity context:\n" + "\n".join(entity_lines))

        # Known graph relationships: entity/document edges established
        # elsewhere in the corpus — directly extracted or, notably, derived
        # by forward-chaining inference (e.g. supersedes_transitivity, see
        # inference_engine.py). Chunk text alone often states only pairwise
        # facts (A supersedes B, B supersedes C); the transitive fact (A
        # supersedes C) may exist only as an inferred graph edge, never as a
        # sentence any single chunk contains. Without surfacing it here as
        # explicit text, a prompt that instructs the model to answer "ONLY"
        # from context has no anchor for that fact and reasons "no direct
        # reference" even when the graph already establishes it. See
        # INF-01/CON-02 in evals/golden_set.json.
        entity_edges = local_results.get("entity_edges", [])
        if entity_edges:
            edge_lines = []
            for e in entity_edges[:10]:  # generous vs. the 5-cap above — chain
                                          # questions (e.g. MH-01) legitimately
                                          # need several hops to answer
                relation = e.get("relation")
                if not relation:
                    continue  # no label to report — nothing informative to add
                line = f"{e['src']} —{relation}→ {e['tgt']}"
                if e.get("source_type") == "inferred":
                    rule = e.get("inferred_by")
                    line += f" (inferred{f' via {rule}' if rule else ''})"
                edge_lines.append(line)
                # Register both endpoints as citations too, not just the prompt
                # text above. A fact surfaced ONLY here (e.g. a transitive
                # supersession chain that no single chunk states outright, see
                # INF-01/CON-02 in evals/golden_set.json) can ground a correct
                # answer while its endpoint document never independently
                # survives the chunk-citation top_k cutoff a few lines up —
                # leaving the citation list silently short even though the
                # answer is right. Extraction assigns entity `type` from an
                # open, LLM-chosen vocabulary (see extractor.py) with no fixed
                # "Document" label to gate on, so this citations the edge
                # unconditionally rather than filtering by type; the
                # dict.fromkeys dedup below already collapses overlap with
                # chunk-derived citations, and the [:10] cap above bounds it.
                citations.append(e["src"])
                citations.append(e["tgt"])
            if edge_lines:
                sections.append(
                    "Known graph relationships:\n" + "\n".join(edge_lines)
                )

        # Unresolved conflicts: an entity in this result set is the subject of
        # an open contradiction (see ContradictionDetector) — two sources
        # disagree on a fact. Surfaced explicitly here rather than left for
        # the LLM to notice on its own by spotting disagreeing chunk text,
        # which only works if both contradictory chunks happen to make top_k.
        if conflicts:
            conflict_lines = []
            for c in conflicts[:5]:  # same cap as entity context, above
                conflict_lines.append(
                    f"{c['src']} —{c['relation']}→ {c['tgt']} ({c['conflict_type']}): "
                    f"sources disagree, unresolved"
                )
            sections.append(
                "⚠ Unresolved conflicts:\n" + "\n".join(conflict_lines)
            )

        # Global: community-synthesized answer
        synthesized = global_results.get("synthesized_answer", "")
        if synthesized:
            sections.append(f"Community knowledge:\n{synthesized}")

        # Global-mode citations: community summaries carry no per-fact
        # provenance, so this is the representative document set attached by
        # GlobalSearch.search() (see Neo4jClient.get_community_source_documents),
        # not a claim that every one of these documents grounds every sentence
        # in the synthesized text — the same coarseness local search would have
        # if it cited "the corpus" instead of a specific chunk. Still strictly
        # better than the unconditional empty list every purely-global-mode
        # answer previously returned regardless of how well-grounded it was.
        for community in global_results.get("communities", []):
            citations.extend(community.get("source_documents", []))

        context = "\n\n---\n\n".join(sections)
        return context, list(dict.fromkeys(citations))  # deduplicate preserving order
