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

        context = "\n\n---\n\n".join(sections)
        return context, list(dict.fromkeys(citations))  # deduplicate preserving order
