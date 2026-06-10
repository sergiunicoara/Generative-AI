"""Assemble final context string from local + global retrieval results."""

from __future__ import annotations


class ContextBuilder:
    def build(
        self,
        local_results: dict,
        global_results: dict,
        weights: tuple[float, float] = (0.6, 0.4),
        top_k: int = 5,
    ) -> tuple[str, list[str]]:
        sections: list[str] = []
        citations: list[str] = []

        # Local: top-k chunks (weighted by score)
        chunks = local_results.get("chunks", [])
        local_weight, global_weight = weights
        chunks_sorted = sorted(
            chunks, key=lambda c: c.get("score", 0) * local_weight, reverse=True
        )[:top_k]

        for chunk in chunks_sorted:
            sections.append(f"[Chunk {chunk['chunk_id']}]\n{chunk['text']}")
            citations.append(chunk["chunk_id"])

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

        # Global: community-synthesized answer
        synthesized = global_results.get("synthesized_answer", "")
        if synthesized:
            sections.append(f"Community knowledge:\n{synthesized}")

        context = "\n\n---\n\n".join(sections)
        return context, list(dict.fromkeys(citations))  # deduplicate preserving order
