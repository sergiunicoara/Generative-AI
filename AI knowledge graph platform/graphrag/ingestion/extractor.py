"""Entity + relation extraction from text chunks using Groq (llama-3.3-70b) with JSON output.

Text generation is routed through ``graphrag.core.llm_client.get_llm`` (Groq);
embeddings stay on Gemini.  The LLM is asked for strict JSON (``json_mode=True``)
and the parsed relations have their confidence clamped to ``[0, 1]`` so the
Bayesian merge formula downstream cannot be corrupted by out-of-range values.
"""

from __future__ import annotations

import asyncio
import json
import re

import structlog

from graphrag.core.config import get_settings
from graphrag.core.llm_client import get_llm
from graphrag.core.models import Chunk, Entity, Relation

log = structlog.get_logger(__name__)

_EXTRACT_PROMPT = """\
Extract entities and relations from the text below.

Entity types to extract: {entity_types}

Return ONLY valid JSON in this exact format:
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "one-sentence description"}}
  ],
  "relations": [
    {{
      "source": "entity name",
      "target": "entity name",
      "relation": "VERB_RELATION",
      "confidence": 0.95
    }}
  ]
}}

confidence is a float [0.0, 1.0] reflecting how clearly the text states this relationship.
Use 0.9+ for explicit statements, 0.6-0.9 for strong implications, below 0.6 for weak inference.

Text:
{text}
"""


class Extractor:
    def __init__(self):
        cfg = get_settings()
        self._model_name = cfg.groq_model
        self._entity_types = cfg.ingestion.get(
            "entity_types", ["PERSON", "ORG", "PRODUCT", "CONCEPT", "LOCATION", "EVENT"]
        )

    async def extract(self, chunk: Chunk) -> tuple[list[Entity], list[Relation]]:
        prompt = _EXTRACT_PROMPT.format(
            entity_types=", ".join(self._entity_types),
            text=chunk.text,
        )

        raw = await get_llm().generate(prompt, json_mode=True)

        try:
            if not raw:
                log.warning("extractor.empty_response", chunk_id=chunk.id)
                return [], []
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            log.warning("extractor.parse_error", chunk_id=chunk.id)
            return [], []

        entities = [
            Entity(
                name=e["name"],
                type=e.get("type", "CONCEPT"),
                description=e.get("description", ""),
                source_chunk_ids=[chunk.id],
                source_doc_id=chunk.document_id,
                extraction_model=self._model_name,
                prompt_version="v1",
            )
            for e in data.get("entities", [])
            if e.get("name")
        ]

        # Build name→entity map for relation linking
        entity_map = {e.name: e for e in entities}

        relations = []
        for r in data.get("relations", []):
            src = entity_map.get(r.get("source", ""))
            tgt = entity_map.get(r.get("target", ""))
            if src and tgt and src.id != tgt.id:
                # Approximate span: find source entity name in chunk text
                span_start: int | None = None
                span_end: int | None = None
                try:
                    pos = chunk.text.find(r.get("source", ""))
                    if pos >= 0:
                        span_start = pos
                        span_end = pos + len(r.get("source", ""))
                except (ValueError, AttributeError):
                    pass   # span computation is best-effort; missing span is harmless

                relations.append(
                    Relation(
                        source_entity_id=src.id,
                        target_entity_id=tgt.id,
                        relation=r.get("relation", "RELATED_TO"),
                        # Clamp to [0, 1] — LLMs occasionally return values outside
                    # this range; the Bayesian merge formula breaks for out-of-range
                    # inputs (confidence > 1 → merged confidence > 1 → corrupts graph).
                    confidence=max(0.0, min(1.0, float(r.get("confidence", 1.0)))),
                        source_chunk_id=chunk.id,
                        extraction_model=self._model_name,
                        prompt_version="v1",
                        chunk_span_start=span_start,
                        chunk_span_end=span_end,
                    )
                )

        # ── Ontology validation ───────────────────────────────────────
        try:
            from graphrag.graph.ontology_registry import get_ontology_registry
            registry = get_ontology_registry()
            if registry._loaded:
                registry.validate_extraction(entities, relations)
        except ImportError:
            pass  # registry not yet available during cold-start — skip silently

        log.info(
            "extractor.done",
            chunk_id=chunk.id,
            entities=len(entities),
            relations=len(relations),
        )
        return entities, relations
