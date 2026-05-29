"""Entity + relation extraction from text chunks using Gemini with structured output."""

from __future__ import annotations

import asyncio
import json
import re

from google import genai
from google.genai import types as genai_types
import structlog

from graphrag.core.config import get_settings
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
        self._client = genai.Client(api_key=cfg.google_api_key)
        self._model_name = cfg.gemini_ingest_model
        self._gen_config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        )
        self._entity_types = cfg.ingestion.get(
            "entity_types", ["PERSON", "ORG", "PRODUCT", "CONCEPT", "LOCATION", "EVENT"]
        )

    async def extract(self, chunk: Chunk) -> tuple[list[Entity], list[Relation]]:
        prompt = _EXTRACT_PROMPT.format(
            entity_types=", ".join(self._entity_types),
            text=chunk.text,
        )

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=self._gen_config,
            ),
        )

        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
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
                        confidence=float(r.get("confidence", 1.0)),
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
