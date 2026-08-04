"""§7 — batch extraction interface. docs/plan.md's abstract signature is
`extract(windows: list[ExtractionWindow]) -> list[ExtractionResult]`; concretely,
an extractor needs the window's actual transcript text to do anything, and
ExtractionWindow (src/domain/conversation.py) deliberately carries only
segment_ids — text stays owned by TranscriptSegment (§5). ExtractionInput bundles
a window with its segments' text as the necessary carrier for that; it is an
extraction-layer type, not a persisted domain model.

LLMs extract typed information only — they never resolve identities, calculate
business scores, or write to Neo4j (Codex prompt non-negotiable rules). Note
what's absent from ExtractedAssertion below: no subject_id, no claim_id, no
Neo4j write. The ingestion pipeline (not the provider) turns this into a full
Claim with resolved identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from src.domain.conversation import ExtractionWindow
from src.domain.enums import Polarity


@dataclass(frozen=True)
class WindowSegmentText:
    segment_id: str
    speaker_label: str
    text: str


@dataclass(frozen=True)
class ExtractionInput:
    window: ExtractionWindow
    segments: list[WindowSegmentText]  # ordered, matching window.segment_ids


class ExtractedAssertion(BaseModel):
    segment_id: str
    speaker_label: str
    predicate: str
    object_text: str
    polarity: Polarity
    evidence_char_start: int
    evidence_char_end: int


class ExtractionResult(BaseModel):
    window_id: str
    assertions: list[ExtractedAssertion] = []


class ExtractionProvider(Protocol):
    async def extract(self, inputs: list[ExtractionInput]) -> list[ExtractionResult]: ...
