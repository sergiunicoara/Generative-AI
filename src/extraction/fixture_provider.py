"""Deterministic fixture extractor — byte-stable output, no LLM/network call.
This is not a stand-in for real extraction quality (see llm_provider.py for
that); it's a fixed, pure function of input text used for tests and for demos
where no real LLM key is configured (pyproject.toml's own open item — no
embedding/LLM provider is pinned yet). §16: 'deterministic fake extraction is
byte-stable' — true here because there is no randomness or hidden state at all,
just regex matching against each segment's text independently.
"""

from __future__ import annotations

import re

from src.domain.enums import Polarity
from src.extraction.provider import ExtractedAssertion, ExtractionInput, ExtractionResult

_NEGATION_CUES = ("not ", "n't ", "no longer ", "never ", "don't ", "doesn't ")
_HYPOTHETICAL_CUES = ("if ", "would ", "could ", "hypothetically", "suppose")

# (pattern, predicate, object_text) — order matters: first match per segment wins.
_RULES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"pric(e|ing)", re.I), "RAISED_OBJECTION", "pricing"),
    (re.compile(r"security", re.I), "HAS_BLOCKER", "security"),
    (re.compile(r"follow up by \w+ \d{1,2}", re.I), "HAS_ACTION_ITEM", "follow_up"),
    (re.compile(r"volks ?wagen", re.I), "MENTIONS_ORG", "volkswagen"),
]


def _detect_polarity(text: str) -> Polarity:
    lowered = text.lower()
    if any(cue in lowered for cue in _NEGATION_CUES):
        return Polarity.NEGATED
    if any(cue in lowered for cue in _HYPOTHETICAL_CUES):
        return Polarity.HYPOTHETICAL
    return Polarity.AFFIRMED


class FixtureExtractionProvider:
    async def extract(self, inputs: list[ExtractionInput]) -> list[ExtractionResult]:
        results = []
        for item in inputs:
            assertions: list[ExtractedAssertion] = []
            for seg in item.segments:
                for pattern, predicate, object_text in _RULES:
                    match = pattern.search(seg.text)
                    if not match:
                        continue
                    assertions.append(ExtractedAssertion(
                        segment_id=seg.segment_id,
                        speaker_label=seg.speaker_label,
                        predicate=predicate,
                        object_text=object_text,
                        polarity=_detect_polarity(seg.text),
                        evidence_char_start=match.start(),
                        evidence_char_end=match.end(),
                    ))
            results.append(ExtractionResult(window_id=item.window.window_id, assertions=assertions))
        return results
