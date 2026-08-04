"""§8 Stage A — deterministic resolution rules.

A rule auto-links only when it returns EXACTLY ONE eligible entity inside the
workspace. This module makes that uniqueness decision; it doesn't query Neo4j
itself — callers supply the already tenant-scoped candidate id list (e.g. from
src/resolution/candidates.py's CandidateGenerator.exact_name_candidates()).
Duplicate exact names in the same workspace correctly yield more than one
candidate id, so this returns None rather than picking arbitrarily (§16:
'duplicate exact names do not deterministic-link').
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DeterministicRule(StrEnum):
    A1_EXACT_EXTERNAL_ID = "A1_EXACT_EXTERNAL_ID"
    A2_EXACT_NORMALIZED_EMAIL = "A2_EXACT_NORMALIZED_EMAIL"
    A3_EXACT_CANONICAL_NAME = "A3_EXACT_CANONICAL_NAME"
    A4_EXACT_APPROVED_ALIAS = "A4_EXACT_APPROVED_ALIAS"


@dataclass(frozen=True)
class DeterministicMatch:
    rule: DeterministicRule
    entity_id: str


def resolve_deterministic(rule: DeterministicRule, candidate_entity_ids: list[str]) -> DeterministicMatch | None:
    unique_ids = set(candidate_entity_ids)
    if len(unique_ids) == 1:
        return DeterministicMatch(rule=rule, entity_id=next(iter(unique_ids)))
    return None
