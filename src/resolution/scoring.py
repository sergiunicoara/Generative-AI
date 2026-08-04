"""§8 probabilistic scoring — the formula, verbatim:

    lexical  = normalized fuzzy similarity
    semantic = cosine similarity over contextual entity representations
    base     = configured blend(lexical, semantic)
    rel_bonus = capped sum of independent relational signals
    final    = min(base + rel_bonus, 1.0)
    margin   = top1_final - top2_final

'Each signal fires at most once and appears in the explanation' — relational_signals
is a set (dedupes by construction) and is carried on ScoredCandidate for the
explanation.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

RELATIONAL_SIGNAL_BONUS = 0.06


@dataclass(frozen=True)
class ScoredCandidate:
    entity_id: str
    entity_type: str
    name: str
    lexical: float
    semantic: float | None
    base: float
    rel_bonus: float
    final: float
    relational_signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class RankingResult:
    ranked: list[ScoredCandidate]  # sorted by final, descending
    margin: float  # top1.final - top2.final; top1.final itself if no runner-up


def lexical_score(mention_surface: str, candidate_name: str) -> float:
    return fuzz.ratio(mention_surface.lower(), candidate_name.lower()) / 100.0


def blend(lexical: float, semantic: float | None, *, lexical_weight: float = 0.6) -> float:
    """Falls back to lexical-only when no semantic score is available — no
    embedding provider is pinned yet (pyproject.toml's own open item)."""
    if semantic is None:
        return lexical
    return lexical_weight * lexical + (1 - lexical_weight) * semantic


def score_candidate(
    *,
    entity_id: str,
    entity_type: str,
    name: str,
    mention_surface: str,
    semantic: float | None = None,
    relational_signals: frozenset[str] = frozenset(),
    lexical_weight: float = 0.6,
    max_rel_bonus: float = 0.18,
) -> ScoredCandidate:
    lex = lexical_score(mention_surface, name)
    base = blend(lex, semantic, lexical_weight=lexical_weight)
    rel_bonus = min(len(relational_signals) * RELATIONAL_SIGNAL_BONUS, max_rel_bonus)
    final = min(base + rel_bonus, 1.0)
    return ScoredCandidate(
        entity_id=entity_id, entity_type=entity_type, name=name,
        lexical=lex, semantic=semantic, base=base, rel_bonus=rel_bonus, final=final,
        relational_signals=tuple(sorted(relational_signals)),
    )


def rank_candidates(scored: list[ScoredCandidate]) -> RankingResult:
    ordered = sorted(scored, key=lambda c: c.final, reverse=True)
    if not ordered:
        return RankingResult(ranked=[], margin=0.0)
    top1 = ordered[0].final
    top2 = ordered[1].final if len(ordered) > 1 else 0.0
    return RankingResult(ranked=ordered, margin=top1 - top2)
