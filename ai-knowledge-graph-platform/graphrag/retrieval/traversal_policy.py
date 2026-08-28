"""Query-conditioned limits for bounded graph expansion.

The policy is deliberately deterministic and cheap.  It gives the existing
Neo4j traversal an explicit budget instead of treating every query as a fixed
two-hop expansion.  The feature is opt-in at the caller because each tenant
must validate its own retrieval/latency frontier before changing defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

from graphrag.retrieval.query_planner import classify_query


@dataclass(frozen=True)
class TraversalPolicy:
    """Bounded graph-expansion policy derived only from the query shape."""

    query_class: str
    max_hops: int
    beam_width: int
    per_seed_cap: int
    total_cap: int
    min_path_score: float
    min_relative_gain: float


_POLICIES: dict[str, dict[str, int | float]] = {
    # Precise lookups should prefer seed evidence and avoid noisy neighbours.
    "factoid": {
        "max_hops": 1, "beam_width": 6, "per_seed_cap": 60,
        "total_cap": 80, "min_path_score": 0.20, "min_relative_gain": 0.35,
    },
    "relational": {
        "max_hops": 3, "beam_width": 12, "per_seed_cap": 120,
        "total_cap": 180, "min_path_score": 0.14, "min_relative_gain": 0.25,
    },
    "contradiction": {
        "max_hops": 2, "beam_width": 10, "per_seed_cap": 100,
        "total_cap": 140, "min_path_score": 0.12, "min_relative_gain": 0.20,
    },
    "multi_hop": {
        "max_hops": 4, "beam_width": 16, "per_seed_cap": 160,
        "total_cap": 240, "min_path_score": 0.10, "min_relative_gain": 0.18,
    },
    # Negation/existence questions need lexical contrast, not broad expansion.
    "negative": {
        "max_hops": 0, "beam_width": 0, "per_seed_cap": 0,
        "total_cap": 0, "min_path_score": 1.0, "min_relative_gain": 1.0,
    },
}


def build_traversal_policy(
    question: str,
    *,
    configured_max_hops: int,
    configured_top_k: int,
    enabled: bool,
    max_depth_cap: int = 4,
) -> TraversalPolicy:
    """Return an adaptive policy or an exact representation of legacy limits."""
    query_class = classify_query(question)
    if not enabled:
        return TraversalPolicy(
            query_class=query_class,
            max_hops=max(0, configured_max_hops),
            beam_width=max(0, configured_top_k),
            per_seed_cap=200,
            total_cap=500,
            min_path_score=0.0,
            min_relative_gain=0.0,
        )

    template = _POLICIES[query_class]
    return TraversalPolicy(
        query_class=query_class,
        max_hops=min(int(template["max_hops"]), max(0, max_depth_cap)),
        beam_width=min(int(template["beam_width"]), max(0, configured_top_k)),
        per_seed_cap=int(template["per_seed_cap"]),
        total_cap=int(template["total_cap"]),
        min_path_score=float(template["min_path_score"]),
        min_relative_gain=float(template["min_relative_gain"]),
    )


def select_traversal_candidates(
    candidates: list[dict], policy: TraversalPolicy,
) -> list[dict]:
    """Keep a unique, high-quality beam and stop after evidence gain collapses.

    Neo4j orders candidates by path score.  Once a candidate falls below both
    the absolute score floor and its score relative to the best candidate,
    remaining candidates cannot restore evidence quality, so expansion stops.
    """
    if policy.beam_width <= 0:
        return []
    selected: list[dict] = []
    seen: set[str] = set()
    best_score: float | None = None
    for candidate in candidates:
        chunk_id = str(candidate.get("chunk_id", ""))
        if not chunk_id or chunk_id in seen:
            continue
        score = max(0.0, float(candidate.get("score", candidate.get("path_score", 0.0)) or 0.0))
        if best_score is None:
            best_score = score
        elif score < policy.min_path_score and score < best_score * policy.min_relative_gain:
            break
        seen.add(chunk_id)
        selected.append(candidate)
        if len(selected) >= policy.beam_width:
            break
    return selected


__all__ = ["TraversalPolicy", "build_traversal_policy", "select_traversal_candidates"]
