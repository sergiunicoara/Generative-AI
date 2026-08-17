"""Query-class routing and bounded fallback plans."""

from __future__ import annotations


QUERY_CLASSES = ("factoid", "relational", "contradiction", "multi_hop", "negative")


def classify_query(question: str) -> str:
    text = question.lower()
    if any(word in text for word in ("conflict", "contradict", "disagree", "match")):
        return "contradiction"
    if any(word in text for word in ("how does", "connected", "relationship", "between")):
        return "relational"
    if any(word in text for word in ("across", "multiple", "compare", "steps", "chain")):
        return "multi_hop"
    # Added 2026-08-17 for NEG-03 (see docs/audit-2026-08-13.md, "What's left").
    # Existence/negation-check questions ("is there X in this corpus?") need the
    # SAME wider top_k as "contradiction" and for the same reason: the fact that
    # answers them correctly is often a single contrasting document that scores
    # low on a literal cross-encoder relevance match against the query's own
    # wording (e.g. a query mentioning "FAA" scores an EASA-only chunk as
    # low-relevance even though it's exactly the contrasting fact needed).
    # top_k=5 (the "factoid" default) was cutting that chunk right at the
    # boundary — verified live via a standalone reranker trace before this fix
    # (rank 8 of 10 pre-cutoff, cross-encoder score -0.10 vs the FAA-only
    # chunks' 1.2-4.0). Narrow trigger by design: only 1 of the 34 golden-set
    # questions matches this phrasing (checked before adding), so this should
    # not reclassify anything that isn't already failing on this pattern.
    if any(phrase in text for phrase in ("is there", "any evidence")):
        return "negative"
    return "factoid"


def retrieval_plan(question: str) -> dict:
    query_class = classify_query(question)
    plans = {
        "factoid": {"mode": "local", "top_k": 5, "fallback": "hybrid"},
        "relational": {"mode": "hybrid", "top_k": 8, "fallback": "agentic"},
        "contradiction": {"mode": "hybrid", "top_k": 8, "fallback": "review"},
        "multi_hop": {"mode": "global", "top_k": 10, "fallback": "agentic"},
        # top_k=10, NOT 8 (unlike contradiction/relational) — found live
        # 2026-08-17 while wiring this through hybrid_retriever.py: the
        # effective local_top_k every query has ACTUALLY been getting is
        # settings.yml's global default (10), because query_planner_enabled
        # computed a per-class top_k that never reached LocalSearch (a
        # separate wiring bug, fixed the same day — see
        # hybrid_retriever.py's inline comment). Using 8 here first silently
        # narrowed the candidate pool below that pre-existing effective
        # default and dropped the exact EASA chunk this class exists to
        # rescue (verified live: chunk present at RRF-fusion rank 10 with
        # top_k=10, absent with top_k=8). 10 matches the passive baseline
        # exactly for local_top_k while still widening rerank_top_k from its
        # default of 5 up to 10 — the actual fix this class needs.
        "negative": {"mode": "hybrid", "top_k": 10, "fallback": "review"},
    }
    return {"query_class": query_class, **plans[query_class]}
