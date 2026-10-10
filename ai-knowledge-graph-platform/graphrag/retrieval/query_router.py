"""Deterministic query routing (plan Phase 4, decision D4).

Extends the keyword planner (graphrag/retrieval/query_planner.py) rather than
replacing it: the legacy classes stay as aliases and keep their measured
mode/top_k plans, so default retrieval is unchanged. The router adds the
requested taxonomy and, per route, what retrieval should do:

    FACTUAL_LOOKUP -> vector/BM25 hybrid over chunks, no graph expansion needed
    ENTITY_LOOKUP  -> entity resolution + 1-hop graph lookup
    RELATIONAL     -> graph traversal + hybrid retrieval
    MULTI_HOP      -> multi-hop traversal + hybrid retrieval
    AGGREGATION    -> structured (controlled, allowlisted) query path
    TEMPORAL       -> time-constrained retrieval (as_of)
    AMBIGUOUS      -> bounded agentic (IRCoT) fallback

No LLM is used. ``policy: observe`` (default) records the route and its reason
on every query but keeps the legacy retrieval behaviour; ``policy: enforce``
applies the route-specific behaviour. Enforcement is opt-in because its effect
on answer quality, latency and tokens has not been measured on the live stack
(docs/query-routing.md, evals/routing_eval_results.json).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum

from graphrag.retrieval.query_planner import classify_query, retrieval_plan


class Route(str, Enum):
    FACTUAL_LOOKUP = "FACTUAL_LOOKUP"
    ENTITY_LOOKUP = "ENTITY_LOOKUP"
    RELATIONAL = "RELATIONAL"
    MULTI_HOP = "MULTI_HOP"
    AGGREGATION = "AGGREGATION"
    TEMPORAL = "TEMPORAL"
    AMBIGUOUS = "AMBIGUOUS"


# Legacy planner class -> route (the legacy name stays available as `legacy_class`).
LEGACY_TO_ROUTE = {
    "factoid": Route.FACTUAL_LOOKUP,
    "negative": Route.FACTUAL_LOOKUP,
    "relational": Route.RELATIONAL,
    "contradiction": Route.RELATIONAL,
    "multi_hop": Route.MULTI_HOP,
}

_AGGREGATION = re.compile(
    r"\b(how many|number of|count of|count the|total (?:number|amount)|how much in total|"
    r"list (?:all|every)|which \w+ (?:have|has) no|without (?:any )?evidence|average|sum of)\b", re.I)
_TEMPORAL = re.compile(
    r"\b(as of|as at|at the time|before|after|since|until|between \d{4}|during|"
    r"in (?:19|20)\d{2}|on \d{4}-\d{2}-\d{2}|(?:19|20)\d{2}-\d{2}-\d{2}|"
    r"previous(?:ly)?|former(?:ly)?|historical(?:ly)?|was (?:valid|in force|current)|"
    r"superseded|which (?:revision|version) was|at that time)\b", re.I)
_ISO_DATE = re.compile(r"\b((?:19|20)\d{2}-\d{2}-\d{2})\b")
# Document identifiers that merely contain a date-shaped number (AD 2024-03-07, SB-2023-11-04).
# They name a document; they are not a time constraint.
_IDENTIFIER = re.compile(r"\b[A-Z]{2,}[\s\-.]?(?:19|20)\d{2}-\d{2}(?:-\d{2})?\b")
# Relation verbs that make "Who/Which/What <verb> ..." a graph question without saying "related".
_RELATION_QUESTION = re.compile(
    r"^\s*(?:who|whom|which|what)\b.*?\b(?:owns?|owned by|operat(?:es?|ors?|ed by)|suppl(?:y|ies|ier|iers|ied by)|"
    r"manufactur\w+|made by|depends? on|depend(?:ed|ing) on|connects? to|feeds?|oversee\w*|regulat(?:es|or)|"
    r"responsible for|subsidiar\w+|parent of|approv(?:es|ed by|ing)|leases?|acquir\w+)\b", re.I)
# Questions that compare which source governs a claim.
_AUTHORITY_COMPARE = re.compile(
    r"\b(?:authoritative|takes? precedence|prevails?|overrides?|conflicts? with|contradicts?)\b", re.I)
# Relation cues; two or more in one question means a chain of hops.
_HOP_CUE = re.compile(
    r"(?:\w+'s\b|\bsuppl(?:y|ies|ier|iers|ied by)\b|\bused (?:in|on|by)\b|\b(?:operated|owned|leased|made|"
    r"issued|serviced) (?:by|to)\b|\bacquired\b|\bsubsidiar\w+|\bservic(?:e|es|ed|ing)\b|"
    r"\bmanufacturer\b|\bcustomers?\b|\baffect\w*\b)", re.I)
_HOP_WORDS = re.compile(r"\b(?:through which|via|ultimately|intermediar\w+|indirectly|transitive\w*)\b", re.I)
_DEICTIC_ANY = re.compile(
    r"\b(?:it|this|that|they|them|these|those|the other|the second|the first|the same|that one|this one|"
    r"same for|other)\b", re.I)
_FILLER = {"can", "you", "please", "still", "again", "also", "then", "more", "other", "same", "second", "first",
           "one", "them", "explain", "about", "and", "so", "now"}
_ENTITY_LOOKUP = re.compile(
    r"^\s*(who is|who are|what is|what's|what are|tell me about|describe|define|"
    r"show (?:me )?(?:the )?(?:entity|profile|details) (?:for|of))\b", re.I)
_RELATION_WORDS = re.compile(r"\b(relat|connect|between|link|own|supplies|depends|affect|impact)", re.I)
_DEICTIC_ONLY = re.compile(r"^\s*(it|this|that|they|them|those|these|he|she|the same|that one)\b", re.I)
_STOP = {"the", "a", "an", "of", "is", "are", "what", "which", "who", "how", "it", "this", "that",
         "and", "or", "to", "in", "on", "for", "about", "me", "tell", "do", "does", "any"}

# What each route asks of retrieval when enforced.
ROUTE_BEHAVIOUR: dict[Route, dict] = {
    Route.FACTUAL_LOOKUP: {"graph_expansion": False, "multihop_hops": 0, "agentic_fallback": True},
    Route.ENTITY_LOOKUP: {"graph_expansion": True, "multihop_hops": 1, "agentic_fallback": True},
    Route.RELATIONAL: {"graph_expansion": True, "multihop_hops": None, "agentic_fallback": True},
    Route.MULTI_HOP: {"graph_expansion": True, "multihop_hops": None, "agentic_fallback": True},
    Route.AGGREGATION: {"graph_expansion": False, "multihop_hops": 0, "agentic_fallback": False},
    Route.TEMPORAL: {"graph_expansion": True, "multihop_hops": None, "agentic_fallback": False},
    Route.AMBIGUOUS: {"graph_expansion": True, "multihop_hops": None, "agentic_fallback": True},
}


@dataclass
class RouteDecision:
    route: Route
    reason: str
    legacy_class: str
    mode: str
    top_k: int
    fallback: str
    signals: list[str] = field(default_factory=list)
    structured_intent: str | None = None
    as_of: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["route"] = self.route.value
        d["behaviour"] = ROUTE_BEHAVIOUR[self.route]
        return d


def _valid_iso_dates(question: str) -> list[str]:
    """ISO dates that are real calendar dates and not part of a document identifier."""
    from datetime import date

    out = []
    for m in _ISO_DATE.finditer(_IDENTIFIER.sub(" ", question)):
        try:
            date.fromisoformat(m.group(1))
        except ValueError:
            continue
        out.append(m.group(1))
    return out


def _has_named_token(question: str) -> bool:
    """An acronym or an identifier with digits anywhere after the first word."""
    toks = re.findall(r"[A-Za-z0-9][\w\-./]*", question)[1:]
    return any(re.search(r"\d", t) or (len(t) >= 2 and t.isupper()) for t in toks)


def _content_words(question: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z0-9][\w\-./]*", question.lower()) if w not in _STOP]


def route_query(question: str, *, tenant: str = "default", explicit_valid_at: str | None = None,
                has_session: bool = False) -> RouteDecision:
    """Classify a question deterministically; the reason names the rule that fired."""
    from graphrag.graph.controlled_query import plan_controlled_query

    legacy = classify_query(question)
    plan = retrieval_plan(question)
    base = dict(legacy_class=legacy, mode=plan["mode"], top_k=int(plan["top_k"]), fallback=plan["fallback"])
    signals: list[str] = []
    words = _content_words(question)

    if explicit_valid_at:
        return RouteDecision(Route.TEMPORAL, "explicit_valid_at", signals=["valid_at"],
                             as_of=explicit_valid_at, **base)

    structured = None
    try:
        structured = plan_controlled_query(question, tenant=tenant)
    except Exception:  # noqa: BLE001 - unsupported shapes are simply not structured
        structured = None
    if _AGGREGATION.search(question):
        signals.append("aggregation_phrase")
        intent = getattr(structured, "intent", None) if structured else None
        return RouteDecision(Route.AGGREGATION,
                             "aggregation_phrase+template" if intent else "aggregation_phrase_no_template",
                             signals=signals, structured_intent=intent, **base)

    dates = _valid_iso_dates(question)
    # Identifiers and non-calendar date-shaped numbers are not time constraints.
    scrubbed = _ISO_DATE.sub(lambda m: m.group(1) if m.group(1) in dates else " ", _IDENTIFIER.sub(" ", question))
    if _TEMPORAL.search(scrubbed):
        signals.append("temporal_phrase")
        return RouteDecision(Route.TEMPORAL, "temporal_phrase", signals=signals,
                             as_of=dates[0] if dates else None, **base)

    if len(_HOP_CUE.findall(question)) >= 2 or _HOP_WORDS.search(question):
        return RouteDecision(Route.MULTI_HOP, "hop_cues", signals=["hop_cues"], **base)
    if legacy in ("multi_hop",):
        return RouteDecision(Route.MULTI_HOP, f"legacy_{legacy}", signals=[legacy], **base)
    if legacy in ("relational", "contradiction"):
        return RouteDecision(Route.RELATIONAL, f"legacy_{legacy}", signals=[legacy], **base)

    if _RELATION_QUESTION.search(question) or _AUTHORITY_COMPARE.search(question):
        return RouteDecision(Route.RELATIONAL, "relation_verb", signals=["relation_verb"], **base)

    # A single named subject ("What is ICAO?") is a lookup, not an unresolved reference.
    if _ENTITY_LOOKUP.search(question) and len(words) == 1 and not _DEICTIC_ANY.search(question):
        return RouteDecision(Route.ENTITY_LOOKUP, "entity_lookup_phrase", signals=["entity_phrase"], **base)

    residual = [w for w in words if w not in _FILLER]
    if (not has_session and _DEICTIC_ANY.search(question) and len(residual) <= 1
            and not _has_named_token(question)):
        return RouteDecision(Route.AMBIGUOUS, "unresolved_reference", signals=["ambiguous"],
                             **{**base, "fallback": "agentic"})

    if len(words) < 2 or (_DEICTIC_ONLY.search(question) and not has_session):
        return RouteDecision(Route.AMBIGUOUS,
                             "too_few_content_words" if len(words) < 2 else "unresolved_reference",
                             signals=["ambiguous"], **{**base, "fallback": "agentic"})

    if _ENTITY_LOOKUP.search(question) and len(words) <= 4 and not _RELATION_WORDS.search(question):
        return RouteDecision(Route.ENTITY_LOOKUP, "entity_lookup_phrase", signals=["entity_phrase"], **base)

    return RouteDecision(Route.FACTUAL_LOOKUP, f"legacy_{legacy}", signals=[legacy], **base)


def baseline_route(question: str) -> Route:
    """The legacy planner's answer, projected onto the route taxonomy (for comparison)."""
    return LEGACY_TO_ROUTE[classify_query(question)]


def enforced_overrides(decision: RouteDecision, cfg: dict) -> dict:
    """Retrieval config changes a route asks for under ``policy: enforce``."""
    b = ROUTE_BEHAVIOUR[decision.route]
    out: dict = {}
    if not b["graph_expansion"]:
        # No traversal, no GNN over the subgraph, no entity neighbourhood.
        out.update({"multihop_depth": 0, "gnn_enabled": False, "entity_context_enabled": False})
    elif b["multihop_hops"] is not None:
        out["multihop_depth"] = min(int(cfg.get("multihop_depth", 2)), b["multihop_hops"])
    if not b["agentic_fallback"]:
        out["agentic_fallback"] = False
    return out
