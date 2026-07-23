"""Relation-vocabulary analysis — propose a canonicalization map and simulate
its impact, without writing anything to Neo4j.

Problem
-------
The extraction prompt gives the LLM no relation vocabulary (`"relation":
"VERB_RELATION"` in `graphrag/ingestion/extractor.py`), so it invents a name per
sentence. Measured on the live graph: automotive carries 999 distinct relation
names across 4682 edges (598 used exactly once); one entity pair
(PlasiAuto SRL -> AutoCorp GmbH) carries 47 parallel edges for a single
relationship, in two languages.

Because edges MERGE on `{relation: $relation}` (`neo4j_client.merge_relation`),
the relation name is part of edge *identity*. Every spelling is therefore a
separate edge, so the Bayesian noisy-OR confidence merge never accumulates, and
degree / PageRank / hub-dampening / Leiden all see the pair over-weighted.

Why this script writes nothing
------------------------------
Collapsing parallel edges unions their `source_doc_ids`, and the multi_source
contradiction strategy fires on `size(r.source_doc_ids) > 1`. Merging therefore
*manufactures* conflicts: measured at +17 for aerospace and +74 for automotive,
against current totals of 95 and 63. A migration has to be justified by numbers
before it is run, and the proposed map has to be reviewed by a human before it
is applied — collapsing 999 names is a judgement call, not a derivation.

So this script produces a reviewable artifact, and the migration (Phase 2)
consumes an approved one.

Usage
-----
    python scripts/analyze_relation_vocabulary.py --tenant aerospace
    python scripts/analyze_relation_vocabulary.py --tenant automotive --min-edges 2
    python scripts/analyze_relation_vocabulary.py --tenant marketing --no-embeddings

Exit codes
----------
  0  — no canonicalization work found
  1  — failed
  2  — work found (a non-empty map was proposed)   [mirrors community_rebuild.py]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog

from graphrag.graph.alias_registry import _normalize

log = structlog.get_logger("analyze_relation_vocabulary")


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EdgeRow:
    """One RELATES_TO edge, reduced to the fields the simulation needs."""
    src: str
    tgt: str
    relation: str
    confidence: float
    source_doc_ids: tuple[str, ...]


@dataclass
class Proposal:
    """A proposed vocabulary collapse.

    `canonical_map` is safe to apply; `ambiguous` sits in the review band and is
    deliberately *not* applied; `unmapped` had no match at all and is left alone.
    """
    canonical_map: dict[str, str] = field(default_factory=dict)
    ambiguous: list[dict] = field(default_factory=list)
    unmapped: list[str] = field(default_factory=list)
    blocked_inverse: list[dict] = field(default_factory=list)


# ── Inverse-relation guard ────────────────────────────────────────────────────
#
# The single most dangerous merge this tool could propose is a relation with its
# own inverse: SUPPLIES_TO / SUPPLIED_BY normalize to "supplies to" / "supplied
# by" and score ~85 on rapidfuzz, but merging them reverses direction for half
# the edges and manufactures directional_reversal conflicts
# (contradiction_strategies.py:212). Never auto-merge across an inverse pair.

# `_OF` matters as much as `_BY`: the live aerospace vocabulary contains
# IS_VARIANT_OF alongside HAS_VARIANT, which fuzzy+embedding rank as a strong
# match and which are in fact converses of each other.
_CONVERSE_SUFFIX = re.compile(r"(_BY|_FROM|_OF)$")


def _voice(relation: str) -> str:
    """Crude direction marker. Two names in opposite voice are candidate
    inverses even when they are lexically near-identical.

    "converse" = the subject is the patient rather than the agent
    (OPERATED_BY, IS_VARIANT_OF, DERIVED_FROM).
    """
    return "converse" if _CONVERSE_SUFFIX.search(relation.upper()) else "active"


def _stem(token: str) -> str:
    """Strip a trailing verb inflection so SUPERSEDED and SUPERSEDES share a stem."""
    for suf in ("ED", "ES", "S"):
        if token.endswith(suf) and len(token) - len(suf) >= 4:
            return token[: -len(suf)]
    return token


def is_participle_flip(a: str, b: str) -> bool:
    """True for same-verb pairs that differ only in participle vs finite form —
    SUPERSEDED / SUPERSEDES, PREPARED / PREPARES, INTRODUCED / INTRODUCES.

    These read as active in both directions (neither carries a `_BY` or `_OF`
    marker) but the participle form is usually the *patient*: "AD 2020-05-11
    SUPERSEDED" means it was superseded. Merging it into SUPERSEDES reverses the
    supersession chain, which drives both document authority and the
    supersedes_transitivity inference rule.
    """
    ta, tb = a.upper().split("_")[0], b.upper().split("_")[0]
    if ta == tb or _stem(ta) != _stem(tb):
        return False
    return ta.endswith("ED") != tb.endswith("ED")


def is_probable_inverse(a: str, b: str, inverse_pairs: set[frozenset[str]]) -> bool:
    """True when *a* and *b* look like inverses of each other.

    Three signals, in order of authority:
      1. the domain ontology declared them so (`relation_rules: {inverse: ...}`)
      2. they differ in grammatical voice (OPERATES vs OPERATED_BY)
      3. they differ in participle form (SUPERSEDES vs SUPERSEDED)
    """
    au, bu = a.upper(), b.upper()
    if au == bu:
        return False
    if frozenset((au, bu)) in inverse_pairs:
        return True
    if _voice(au) != _voice(bu):
        return True
    return is_participle_flip(au, bu)


def build_inverse_pairs(relation_rules: dict) -> set[frozenset[str]]:
    """Collect declared inverse pairs from a domain ontology's relation_rules.

    e.g. aerospace declares `MANDATED_BY: {inverse: MANDATES}` and
    `CERTIFIES: {inverse: CERTIFIED_BY}`.
    """
    pairs: set[frozenset[str]] = set()
    for name, spec in (relation_rules or {}).items():
        if not isinstance(spec, dict):
            continue
        inv = spec.get("inverse")
        if inv:
            pairs.add(frozenset((str(name).upper(), str(inv).upper())))
    return pairs


# ── Similarity ────────────────────────────────────────────────────────────────

def fuzzy_ratio(a: str, b: str) -> float:
    """rapidfuzz ratio over the *normalized* forms, so MANUFACTURES_AT and
    MANUFACTURES_IN compare as "manufactures at" / "manufactures in".

    Falls back to a stdlib ratio when rapidfuzz is unavailable, so the analysis
    degrades rather than failing (same fail-open posture as AliasRegistry.resolve).
    """
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0
    try:
        from rapidfuzz import fuzz
        return float(fuzz.ratio(na, nb))
    except ImportError:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, na, nb).ratio() * 100.0


def cosine(a, b) -> float:
    """Cosine similarity. Fast path for the unit-normalized numpy vectors
    produced by `unit_vectors()`, where the dot product *is* the cosine —
    the threshold sweep runs this O(n^2) per level over 3072-dim vectors, so
    the pure-Python path is far too slow for a real vocabulary."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return float(np.dot(a, b))
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def unit_vectors(embeddings: dict[str, list[float]]) -> dict[str, "np.ndarray"]:
    """Pre-normalize once so every later comparison is a single dot product."""
    out = {}
    for k, v in embeddings.items():
        arr = np.asarray(v, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        out[k] = arr / norm if norm else arr
    return out


def _best_match(
    name: str,
    targets: list[str],
    embeddings: dict[str, list[float]] | None,
    inverse_pairs: set[frozenset[str]],
) -> tuple[tuple[str, float, float], tuple[str, float, float]]:
    """Best (target, fuzzy, cosine) for *name*, as (allowed, blocked_as_inverse).

    The blocked candidate is returned rather than discarded so the report can
    show what the inverse guard withheld and why — a silent guard is impossible
    to review or to tune.
    """
    best = ("", 0.0, 0.0)
    blocked = ("", 0.0, 0.0)
    emb_name = (embeddings or {}).get(name)
    for t in targets:
        if t == name:
            continue
        f = fuzzy_ratio(name, t)
        c = 0.0
        if emb_name is not None and embeddings and t in embeddings:
            c = cosine(emb_name, embeddings[t])
        strength = max(f / 100.0, c)
        if is_probable_inverse(name, t, inverse_pairs):
            if strength > max(blocked[1] / 100.0, blocked[2]):
                blocked = (t, f, c)
            continue
        # rank on whichever signal is stronger relative to its own scale
        if strength > max(best[1] / 100.0, best[2]):
            best = (t, f, c)
    return best, blocked


# ── Proposal ──────────────────────────────────────────────────────────────────

def _record_blocked(
    proposal: Proposal,
    name: str,
    blocked: tuple[str, float, float],
    counts: dict[str, int],
    fuzzy_threshold: float,
    embedding_threshold: float,
) -> None:
    """Note a merge the inverse guard withheld — but only one that would
    otherwise have been applied, so the list stays short and actionable."""
    target, f, c = blocked
    if not target or (f < fuzzy_threshold and c < embedding_threshold):
        return
    proposal.blocked_inverse.append({
        "relation": name, "candidate": target,
        "fuzzy": round(f, 1), "cosine": round(c, 4),
        "edges": counts.get(name, 0),
        "reason": "participle_flip" if is_participle_flip(name, target)
                  else "opposite_voice_or_declared_inverse",
    })


def propose_canonical_map(
    counts: dict[str, int],
    canonical_terms: set[str],
    embeddings: dict[str, list[float]] | None = None,
    *,
    inverse_pairs: set[frozenset[str]] | None = None,
    fuzzy_threshold: float = 85.0,
    embedding_threshold: float = 0.92,
    review_fuzzy_min: float = 70.0,
    review_embedding_min: float = 0.85,
) -> Proposal:
    """Propose `observed_name -> canonical_name` collapses.

    Mirrors the cascade order of `AliasRegistry.resolve()` (exact -> normalized
    -> fuzzy -> embedding) and reuses its configured thresholds rather than
    inventing new ones. Scores inside the review band are reported as ambiguous
    and never auto-applied.

    Two passes:
      1. observed name -> a term the ontology actually declares
      2. leftovers clustered against each other, the highest-edge-count member
         of each cluster becoming its canonical form
    """
    inverse_pairs = inverse_pairs or set()
    proposal = Proposal()

    canon_list = sorted(canonical_terms)
    canon_by_norm = {_normalize(t): t for t in canon_list}

    # Highest edge count first, so frequent names win as cluster heads.
    ordered = sorted(counts, key=lambda n: (-counts[n], n))

    leftovers: list[str] = []

    # ── Pass 1: map onto the declared vocabulary ──────────────────────────────
    for name in ordered:
        if name in canonical_terms:
            continue                                    # already canonical

        norm = _normalize(name)
        hit = canon_by_norm.get(norm)
        if hit and not is_probable_inverse(name, hit, inverse_pairs):
            proposal.canonical_map[name] = hit
            continue

        (target, f, c), blocked = _best_match(name, canon_list, embeddings, inverse_pairs)
        _record_blocked(proposal, name, blocked, counts,
                        fuzzy_threshold, embedding_threshold)
        if not target:
            leftovers.append(name)
            continue

        if f >= fuzzy_threshold:
            proposal.canonical_map[name] = target
        elif c >= embedding_threshold:
            proposal.canonical_map[name] = target
        elif f >= review_fuzzy_min or c >= review_embedding_min:
            proposal.ambiguous.append({
                "relation": name, "candidate": target,
                "fuzzy": round(f, 1), "cosine": round(c, 4),
                "edges": counts[name], "band": "canonical_vocabulary",
            })
        else:
            leftovers.append(name)

    # ── Pass 2: cluster the leftovers against each other ──────────────────────
    heads: list[str] = []
    for name in leftovers:
        (target, f, c), blocked = _best_match(name, heads, embeddings, inverse_pairs)
        _record_blocked(proposal, name, blocked, counts,
                        fuzzy_threshold, embedding_threshold)
        if target and (f >= fuzzy_threshold or c >= embedding_threshold):
            proposal.canonical_map[name] = target
        elif target and (f >= review_fuzzy_min or c >= review_embedding_min):
            proposal.ambiguous.append({
                "relation": name, "candidate": target,
                "fuzzy": round(f, 1), "cosine": round(c, 4),
                "edges": counts[name], "band": "observed_cluster",
            })
            heads.append(name)
        else:
            heads.append(name)
            proposal.unmapped.append(name)

    return proposal


# ── Threshold calibration ─────────────────────────────────────────────────────
#
# The alias thresholds do NOT transfer to relation names. `alias_embedding_threshold`
# (0.92) was tuned for entity name + description strings, which are long and
# near-duplicated. Relation names are 1-3 tokens, so cosine sits on a different
# scale entirely — measured on the live aerospace vocabulary:
#
#     MANUFACTURES_AT / MANUFACTURES_IN   0.856   (true merge)
#     LIVREAZA_CATRE  / DELIVERS_TO       0.620   (true merge, cross-language)
#     unrelated pairs                     0.25-0.45
#
# At 0.92 nothing merges and the embedding pass is dead weight. Rather than pick
# a replacement constant by feel — this repo's history has several reverts from
# exactly that (see the alias_embedding_threshold comment in settings.yml) —
# report the sweep and let the operating point be chosen against evidence.

CALIBRATION_LEVELS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)


def calibrate(
    counts: dict[str, int],
    canonical_terms: set[str],
    embeddings: dict[str, list[float]] | None,
    inverse_pairs: set[frozenset[str]],
    *,
    levels: tuple[float, ...] = CALIBRATION_LEVELS,
    fuzzy_threshold: float = 85.0,
    sample: int = 4,
) -> list[dict]:
    """How many merges each embedding threshold would produce, with samples.

    This is the artifact a human needs in order to choose a threshold: the point
    where merge count starts climbing steeply is where false positives begin.
    """
    if not embeddings:
        return []
    out = []
    for level in levels:
        proposal = propose_canonical_map(
            counts, canonical_terms, embeddings,
            inverse_pairs=inverse_pairs,
            fuzzy_threshold=fuzzy_threshold,
            embedding_threshold=level,
            review_fuzzy_min=101.0,      # suppress the review band during a sweep
            review_embedding_min=1.01,
        )
        pairs = sorted(proposal.canonical_map.items(),
                       key=lambda kv: -counts.get(kv[0], 0))
        out.append({
            "embedding_threshold": level,
            "merges": len(proposal.canonical_map),
            "examples": [f"{k} -> {v}" for k, v in pairs[:sample]],
        })
    return out


# ── Simulation ────────────────────────────────────────────────────────────────

def noisy_or(confidences: list[float]) -> float:
    """The same independent-evidence fusion the write path applies on re-observation
    (`neo4j_client.merge_relations_batch`): 1 - PROD(1 - c)."""
    acc = 0.0
    for c in confidences:
        acc = 1.0 - (1.0 - acc) * (1.0 - c)
    return acc


def simulate(
    edges: list[EdgeRow],
    canonical_map: dict[str, str],
    *,
    confidence_threshold: float = 0.7,
    degree_multiplier: float = 20.0,
    open_conflicts: int = 0,
) -> dict:
    """Compute what applying *canonical_map* would do, without applying it.

    Everything here is derived from the edge rows in Python rather than in
    Cypher, so the arithmetic is unit-testable and needs no APOC.
    """
    def canon(rel: str) -> str:
        return canonical_map.get(rel, rel)

    # ── Group edges by their post-merge identity ──────────────────────────────
    groups: dict[tuple[str, str, str], list[EdgeRow]] = defaultdict(list)
    for e in edges:
        groups[(e.src, e.tgt, canon(e.relation))].append(e)

    edges_before = len(edges)
    edges_after = len(groups)
    relations_before = len({e.relation for e in edges})
    relations_after = len({canon(e.relation) for e in edges})

    # ── multi_source conflicts manufactured by the merge ──────────────────────
    # Strategy 1 fires on size(r.source_doc_ids) > 1. A group gains one only if
    # it spans >1 distinct document AND no member already did.
    new_multi_source = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        already = any(len(set(m.source_doc_ids)) > 1 for m in members)
        if already:
            continue
        docs = {d for m in members for d in m.source_doc_ids}
        if len(docs) > 1:
            new_multi_source += 1

    # ── Confidence: how many edges cross the GNN threshold after fusion ───────
    crossed_up = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        before_max = max(m.confidence for m in members)
        after = noisy_or([m.confidence for m in members])
        if before_max < confidence_threshold <= after:
            crossed_up += 1

    high_conf_before = sum(1 for e in edges if e.confidence >= confidence_threshold)
    noise_before = sum(1 for e in edges if e.confidence < 0.3)
    after_confidences = [noisy_or([m.confidence for m in ms]) for ms in groups.values()]
    high_conf_after = sum(1 for c in after_confidences if c >= confidence_threshold)
    noise_after = sum(1 for c in after_confidences if c < 0.3)

    # ── Degree: undirected incident-edge count, matching the validator ────────
    def degrees(pairs) -> dict[str, int]:
        d: dict[str, int] = defaultdict(int)
        for src, tgt in pairs:
            d[src] += 1
            d[tgt] += 1
        return d

    deg_before = degrees((e.src, e.tgt) for e in edges)
    deg_after = degrees((s, t) for (s, t, _) in groups)

    def flagged(d: dict[str, int]) -> set[str]:
        if not d:
            return set()
        mean = sum(d.values()) / len(d)
        return {n for n, v in d.items() if v > mean * degree_multiplier}

    flag_before, flag_after = flagged(deg_before), flagged(deg_after)

    return {
        "relations_before": relations_before,
        "relations_after": relations_after,
        "edges_before": edges_before,
        "edges_after": edges_after,
        "edges_collapsed": edges_before - edges_after,
        "new_multi_source_conflicts": new_multi_source,
        "open_conflicts_before": open_conflicts,
        "open_conflicts_after_est": open_conflicts + new_multi_source,
        "conflicts_per_1k_edges_before": round(open_conflicts / max(edges_before, 1) * 1000, 2),
        "conflicts_per_1k_edges_after": round(
            (open_conflicts + new_multi_source) / max(edges_after, 1) * 1000, 2),
        "edges_crossing_confidence_threshold": crossed_up,
        "high_conf_rate_before": round(high_conf_before / max(edges_before, 1), 4),
        "high_conf_rate_after": round(high_conf_after / max(edges_after, 1), 4),
        "noise_edge_rate_before": round(noise_before / max(edges_before, 1), 4),
        "noise_edge_rate_after": round(noise_after / max(edges_after, 1), 4),
        "degree_quarantine_flagged_before": sorted(flag_before),
        "degree_quarantine_flagged_after": sorted(flag_after),
        "degree_quarantine_newly_flagged": sorted(flag_after - flag_before),
        "degree_quarantine_no_longer_flagged": sorted(flag_before - flag_after),
    }


# ── Neo4j IO (read-only) ──────────────────────────────────────────────────────

async def fetch_edges(neo4j, tenant: str) -> list[EdgeRow]:
    rows = await neo4j.run(
        """
        MATCH (a:Entity {tenant: $tenant})-[r:RELATES_TO]->(b:Entity {tenant: $tenant})
        RETURN a.name AS src, b.name AS tgt, r.relation AS relation,
               coalesce(r.confidence, 1.0) AS confidence,
               coalesce(r.source_doc_ids, []) AS source_doc_ids
        """,
        tenant=tenant,
    )
    return [
        EdgeRow(
            src=r["src"], tgt=r["tgt"], relation=r["relation"],
            confidence=float(r["confidence"]),
            source_doc_ids=tuple(r["source_doc_ids"] or ()),
        )
        for r in rows if r.get("relation")
    ]


async def fetch_open_conflicts(neo4j, tenant: str) -> int:
    rows = await neo4j.run(
        "MATCH (c:Conflict {tenant: $tenant, status: 'open'}) RETURN count(c) AS n",
        tenant=tenant,
    )
    return int(rows[0]["n"]) if rows else 0


def load_canonical_vocabulary(tenant: str) -> tuple[set[str], set[frozenset[str]]]:
    """The vocabulary the tenant's ontology actually declares, plus the built-ins.

    Returns (canonical_terms, declared_inverse_pairs).
    """
    from graphrag.graph.ontology_registry import _RELATION_RULES

    terms = {t.upper() for t in _RELATION_RULES}
    inverse_pairs: set[frozenset[str]] = set()

    try:
        from graphrag.graph.domain_ontology import (
            get_ontology_path_for_tenant,
            get_relation_rules,
            load_domain_ontology,
        )
        path = get_ontology_path_for_tenant(tenant)
        if path:
            rules = get_relation_rules(load_domain_ontology(path))
            terms |= {r.upper() for r in rules}
            inverse_pairs = build_inverse_pairs(rules)
            for spec in rules.values():
                if isinstance(spec, dict) and spec.get("inverse"):
                    terms.add(str(spec["inverse"]).upper())
    except Exception as exc:      # fail open — built-ins still give a usable run
        log.warning("relation_vocabulary.ontology_load_failed",
                    tenant=tenant, error=str(exc))

    terms.discard("RELATED_TO")   # the generic fallback is not a merge target
    return terms, inverse_pairs


async def embed_relation_names(names: list[str]) -> dict[str, list[float]]:
    """Embed the relation names themselves.

    Note two dead ends ruled out during design: `AliasRegistry`'s embedding
    helpers are bound to the `entity_embeddings` vector index and require an
    entity type, and `edge_embeddings._derive_relation_embedding()` is a
    SHA-256-seeded vector that is deliberately non-semantic. Neither can compare
    relation names, so embed the strings directly. The underscore-stripped form
    is embedded because the model reads "delivers to" better than "DELIVERS_TO";
    this is also what lets the multilingual model match LIVREAZA_CATRE.
    """
    from graphrag.ingestion.embedder import Embedder

    texts = [_normalize(n) or n.lower() for n in names]
    vectors = await Embedder().embed_texts(texts)
    return unit_vectors(dict(zip(names, vectors)))


# ── Orchestration ─────────────────────────────────────────────────────────────

async def analyze(
    tenant: str,
    min_edges: int,
    use_embeddings: bool,
    embedding_threshold: float | None = None,
    fuzzy_threshold: float | None = None,
) -> dict:
    from graphrag.core.config import get_settings
    from graphrag.graph.neo4j_client import get_neo4j

    neo4j = get_neo4j()
    ing = get_settings().ingestion

    edges = await fetch_edges(neo4j, tenant)
    if not edges:
        return {"tenant": tenant, "error": "no edges found"}

    counts: dict[str, int] = defaultdict(int)
    for e in edges:
        counts[e.relation] += 1
    considered = {n: c for n, c in counts.items() if c >= min_edges}

    canonical_terms, inverse_pairs = load_canonical_vocabulary(tenant)

    embeddings = None
    if use_embeddings:
        try:
            embeddings = await embed_relation_names(
                sorted(set(considered) | canonical_terms)
            )
        except Exception as exc:
            log.warning("relation_vocabulary.embedding_failed", error=str(exc))

    fuzzy_t = fuzzy_threshold if fuzzy_threshold is not None \
        else float(ing.get("alias_fuzzy_threshold", 85))
    embed_t = embedding_threshold if embedding_threshold is not None \
        else float(ing.get("alias_embedding_threshold", 0.92))

    proposal = propose_canonical_map(
        considered, canonical_terms, embeddings,
        inverse_pairs=inverse_pairs,
        fuzzy_threshold=fuzzy_t,
        embedding_threshold=embed_t,
        review_fuzzy_min=float(ing.get("review_fuzzy_min", 70)),
        review_embedding_min=float(ing.get("review_embedding_min", 0.85)),
    )

    open_conflicts = await fetch_open_conflicts(neo4j, tenant)
    sim = simulate(edges, proposal.canonical_map, open_conflicts=open_conflicts)

    return {
        "tenant": tenant,
        "min_edges": min_edges,
        "embeddings_used": embeddings is not None,
        "fuzzy_threshold": fuzzy_t,
        "embedding_threshold": embed_t,
        "canonical_vocabulary_size": len(canonical_terms),
        "calibration": calibrate(considered, canonical_terms, embeddings,
                                 inverse_pairs, fuzzy_threshold=fuzzy_t),
        "canonical_map": dict(sorted(proposal.canonical_map.items())),
        "ambiguous": sorted(proposal.ambiguous, key=lambda a: -a["edges"]),
        "blocked_inverse": sorted(proposal.blocked_inverse, key=lambda a: -a["edges"]),
        "unmapped": sorted(proposal.unmapped),
        "simulation": sim,
    }


def print_summary(report: dict) -> None:
    s = report.get("simulation", {})
    t = report["tenant"]
    print(f"\n=== relation vocabulary: {t} ===")
    cal = report.get("calibration") or []
    if cal:
        print(f"  embedding-threshold sweep (current: {report.get('embedding_threshold')}):")
        for row in cal:
            ex = row["examples"][0] if row["examples"] else ""
            print(f"      {row['embedding_threshold']:.2f} -> {row['merges']:4d} merges   {ex}")
    print(f"  proposed merges     : {len(report['canonical_map'])}")
    print(f"  ambiguous (review)  : {len(report['ambiguous'])}")
    blocked = report.get("blocked_inverse") or []
    print(f"  blocked as inverse  : {len(blocked)}")
    for b in blocked[:5]:
        print(f"      {b['relation']} !-> {b['candidate']}  ({b['reason']})")
    print(f"  unmapped (untouched): {len(report['unmapped'])}")
    print(f"  distinct relations  : {s.get('relations_before')} -> {s.get('relations_after')}")
    print(f"  edges               : {s.get('edges_before')} -> {s.get('edges_after')}"
          f"  ({s.get('edges_collapsed')} collapsed)")
    print(f"  NEW multi_source conflicts : +{s.get('new_multi_source_conflicts')}"
          f"  (open: {s.get('open_conflicts_before')} -> ~{s.get('open_conflicts_after_est')})")
    print(f"  conflicts / 1k edges: {s.get('conflicts_per_1k_edges_before')}"
          f" -> {s.get('conflicts_per_1k_edges_after')}")
    print(f"  edges crossing conf 0.7    : +{s.get('edges_crossing_confidence_threshold')}")
    print(f"  high-conf rate      : {s.get('high_conf_rate_before')} -> {s.get('high_conf_rate_after')}")
    print(f"  noise-edge rate     : {s.get('noise_edge_rate_before')} -> {s.get('noise_edge_rate_after')}")
    newly = s.get("degree_quarantine_newly_flagged") or []
    gone = s.get("degree_quarantine_no_longer_flagged") or []
    print(f"  degree-quarantine   : +{len(newly)} newly flagged, -{len(gone)} released")
    if newly:
        print(f"      newly flagged: {newly[:5]}")
    if gone:
        print(f"      released     : {gone[:5]}")


async def main(args) -> int:
    tenants = [args.tenant] if args.tenant else ["aerospace", "automotive", "marketing"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    for tenant in tenants:
        try:
            report = await analyze(
                tenant, args.min_edges, not args.no_embeddings,
                embedding_threshold=args.embedding_threshold,
                fuzzy_threshold=args.fuzzy_threshold,
            )
        except Exception as exc:
            log.error("relation_vocabulary.failed", tenant=tenant, error=str(exc))
            print(f"[{tenant}] FAILED: {exc}")
            exit_code = 1
            continue

        if report.get("error"):
            print(f"[{tenant}] {report['error']}")
            continue

        path = out_dir / f"relation_canonicalization_{tenant}.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print_summary(report)
        print(f"  written: {path}")

        if report["canonical_map"] and exit_code == 0:
            exit_code = 2
    return exit_code


if __name__ == "__main__":
    import io
    import sys

    # Corpora contain Romanian diacritics; a cp1252 stdout raises
    # UnicodeEncodeError and kills the run (see tasks/lessons.md — the same
    # failure crashed the ingestion workers). Wrap only under __main__ so
    # pytest's capture is left alone.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Propose and simulate a relation-vocabulary canonicalization "
                    "(read-only — writes nothing to Neo4j)."
    )
    parser.add_argument("--tenant", default="",
                        help="Tenant to analyze (default: aerospace, automotive, marketing)")
    parser.add_argument("--out-dir", default="exports",
                        help="Directory for the JSON report (default: exports)")
    parser.add_argument("--min-edges", type=int, default=1,
                        help="Ignore relation names with fewer than N edges (default: 1)")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Skip the embedding pass (fuzzy + normalized only)")
    parser.add_argument("--embedding-threshold", type=float, default=None,
                        help="Cosine threshold for a merge. Defaults to "
                             "ingestion.alias_embedding_threshold (0.92), which is "
                             "tuned for entity names and is far too strict for "
                             "relation names — see the calibration sweep in the output.")
    parser.add_argument("--fuzzy-threshold", type=float, default=None,
                        help="rapidfuzz ratio for a merge (default: "
                             "ingestion.alias_fuzzy_threshold, 85)")
    sys.exit(asyncio.run(main(parser.parse_args())))
