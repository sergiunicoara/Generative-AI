"""Alias registry — resolves alternative entity names to canonical ones.

Problem solved
--------------
The same entity appears under different names across documents:
  "SpaceX" / "Space Exploration Technologies" / "Space Exploration Corp."
Without resolution, each variant creates a separate node, breaking
retrieval, GNN propagation, and community detection.

Strategy
--------
1. Exact match on stored aliases (instant)
2. Normalized match — strip punctuation, lowercase, collapse whitespace
3. Embedding similarity against known entity embeddings (cosine > 0.97 —
   see `ingestion.alias_embedding_threshold` in config/settings.yml)
4. Queue ambiguous cases for human review rather than auto-creating

The registry is loaded once per process and refreshed after every
ingestion batch.
"""

from __future__ import annotations

import os
import re
import structlog
from dataclasses import dataclass
from typing import NamedTuple

log = structlog.get_logger(__name__)


@dataclass
class AmbiguousMatch:
    """Returned by resolve() when a name is close but below the auto-merge threshold.

    The caller decides whether to enqueue for human review or treat as a new entity.
    """
    candidate: tuple[str, str]   # (canonical_name, canonical_type)
    score: float                 # rapidfuzz ratio (fuzzy) or cosine similarity (embedding)
    match_type: str              # "fuzzy" | "embedding"
    # Other candidates above the review threshold that scored below ``candidate``:
    # (name, type, score) tuples, kept so a reviewer can see what else it nearly matched.
    runner_ups: tuple = ()


class ResolvedMatch(NamedTuple):
    """resolve(..., with_detail=True)'s return for an auto-resolved match.

    Only used by the one caller that opts into ``with_detail=True``
    (graph_writer.py's entity-write path) — every other caller keeps
    resolve()'s default, unchanged ``(name, type)`` tuple return, which
    14+ existing tests and two other call sites still assert against.
    ``canonical[0]``/``canonical[1]`` index the same as a plain tuple, but
    ``*canonical`` unpacks all 4 fields, not 2 — any caller that opts into
    ``with_detail`` must index explicitly, not star-unpack.
    """
    name: str
    type: str
    method: str                  # "exact" | "fuzzy"
    score: float | None = None   # rapidfuzz ratio for "fuzzy"; None for "exact"
    runner_ups: tuple[tuple[str, str, float], ...] = ()
    # (name, type, score) for candidates that scored above the review
    # threshold but lost to the winner above — persisted for later audit,
    # never used to change which candidate wins. Always () for "exact"
    # matches (nothing to have lost to) and for every caller not using
    # with_detail=True.


class EmbeddingMatch(NamedTuple):
    """find_duplicate_by_embedding(..., with_detail=True)'s return for an
    auto-resolved embedding match — same non-breaking-addition shape as
    ResolvedMatch. Default (with_detail=False) callers keep the original
    ``(name, type, score) | None`` return untouched.
    """
    name: str
    type: str
    score: float
    runner_ups: tuple[tuple[str, str, float], ...] = ()


# Defaults — overridden by ingestion.alias_embedding_threshold /
# ingestion.alias_fuzzy_threshold in config/settings.yml.
_DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD = 0.97   # cosine similarity for dedup
_DEFAULT_FUZZY_SCORE_THRESHOLD = 85              # rapidfuzz ratio for soft match

# ANN candidate pool for the two embedding-search methods below, before the
# tenant/type/threshold WHERE filter. entity_embeddings is a shared index
# across all tenants, so a small k (previously hardcoded 5/10) can starve
# out a tenant's own true duplicate if other tenants score higher for the
# same query vector — silent false negative on dedup. 100 is generous
# against current entity counts (~3.4k total across tenants) and measured
# latency-neutral vs. the old k=5 (see tasks/lessons.md A148).
_FETCH_K = 100
_REDIS_TTL = 86400                       # 24h — alias table refreshed each ingestion batch

# Redis key pattern:  graphrag:aliases:{tenant}   → Hash { normalized_alias: "canonical|type" }
_REDIS_KEY = "graphrag:aliases:{tenant}"


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Regulatory agency prefixes that are stripped when normalizing AD / regulation names.
# "EASA AD 2022-0201" and "AD 2022-0201" refer to the same document —
# the prefix is a source attribution, not a distinguishing identifier.
#
# [\s-]+ (not \s+): real corpora hyphenate the prefix onto the identifier
# ("FAA-AD-2022-03-07") at least as often as they space-separate it
# ("FAA AD 2022-03-07") — a space-only pattern silently never fires for the
# hyphenated form, leaving it and its space-separated twin as two permanently
# unmerged entity nodes. That gap broke forward-chaining transitivity across
# a whole supersession chain (INF-01 in evals/golden_set.json) even though
# every other piece of the transitivity/citation pipeline worked correctly.
_REGULATORY_PREFIXES = re.compile(
    r"^(easa|faa|icao|dot|tc|stc|pma|tso)[\s-]+",
    re.IGNORECASE,
)


def _normalize_regulatory(text: str) -> str:
    """Like _normalize but also strips known regulatory agency prefixes.

    Applied to AIRWORTHINESS_DIRECTIVE and REGULATION entity names so that
    'EASA AD 2022-0201' and 'AD 2022-0201' resolve to the same canonical key,
    enabling forward-chaining transitivity to fire across the supersession chain.
    """
    stripped = _REGULATORY_PREFIXES.sub("", text.strip())
    return _normalize(stripped)


def canonical_document_key(name: str) -> str:
    """Stable key for deciding whether two strings name the same document.

    Public wrapper over the same normalization the alias registry uses
    internally, exposed so citation-registration and evaluation code can ask
    "do these two names refer to one document?" without reaching for a
    private helper or re-implementing the rules and drifting from them.

    Motivating case (2026-08-17, see docs/audit-2026-08-13.md): citations
    reach the caller under two different naming systems — document-derived
    ones use the source filename stem (``FAA-AD-2024-01-02``) while
    entity-derived ones use the surface form the corpus text actually
    writes (``AD 2024-01-02``). Both denote one document, and this function
    is what lets the two be matched up:

        canonical_document_key("FAA-AD-2024-01-02") == "ad 2024 01 02"
        canonical_document_key("AD 2024-01-02")     == "ad 2024 01 02"

    Non-regulatory names are unaffected by prefix stripping and simply
    normalize on punctuation/case, so this is safe to apply to any citation
    string: ``SWA_fleet_registry_2024`` -> ``swa fleet registry 2024``.
    """
    return _normalize_regulatory(name)


# Conservative Romanian noun-suffix list (longest first) for merging
# inflectional variants — e.g. "furnizor" / "furnizori" / "furnizorul" /
# "furnizorii" / "furnizorilor" all stem to "furnizor".
_RO_NOUN_SUFFIXES = ("ilor", "ului", "urile", "elor", "iilor",
                     "uri", "ii", "ul", "lor", "le", "i", "a", "e")
_RO_MIN_STEM_LEN = 3


def _stem_ro_token(token: str) -> str:
    """Strip a single Romanian noun inflection suffix, keeping at least
    `_RO_MIN_STEM_LEN` characters of the stem to avoid over-stemming short words."""
    for suf in _RO_NOUN_SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= _RO_MIN_STEM_LEN:
            return token[: -len(suf)]
    return token


def _normalize_ro(text: str) -> str:
    """Like _normalize, but additionally stems each token's Romanian noun
    inflection — used as a fallback dedup key so "furnizor" / "furnizori" /
    "furnizorul" / "furnizorii" / "furnizorilor" resolve to the same entity.
    """
    return " ".join(_stem_ro_token(t) for t in _normalize(text).split(" ") if t)


# ── Lexical guards for the fuzzy tier ────────────────────────────────────────
# A character ratio cannot tell "Boeing 737-800" from "Boeing 747-800" or
# "Halvorsen Grid Services" from "Halvorsen Rail Services": one token differs
# and the ratio stays above the auto-merge threshold. These guards compare
# tokens instead. See docs/entity-resolution.md ("Lexical guards").

# Words that carry no identity: connectives and legal-form suffixes.
_IGNORED_TOKENS = frozenset({
    "and", "of", "the", "der", "die", "das", "de", "la", "le", "und", "et", "for",
    "inc", "incorporated", "ltd", "limited", "llc", "gmbh", "ag", "sa", "srl",
    "plc", "corp", "corporation", "co", "company", "bv", "nv", "kg", "se",
})


def _content_tokens(key: str) -> list[str]:
    return [t for t in key.split(" ") if t and t not in _IGNORED_TOKENS]


def _is_identifier(token: str) -> bool:
    return any(ch.isdigit() for ch in token)


def _is_abbreviation(short: str, long: str) -> bool:
    """`assy` for `assembly`, `air` for `airlines`: same first letter and the
    short token's letters appear in order in the long one."""
    if len(short) < 2 or len(short) >= len(long) or short[0] != long[0]:
        return False
    rest = iter(long[1:])
    return all(ch in rest for ch in short[1:])


def _tokens_compatible(a: str, b: str) -> bool:
    """Could `a` and `b` be the same word written differently?

    Identifier tokens (any digit) must be equal: 450 and 470 are different
    parts however close the characters are. Words may differ by a typo, a
    plural or an abbreviation.
    """
    if a == b:
        return True
    if _is_identifier(a) or _is_identifier(b):
        return False
    from rapidfuzz import fuzz
    from rapidfuzz.distance import DamerauLevenshtein
    if fuzz.ratio(a, b) >= 80:
        return True
    # One adjacent transposition ("siet" for "site") scores only 75; a
    # one-letter substitution ("rail"/"mail") stays a different word.
    if sorted(a) == sorted(b) and DamerauLevenshtein.distance(a, b) == 1:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return _is_abbreviation(short, long)


def token_conflict(raw_key: str, candidate_key: str) -> str | None:
    """Return why two normalized names cannot be the same entity, or None.

    A conflict is a substitution: each side has a content token that matches
    nothing on the other side ("marine" against "air", "450" against "470").
    One side merely having extra tokens ("Pratt & Whitney Inc.", "Boeing 737"
    against "Boeing 737-800") is not a conflict; see `token_extra_identifier`.
    """
    raw_tokens, candidate_tokens = _content_tokens(raw_key), _content_tokens(candidate_key)
    raw_left = [t for t in raw_tokens if not any(_tokens_compatible(t, c) for c in candidate_tokens)]
    candidate_left = [c for c in candidate_tokens if not any(_tokens_compatible(c, t) for t in raw_tokens)]
    if not raw_left or not candidate_left:
        return None
    if any(_is_identifier(t) for t in raw_left + candidate_left):
        return "identifier_mismatch"
    return "token_substitution"


def token_extra_identifier(raw_key: str, candidate_key: str) -> bool:
    """True when one side has an identifier token the other lacks: "Boeing
    737" against "Boeing 737-800" is under-specified, never an auto-merge."""
    raw_ids = {t for t in _content_tokens(raw_key) if _is_identifier(t)}
    candidate_ids = {t for t in _content_tokens(candidate_key) if _is_identifier(t)}
    return raw_ids != candidate_ids


def _initials(key: str) -> str:
    return "".join(t[0] for t in _content_tokens(key) if not _is_identifier(t))


def _looks_like_acronym(raw_name: str) -> bool:
    stripped = raw_name.strip().replace(".", "")
    return 2 <= len(stripped) <= 6 and stripped.isalpha() and stripped.isupper()


def _same_type(entity_type: str | None, candidate_type: str | None) -> bool:
    if not entity_type or not candidate_type:
        return True
    return entity_type.strip().upper() == candidate_type.strip().upper()


async def _get_redis():
    """Return an async Redis client, or None when the optional cache is off.

    Alias resolution is correct without Redis: Neo4j remains the source of
    truth and Redis only shares a warmed alias table between workers.  Do not
    probe localhost merely because redis-py is installed; an explicit
    ``REDIS_URL`` opts a deployment into this cache.  A configured-but-down
    cache receives one short connection attempt, then safely falls back.
    """
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        log.debug("alias_registry.redis_not_configured")
        return None

    try:
        import redis.asyncio as aioredis
        from redis.backoff import NoBackoff
        from redis.retry import Retry
    except ImportError:
        log.debug("alias_registry.redis_not_installed")
        return None

    client = None
    try:
        client = aioredis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
            retry=Retry(NoBackoff(), 0),
            retry_on_timeout=False,
        )
        await client.ping()
        return client
    except Exception as exc:  # noqa: BLE001 — redis-py raises a wide family here
        if client is not None:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001 — fallback must stay available
                pass
        log.warning("alias_registry.redis_unavailable",
                    error=str(exc)[:160], hint="falling back to Neo4j alias lookups")
        return None


class AliasRegistry:
    """
    In-memory alias registry backed by Neo4j.

    Usage::

        registry = AliasRegistry(neo4j_client)
        await registry.load()
        canonical = registry.resolve("Space Exploration Technologies")
        # → ("SpaceX", "ORG") or None if unknown
    """

    def __init__(self, neo4j_client, tenant: str = "default"):
        self._neo4j = neo4j_client
        self._tenant = tenant
        # normalized_alias → (canonical_name, canonical_type)
        # Keyed per-tenant so the registry is safe for single-tenant usage
        # (multi-tenant deployments should use one registry instance per tenant)
        self._exact: dict[str, tuple[str, str]] = {}
        # Fuzzy matching remains an exact rapidfuzz comparison, but candidates
        # are indexed by normalized length.  A name outside the mathematically
        # possible length range for the review threshold cannot affect either
        # auto-merge or the ambiguous-review band, so it is safe to skip.
        self._fuzzy_by_length: dict[int, list[str]] = {}
        self._fuzzy_index_size = 0
        # Romanian-noun-stem fallback table — see _normalize_ro(). First-wins
        # (setdefault) so the earliest-seen canonical for a given stem wins.
        self._stemmed: dict[str, tuple[str, str]] = {}
        self._loaded = False

        from graphrag.core.config import get_settings
        ingestion_cfg = get_settings().ingestion
        self._embedding_threshold = ingestion_cfg.get(
            "alias_embedding_threshold", _DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD
        )
        self._fuzzy_threshold = ingestion_cfg.get(
            "alias_fuzzy_threshold", _DEFAULT_FUZZY_SCORE_THRESHOLD
        )
        # Embedding similarity alone auto-merged two entities of the same
        # tenant+type with zero corroborating signal — two lexically
        # unrelated names in the same domain can score above the embedding
        # threshold purely from shared domain vocabulary. This floor (a
        # rapidfuzz ratio, same scale as alias_fuzzy_threshold) requires the
        # candidate name to also be lexically plausible before auto-merge;
        # candidates that clear the embedding threshold but fail this floor
        # fall through to the existing ambiguous-band review queue instead
        # of merging blind. See docs/IMPLEMENTATION_AUDIT.md ("fail open").
        self._embedding_min_lexical = ingestion_cfg.get(
            "alias_embedding_min_lexical_similarity", 30
        )
        # Contextual disambiguation tie-break — see find_duplicate_by_embedding
        # and _cooccurrence_tiebreak. Disabled by default pending a
        # golden-eval validation pass, same precedent as include_superseded
        # (tasks/lessons.md A128): a similarly-shaped naive change once
        # caused an unexplained regression, so this ships opt-in.
        self._cooccurrence_tiebreak_enabled = ingestion_cfg.get(
            "alias_cooccurrence_tiebreak_enabled", False
        )
        self._embedding_tie_margin = ingestion_cfg.get(
            "alias_embedding_tie_margin", 0.01
        )
        # Token guards, type scoping, acronym and containment review on the
        # fuzzy tier (see token_conflict). On by default; false restores the
        # character-ratio-only behaviour without a deploy.
        self._lexical_guards = ingestion_cfg.get("alias_lexical_guards", True)
        # Curated aliases ([{alias, canonical, type, tenant?}]) applied in
        # memory after every load, for names nothing in the text links,
        # chiefly acronyms. Never creates an entity.
        self._curated_alias_files = list(ingestion_cfg.get("curated_alias_files") or [])
        # head token -> keys, initials -> keys; built with the fuzzy index.
        self._by_head: dict[str, list[str]] = {}
        self._by_initials: dict[str, list[str]] = {}

    async def load(self) -> None:
        """Refresh alias table from Neo4j and push to Redis for cross-worker sharing."""
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity {tenant: $tenant})
            OPTIONAL MATCH (e)<-[:ALIAS_OF]-(a:Alias {tenant: $tenant})
            RETURN e.name AS canonical_name,
                   e.type AS canonical_type,
                   collect(a.value) AS aliases
            """,
            tenant=self._tenant,
        )
        _REGULATION_TYPES = {"AIRWORTHINESS_DIRECTIVE", "REGULATION", "TYPE_CERTIFICATE",
                              "SUPPLEMENTAL_TYPE_CERTIFICATE", "AIRWORTHINESS_APPROVAL"}

        self._exact.clear()
        self._stemmed.clear()
        self._fuzzy_by_length.clear()
        for row in rows:
            cname = row["canonical_name"]
            ctype = row["canonical_type"]
            self._exact[_normalize(cname)] = (cname, ctype)
            self._stemmed.setdefault(_normalize_ro(cname), (cname, ctype))
            # Also index under prefix-stripped key for regulatory entities so
            # "EASA AD 2022-0201" and "AD 2022-0201" resolve to the same node.
            if ctype in _REGULATION_TYPES:
                self._exact[_normalize_regulatory(cname)] = (cname, ctype)
            for alias in row.get("aliases") or []:
                if alias:
                    self._exact[_normalize(alias)] = (cname, ctype)
                    self._stemmed.setdefault(_normalize_ro(alias), (cname, ctype))
                    if ctype in _REGULATION_TYPES:
                        self._exact[_normalize_regulatory(alias)] = (cname, ctype)
        for path in self._curated_alias_files:
            try:
                self.apply_curated_aliases(load_curated_aliases(path))
            except (OSError, ValueError) as exc:
                log.error("alias_registry.curated_aliases_failed", path=str(path), error=str(exc))
        self._rebuild_fuzzy_index()
        self._loaded = True
        log.info("alias_registry.loaded", tenant=self._tenant, entries=len(self._exact))

        # Push to Redis so other workers share this alias table without each
        # doing a full Neo4j round-trip on startup.
        redis = await _get_redis()
        if redis:
            rkey = _REDIS_KEY.format(tenant=self._tenant)
            try:
                pipe = redis.pipeline()
                pipe.delete(rkey)
                if self._exact:
                    # Store as hash: normalized_alias → "canonical_name|canonical_type"
                    mapping = {k: f"{v[0]}|{v[1]}" for k, v in self._exact.items()}
                    pipe.hset(rkey, mapping=mapping)
                    pipe.expire(rkey, _REDIS_TTL)
                await pipe.execute()
                log.info("alias_registry.redis_pushed", tenant=self._tenant,
                         entries=len(self._exact))
            except Exception as exc:
                # ERROR: without this push every other worker loads aliases from
                # Neo4j independently instead of the shared Redis cache, causing
                # duplicate Neo4j scans on each worker startup.
                log.error("alias_registry.redis_push_failed", error=str(exc))
            finally:
                await redis.aclose()

    def resolve(
        self, raw_name: str, *, with_detail: bool = False, entity_type: str | None = None,
    ) -> tuple[str, str] | ResolvedMatch | AmbiguousMatch | None:
        """
        Resolve a raw name to (canonical_name, canonical_type).
        Checks in-memory cache first (O(1)), then falls back to None.
        Redis is used only during load() — resolve() stays sync and fast.
        Returns None if not found — caller should treat as new entity.

        ``with_detail=True`` returns a ``ResolvedMatch`` (name, type, method,
        score) instead of a bare tuple for an auto-resolved match — used by
        graph_writer.py to persist resolution_status/resolution_method.
        Default is unchanged so every other caller (and the 14+ existing
        tests that assert a literal ``(name, type)`` tuple) keeps working.

        ``entity_type`` scopes the fuzzy, acronym and containment steps to
        candidates of that type; the exact and alias steps are unscoped,
        since an alias is an explicit statement. With the lexical guards on,
        a fuzzy candidate with a conflicting token is skipped, one with an
        extra identifier can only reach review, and a miss is still sent to
        review when it is an acronym of a known name or names a known
        entity's head token.
        """
        key = _normalize(raw_name)

        def _match(
            pair: tuple[str, str], method: str, score: float | None = None,
            runner_ups: tuple[tuple[str, str, float], ...] = (),
        ):
            if with_detail:
                return ResolvedMatch(pair[0], pair[1], method, score, runner_ups)
            return pair

        # 1. Exact / normalized match (in-memory — loaded from Redis or Neo4j)
        if key in self._exact:
            return _match(self._exact[key], "exact")

        # 1b. Regulatory prefix-stripped match
        # "EASA AD 2022-0201" → "AD 2022-0201" canonical key
        reg_key = _normalize_regulatory(raw_name)
        if reg_key != key and reg_key in self._exact:
            log.debug("alias_registry.regulatory_prefix_match", raw=raw_name,
                      canonical=self._exact[reg_key][0])
            return _match(self._exact[reg_key], "exact")

        # 1c. Romanian noun-stem fallback match
        # "furnizori" / "furnizorul" / "furnizorii" / "furnizorilor" → "furnizor"
        stem_key = _normalize_ro(raw_name)
        if stem_key in self._stemmed:
            log.debug("alias_registry.stem_match", raw=raw_name,
                      canonical=self._stemmed[stem_key][0])
            return _match(self._stemmed[stem_key], "exact")

        # 2. Fuzzy match (optional, requires rapidfuzz).  Candidate selection
        # uses only a score upper bound derived from string lengths, therefore
        # preserves all matches that can reach the review threshold.
        try:
            from rapidfuzz import fuzz
            from graphrag.core.config import get_settings
            review_min = get_settings().ingestion.get("review_fuzzy_min", 70)
            best_score = 0
            best_match = None
            # Track every distinct canonical entity that scores >= review_min,
            # in the same iteration order _fuzzy_candidates() produces, so a
            # stable descending sort reproduces the exact winner the `score >
            # best_score` loop below picks (ties keep the earliest-seen
            # candidate) while also keeping whichever it beat, for
            # docs/IMPLEMENTATION_AUDIT.md's runner-up persistence.
            scored_candidates: list[tuple[float, tuple[str, str]]] = []
            seen_canonical: set[tuple[str, str]] = set()
            review_only = False
            for stored_key in self._fuzzy_candidates(key, minimum_score=review_min):
                canonical = self._exact[stored_key]
                if self._lexical_guards and (
                    not _same_type(entity_type, canonical[1]) or token_conflict(key, stored_key)
                ):
                    continue
                score = fuzz.ratio(key, stored_key)
                if score > best_score:
                    best_score = score
                    best_match = canonical
                    review_only = self._lexical_guards and token_extra_identifier(key, stored_key)
                if score >= review_min and canonical not in seen_canonical:
                    scored_candidates.append((score, canonical))
                    seen_canonical.add(canonical)
            if best_score >= self._fuzzy_threshold and best_match and not review_only:
                log.debug(
                    "alias_registry.fuzzy_match",
                    raw=raw_name,
                    canonical=best_match[0],
                    score=best_score,
                )
                scored_candidates.sort(key=lambda item: item[0], reverse=True)
                runner_ups = tuple(
                    (name, type_, score)
                    for score, (name, type_) in scored_candidates
                    if (name, type_) != best_match
                )[:2]
                return _match(best_match, "fuzzy", float(best_score), runner_ups)
            # Ambiguous band — close but not confident enough to auto-merge
            # (or above the threshold but only under-specified, review_only)
            if review_min <= best_score and best_match:
                log.debug(
                    "alias_registry.fuzzy_ambiguous",
                    raw=raw_name,
                    candidate=best_match[0],
                    score=best_score,
                )
                scored_candidates.sort(key=lambda item: item[0], reverse=True)
                ambiguous_runner_ups = tuple(
                    (name, type_, float(score))
                    for score, (name, type_) in scored_candidates
                    if (name, type_) != best_match
                )[:2]
                return AmbiguousMatch(candidate=best_match, score=float(best_score), match_type="fuzzy",
                                      runner_ups=ambiguous_runner_ups)
        except ImportError:
            pass  # rapidfuzz not installed — skip fuzzy step

        if self._lexical_guards:
            return self._review_candidate(raw_name, key, entity_type)
        return None

    def _review_candidate(self, raw_name: str, key: str, entity_type: str | None) -> AmbiguousMatch | None:
        """A miss that still deserves a reviewer: an acronym whose letters are
        the initials of known names ("FMEA"), or a name made only of tokens of
        a known name, including its head ("Northwind", "Inverter Module").
        Never an automatic merge; the score is 0 because no ratio applies."""
        self._ensure_token_index()
        found: list[tuple[str, str]] = []
        kind = ""
        if _looks_like_acronym(raw_name):
            kind = "acronym"
            found = [self._exact[k] for k in self._by_initials.get(key.replace(" ", ""), ())]
        else:
            tokens = _content_tokens(key)
            if tokens:
                kind = "containment"
                for k in self._by_head.get(tokens[0], ()):
                    candidate_tokens = _content_tokens(k)
                    if len(candidate_tokens) > len(tokens) and all(t in candidate_tokens for t in tokens):
                        found.append(self._exact[k])
        unique: list[tuple[str, str]] = []
        for pair in found:
            if pair not in unique and _same_type(entity_type, pair[1]):
                unique.append(pair)
        if not unique:
            return None
        log.debug("alias_registry.review_candidate", raw=raw_name, kind=kind, candidates=len(unique))
        return AmbiguousMatch(candidate=unique[0], score=0.0, match_type=kind,
                              runner_ups=tuple((n, t, 0.0) for n, t in unique[1:3]))

    def apply_curated_aliases(self, entries: list[dict]) -> tuple[int, int]:
        """Add curated aliases in memory; return (applied, skipped).

        An entry applies only when its canonical (name, type) is already a
        known entity of this tenant and its tenant, if given, is this one.
        An alias that already points elsewhere is left alone and skipped.
        """
        known = set(self._exact.values())
        applied = skipped = 0
        for entry in entries:
            pair = (entry["canonical"], entry["type"])
            tenant = entry.get("tenant")
            key = _normalize(entry["alias"])
            if (tenant and tenant != self._tenant) or pair not in known or self._exact.get(key, pair) != pair:
                skipped += 1
                continue
            self._exact[key] = pair
            applied += 1
        if applied:
            self._rebuild_fuzzy_index()
        log.info("alias_registry.curated_aliases", tenant=self._tenant, applied=applied, skipped=skipped)
        return applied, skipped

    def _ensure_token_index(self) -> None:
        if self._fuzzy_index_size != len(self._exact):
            self._rebuild_fuzzy_index()

    async def register_alias(
        self,
        raw_value: str,
        canonical_name: str,
        canonical_type: str,
        source_doc_id: str = "",
        source_chunk_id: str = "",
        evidence_kind: str = "resolution",
        evidence_quote: str = "",
        confidence: float = 1.0,
    ) -> None:
        """Persist a tenant-scoped alias and its best available provenance."""
        from uuid import uuid4
        await self._neo4j.run(
            """
            MATCH (e:Entity {name: $canonical_name, type: $canonical_type, tenant: $tenant})
            MERGE (a:Alias {value: $raw_value, tenant: $tenant})
            ON CREATE SET a.id           = $alias_id,
                          a.normalized   = $normalized,
                          a.source_doc   = $source_doc,
                          a.source_chunk_id = $source_chunk_id,
                          a.evidence_kind = $evidence_kind,
                          a.evidence_quote = $evidence_quote,
                          a.confidence   = $confidence,
                          a.created_at   = datetime()
            ON MATCH SET a.source_doc = CASE WHEN $evidence_kind <> 'resolution' THEN $source_doc ELSE a.source_doc END,
                         a.source_chunk_id = CASE WHEN $evidence_kind <> 'resolution' THEN $source_chunk_id ELSE a.source_chunk_id END,
                         a.evidence_kind = CASE WHEN $evidence_kind <> 'resolution' THEN $evidence_kind ELSE a.evidence_kind END,
                         a.evidence_quote = CASE WHEN $evidence_kind <> 'resolution' THEN $evidence_quote ELSE a.evidence_quote END,
                         a.confidence = CASE WHEN $confidence > coalesce(a.confidence, 0.0) THEN $confidence ELSE a.confidence END
            MERGE (a)-[:ALIAS_OF]->(e)
            """,
            canonical_name=canonical_name,
            canonical_type=canonical_type,
            tenant=self._tenant,
            raw_value=raw_value,
            alias_id=str(uuid4()),
            normalized=_normalize(raw_value),
            source_doc=source_doc_id,
            source_chunk_id=source_chunk_id,
            evidence_kind=evidence_kind,
            evidence_quote=evidence_quote,
            confidence=confidence,
        )
        self._exact[_normalize(raw_value)] = (canonical_name, canonical_type)
        self._stemmed.setdefault(_normalize_ro(raw_value), (canonical_name, canonical_type))
        self._index_fuzzy_key(_normalize(raw_value))
        log.info(
            "alias_registry.alias_added",
            raw=raw_value,
            canonical=canonical_name,
        )

    def _rebuild_fuzzy_index(self) -> None:
        """Build normalized-length buckets after a full alias-table refresh."""
        self._fuzzy_by_length.clear()
        self._by_head.clear()
        self._by_initials.clear()
        for key in self._exact:
            self._fuzzy_by_length.setdefault(len(key), []).append(key)
            self._index_tokens(key)
        self._fuzzy_index_size = len(self._exact)

    def _index_tokens(self, key: str) -> None:
        tokens = _content_tokens(key)
        if tokens:
            bucket = self._by_head.setdefault(tokens[0], [])
            if key not in bucket:
                bucket.append(key)
        initials = _initials(key)
        if len(initials) >= 2:
            bucket = self._by_initials.setdefault(initials, [])
            if key not in bucket:
                bucket.append(key)

    def _index_fuzzy_key(self, key: str) -> None:
        """Add a newly registered key without rebuilding the whole index."""
        bucket = self._fuzzy_by_length.setdefault(len(key), [])
        if key not in bucket:
            bucket.append(key)
        self._index_tokens(key)
        self._fuzzy_index_size = len(self._exact)

    def _fuzzy_candidates(self, key: str, minimum_score: float) -> list[str]:
        """Return all keys that can still reach ``minimum_score``.

        For two strings of lengths ``n`` and ``m``, even the best possible
        edit-distance ratio is ``2 * min(n, m) / (n + m)``.  The resulting
        bounds exclude only impossible fuzzy candidates, not merely unlikely
        ones, so matching semantics and the human-review band are unchanged.
        """
        if self._fuzzy_index_size != len(self._exact):
            # Tests and a few legacy writers populate _exact directly.  Keep
            # the index self-healing until those callers migrate to the public
            # registration path.
            self._rebuild_fuzzy_index()
        if not key:
            return []
        threshold = max(0.0, min(100.0, float(minimum_score))) / 100.0
        if threshold <= 0.0:
            return list(self._exact)
        key_length = len(key)
        min_length = max(1, int((threshold * key_length) / (2.0 - threshold)))
        max_length = int(key_length * ((2.0 / threshold) - 1.0))
        candidates: list[str] = []
        for length in range(min_length, max_length + 1):
            candidates.extend(self._fuzzy_by_length.get(length, ()))
        return candidates

    async def find_duplicate_by_embedding(
        self,
        embedding: list[float],
        entity_type: str,
        exclude_name: str = "",
        *,
        with_detail: bool = False,
        co_mentioned_names: list[str] | None = None,
    ) -> tuple[str, str, float] | EmbeddingMatch | None:
        """
        Search for an existing entity whose embedding is very close
        to the given one — tenant-scoped.  Returns (name, type, similarity) or None.

        Embedding similarity alone is not sufficient to auto-merge: the top
        embedding-scoring candidate must also clear a lexical-similarity
        floor (rapidfuzz ratio against `exclude_name`, the entity being
        resolved) as a corroborating signal. A candidate that clears the
        embedding threshold but fails the lexical floor is not returned here
        — the caller's existing ambiguous-band path (find_candidate_by_embedding)
        still surfaces it for human review instead of silently dropping it.

        ``with_detail=True`` returns an ``EmbeddingMatch`` (name, type, score,
        runner_ups) instead of a bare tuple — mirrors ``resolve()``'s own
        ``with_detail`` precedent. Default is unchanged: every existing
        caller keeps its original ``(name, type, score) | None`` contract.

        ``co_mentioned_names``, if given AND
        ``ingestion.alias_cooccurrence_tiebreak_enabled`` is set, breaks a
        near-tie (top two corroborating candidates within
        ``ingestion.alias_embedding_tie_margin`` of each other) using shared
        graph structure instead of embedding score alone — see
        ``_cooccurrence_tiebreak``'s docstring. Omitting it (the default for
        every caller not passing it) skips the tie-break unconditionally,
        independent of the config flag.
        """
        rows = await self._neo4j.run(
            """
            CALL db.index.vector.queryNodes('entity_embeddings', $fetch_k, $embedding)
            YIELD node AS e, score
            WHERE e.type = $entity_type
              AND e.name <> $exclude_name
              AND e.tenant = $tenant
              AND score >= $threshold
            RETURN e.name AS name, e.type AS type, score
            ORDER BY score DESC
            LIMIT 5
            """,
            fetch_k=_FETCH_K,
            embedding=embedding,
            entity_type=entity_type,
            exclude_name=exclude_name,
            tenant=self._tenant,
            threshold=self._embedding_threshold,
        )
        if not rows:
            return None
        try:
            from rapidfuzz import fuzz
        except ImportError:
            # rapidfuzz not installed — no lexical signal available, fall
            # back to the prior embedding-only behavior rather than crash.
            r = rows[0]
            if with_detail:
                return EmbeddingMatch(r["name"], r["type"], float(r["score"]))
            return r["name"], r["type"], float(r["score"])

        key = _normalize(exclude_name)
        corroborated: list[tuple[str, str, float]] = []
        for r in rows:
            lexical_score = fuzz.ratio(key, _normalize(r["name"]))
            if lexical_score >= self._embedding_min_lexical:
                corroborated.append((r["name"], r["type"], float(r["score"])))
            else:
                log.debug(
                    "alias_registry.embedding_dedup_no_lexical_corroboration",
                    raw=exclude_name,
                    candidate=r["name"],
                    embedding_score=round(float(r["score"]), 4),
                    lexical_score=lexical_score,
                )
        if not corroborated:
            return None

        winner = corroborated[0]
        if (
            len(corroborated) > 1
            and co_mentioned_names
            and self._cooccurrence_tiebreak_enabled
            and (corroborated[0][2] - corroborated[1][2]) <= self._embedding_tie_margin
        ):
            winner = await self._cooccurrence_tiebreak(
                corroborated, co_mentioned_names, entity_type,
            )

        if with_detail:
            runner_ups = tuple(c for c in corroborated if c != winner)
            return EmbeddingMatch(winner[0], winner[1], winner[2], runner_ups)
        return winner

    async def _cooccurrence_tiebreak(
        self,
        candidates: list[tuple[str, str, float]],
        co_mentioned_names: list[str],
        entity_type: str,
    ) -> tuple[str, str, float]:
        """Break a near-tie between embedding candidates using shared graph
        structure: how many of the OTHER entities extracted from the same
        chunk as this mention already have a RELATES_TO edge or a co-MENTIONS
        chunk with each candidate. Highest co-occurrence count wins; a tie in
        THAT count keeps the original (highest embedding score) ordering.

        Deliberately narrow: only called for an already-lexically-corroborated,
        already-score-near-tied candidate pair (see find_duplicate_by_embedding)
        — never a substitute for the existing score cascade, only a nudge.
        Ships disabled by default; see
        ``ingestion.alias_cooccurrence_tiebreak_enabled`` and
        docs/IMPLEMENTATION_AUDIT.md ("contextual disambiguation").
        """
        rows = await self._neo4j.run(
            """
            UNWIND $candidates AS cand
            MATCH (c:Entity {name: cand.name, type: $entity_type, tenant: $tenant})
            UNWIND $co_names AS co_name
            OPTIONAL MATCH (c)-[:RELATES_TO]-(co:Entity {name: co_name, tenant: $tenant})
            OPTIONAL MATCH (c)<-[:MENTIONS]-(:Chunk)-[:MENTIONS]->
                            (co2:Entity {name: co_name, tenant: $tenant})
            WITH cand, (co IS NOT NULL OR co2 IS NOT NULL) AS linked
            RETURN cand.name AS name, count(CASE WHEN linked THEN 1 END) AS cooccurrence_score
            """,
            candidates=[{"name": name} for name, _, _ in candidates],
            co_names=co_mentioned_names,
            entity_type=entity_type,
            tenant=self._tenant,
        )
        scores = {r["name"]: r["cooccurrence_score"] for r in rows}
        # Highest co-occurrence score wins; a tie in that score keeps the
        # original (highest-embedding-score-first) order — negate the score
        # rather than use reverse=True, which would also reverse the index
        # tie-break and pick the LOWEST-embedding-score candidate on a tie.
        ranked = sorted(
            candidates,
            key=lambda c: (-scores.get(c[0], 0), candidates.index(c)),
        )
        return ranked[0]

    async def find_candidate_by_embedding(
        self,
        embedding: list[float],
        entity_type: str,
        exclude_name: str = "",
        lower: float = 0.85,
        upper: float | None = None,
    ) -> tuple[str, str, float] | None:
        """Search for an entity whose embedding falls in the ambiguous band [lower, upper).

        Used to surface near-duplicate candidates for human review when confidence
        is too low to auto-merge but high enough to be suspicious. Returns the
        best match in the band, or None if none found.
        """
        if upper is None:
            upper = self._embedding_threshold
        rows = await self._neo4j.run(
            """
            CALL db.index.vector.queryNodes('entity_embeddings', $fetch_k, $embedding)
            YIELD node AS e, score
            WHERE e.type = $entity_type
              AND e.name <> $exclude_name
              AND e.tenant = $tenant
              AND score >= $lower
              AND score < $upper
            RETURN e.name AS name, e.type AS type, score
            ORDER BY score DESC
            LIMIT 1
            """,
            fetch_k=_FETCH_K,
            embedding=embedding,
            entity_type=entity_type,
            exclude_name=exclude_name,
            tenant=self._tenant,
            lower=lower,
            upper=upper,
        )
        if rows:
            r = rows[0]
            return r["name"], r["type"], float(r["score"])
        return None


# ── Per-tenant registry pool ──────────────────────────────────────────────────
# One AliasRegistry instance per tenant — entity identity is tenant-scoped.

_registries: dict[str, AliasRegistry] = {}


def load_curated_aliases(path) -> list[dict]:
    """Read a curated alias file: YAML or JSON, a list of
    ``{alias, canonical, type, tenant?}`` or ``{"aliases": [...]}``."""
    from pathlib import Path
    import json
    import yaml

    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text) if str(path).endswith(".json") else yaml.safe_load(text)
    entries = data.get("aliases", []) if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a list of aliases")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not all(isinstance(entry.get(f), str) and entry.get(f)
                                                  for f in ("alias", "canonical", "type")):
            raise ValueError(f"{path}: entry {index} needs alias, canonical and type")
    return entries


def get_alias_registry(neo4j_client=None, tenant: str = "default") -> AliasRegistry:
    global _registries
    if tenant not in _registries:
        if neo4j_client is None:
            from graphrag.graph.neo4j_client import get_neo4j
            neo4j_client = get_neo4j()
        _registries[tenant] = AliasRegistry(neo4j_client, tenant=tenant)
    return _registries[tenant]


async def load_alias_registry(neo4j_client=None, tenant: str = "default") -> AliasRegistry:
    """
    Load (or warm) the alias registry for a tenant.

    Tries Redis first — if another worker already pushed the alias table,
    this skips the Neo4j MATCH entirely.  Falls back to Neo4j on Redis miss
    and then pushes the result to Redis for the next worker.
    """
    registry = get_alias_registry(neo4j_client=neo4j_client, tenant=tenant)
    if registry._loaded:
        return registry

    # Try Redis warm-load — avoids full Neo4j scan on worker startup
    redis = await _get_redis()
    if redis:
        rkey = _REDIS_KEY.format(tenant=tenant)
        try:
            mapping = await redis.hgetall(rkey)
            if mapping:
                registry._exact = {
                    k: tuple(v.split("|", 1))   # type: ignore[assignment]
                    for k, v in mapping.items()
                }
                registry._stemmed = {}
                for k, v in registry._exact.items():
                    registry._stemmed.setdefault(_normalize_ro(k), v)
                registry._loaded = True
                log.info("alias_registry.redis_warm_load", tenant=tenant,
                         entries=len(registry._exact))
                # No aclose() here — the finally block below owns the close.
                # Closing in both places double-closed the client and relied on
                # the finally's bare `except Exception: pass` to swallow it.
                return registry
        except Exception as exc:
            log.warning("alias_registry.redis_load_failed", error=str(exc))
        finally:
            try:
                await redis.aclose()
            except Exception as exc:  # noqa: BLE001 — close failure must not mask the result
                log.debug("alias_registry.redis_close_failed", error=str(exc))

    # Redis miss or unavailable — load from Neo4j (also pushes to Redis)
    await registry.load()
    return registry
