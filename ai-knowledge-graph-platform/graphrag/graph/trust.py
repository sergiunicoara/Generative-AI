"""Trust metadata: origin, verification, authority, temporal validity, staleness.

Extends the existing provenance fields (``source_type``, ``confidence``,
``confidence_state``, ``authority_level``, ``valid_from/valid_to``) rather than
adding a parallel system (plan decision D5):

* ``origin`` — how a fact entered the graph: EXTRACTED (from a document by the
  extractor), IMPORTED (mapped from a structured source), INFERRED (by a rule),
  GENERATED (by a model without document grounding), MANUAL (entered by a person).
  Derived from ``source_type`` when not stored; ``source_type`` keeps its meaning.
* ``verification_status`` — UNVERIFIED unless a reviewer acted: VERIFIED
  (approved / manual override, with ``verified_by``) or REJECTED (retracted).
  Never inferred from the existence of a source.
* ``generated_by`` (extraction model or ``rule:<name>``), ``observed_at``
  (``extracted_at``), ``stale_after`` (fact needs re-verification after this),
  ``schema_version``, ``tenant``.

Trust is turned into separately observable multiplicative factors; nothing is
collapsed into a single opaque number.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Origin(str, Enum):
    EXTRACTED = "EXTRACTED"
    IMPORTED = "IMPORTED"
    INFERRED = "INFERRED"
    GENERATED = "GENERATED"
    MANUAL = "MANUAL"


class VerificationStatus(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"


_ORIGIN_BY_SOURCE_TYPE = {
    "document": Origin.EXTRACTED, "inferred": Origin.INFERRED,
    "llm": Origin.GENERATED, "manual": Origin.MANUAL,
}


def origin_for(source_type: Any, explicit: str | None = None) -> Origin:
    if explicit:
        return Origin(str(explicit).upper())
    value = getattr(source_type, "value", source_type)
    return _ORIGIN_BY_SOURCE_TYPE.get(str(value or "document").lower(), Origin.EXTRACTED)


# Defaults are deliberately mild; every factor is reported, so their effect is visible.
AUTHORITY_FACTOR = {1: 1.0, 2: 0.95, 3: 0.85, 4: 0.70}
ORIGIN_FACTOR = {Origin.MANUAL: 1.0, Origin.EXTRACTED: 1.0, Origin.IMPORTED: 1.0,
                 Origin.INFERRED: 0.85, Origin.GENERATED: 0.6}
VERIFICATION_FACTOR = {VerificationStatus.VERIFIED: 1.0, VerificationStatus.UNVERIFIED: 0.9,
                       VerificationStatus.REJECTED: 0.0}
STATE_FACTOR = {"APPROVED": 1.0, "ASSERTED": 1.0, "INFERRED": 1.0, "DISPUTED": 0.5, "RETRACTED": 0.0}
SUPERSEDED_FACTOR = 0.5
STALE_FACTOR = 0.7


def _parse(ts: Any) -> datetime | None:
    if ts in (None, "", "None"):
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass
class TrustAssessment:
    """Why a piece of evidence is (or is not) trusted as current."""
    origin: str
    verification_status: str
    confidence_state: str = "ASSERTED"
    authority_level: int | None = None
    superseded: bool = False
    expired: bool = False
    not_yet_valid: bool = False
    stale: bool = False
    verified_by: str | None = None
    generated_by: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    stale_after: str | None = None
    schema_version: str | None = None
    factors: dict[str, float] = field(default_factory=dict)

    @property
    def current(self) -> bool:
        """Usable as a statement about the present (or the requested ``as_of``)."""
        return not (self.superseded or self.expired or self.not_yet_valid or self.stale
                    or self.confidence_state == "RETRACTED"
                    or self.verification_status == VerificationStatus.REJECTED.value)

    @property
    def trust_factor(self) -> float:
        f = 1.0
        for v in self.factors.values():
            f *= v
        return round(f, 6)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["current"] = self.current
        d["trust_factor"] = self.trust_factor
        return d


def assess(record: dict, *, at: datetime | None = None, kind: str = "edge") -> TrustAssessment:
    """Assess an edge row (``edge_trust_fields``) or a chunk row (``document_trust_fields``)."""
    at = at or datetime.now(timezone.utc)
    if kind == "chunk":
        vf, vt, sa = record.get("doc_valid_from"), record.get("doc_valid_to"), record.get("doc_stale_after")
        origin = origin_for(record.get("source_type"), record.get("origin"))
    else:
        vf, vt, sa = record.get("valid_from"), record.get("valid_to"), record.get("stale_after")
        origin = origin_for(record.get("source_type"), record.get("origin"))
    verification = str(record.get("verification_status") or VerificationStatus.UNVERIFIED.value).upper()
    state = str(record.get("confidence_state") or "ASSERTED").upper()
    level = record.get("authority_level")
    try:
        level = int(level) if level is not None else None
    except (TypeError, ValueError):
        level = None
    a = TrustAssessment(
        origin=origin.value, verification_status=verification, confidence_state=state,
        authority_level=level, superseded=bool(record.get("superseded")),
        verified_by=record.get("verified_by"), generated_by=record.get("generated_by"),
        valid_from=vf, valid_to=vt, stale_after=sa, schema_version=record.get("schema_version"),
    )
    vf_dt, vt_dt, sa_dt = _parse(vf), _parse(vt), _parse(sa)
    a.not_yet_valid = bool(vf_dt and vf_dt > at)
    a.expired = bool(vt_dt and vt_dt <= at)
    a.stale = bool(sa_dt and sa_dt <= at)
    a.factors = {
        "authority": AUTHORITY_FACTOR.get(level, 1.0) if level is not None else 1.0,
        "origin": ORIGIN_FACTOR.get(origin, 1.0),
        "verification": VERIFICATION_FACTOR.get(VerificationStatus(verification)
                                                if verification in VerificationStatus.__members__ else
                                                VerificationStatus.UNVERIFIED, 0.9),
        "state": STATE_FACTOR.get(state, 1.0),
        "temporal": 0.0 if (a.expired or a.not_yet_valid) else 1.0,
        "supersession": SUPERSEDED_FACTOR if a.superseded else 1.0,
        "staleness": STALE_FACTOR if a.stale else 1.0,
    }
    return a


def apply_edge_trust(edges: list[dict], *, at: datetime | None = None) -> list[dict]:
    """Multiply each edge's ``confidence`` by its trust factor, keeping the parts.

    Adds ``trust`` (the assessment) and ``confidence_before_trust``; graph scoring
    (GNN adjacency, path confidence) then sees authority, origin, verification,
    dispute and staleness, and each part stays inspectable.
    """
    for e in edges:
        t = assess(e, at=at, kind="edge")
        e.setdefault("confidence_before_trust", e.get("confidence", 1.0))
        e["confidence"] = float(e.get("confidence_before_trust") or 1.0) * t.trust_factor
        e["trust"] = t.to_dict()
    return edges


def assess_chunk(chunk: dict, *, at: datetime | None = None) -> TrustAssessment:
    t = assess(chunk, at=at, kind="chunk")
    chunk["trust"] = t.to_dict()
    return t


def rank_conflicting_claims(claims: list[dict], *, at: datetime | None = None,
                            margin: float = 0.15) -> dict:
    """Order competing claims by trust; never collapse them silently.

    Each claim is a row with trust fields plus ``claim`` (e.g. the target value)
    and optionally ``confidence``. Returns every claim with its components and a
    ``winner`` only when the best claim is current and beats the runner-up by
    ``margin`` (relative); otherwise ``winner`` is None and ``status`` is
    ``unresolved``. A suggestion only: nothing is written.
    """
    ranked = []
    for c in claims:
        t = assess(c, at=at, kind="chunk" if "doc_valid_to" in c else "edge")
        base = float(c.get("confidence") or 1.0)
        ranked.append({**c, "trust": t.to_dict(), "score": base * t.trust_factor, "current": t.current})
    ranked.sort(key=lambda r: (r["current"], r["score"]), reverse=True)
    winner, status = None, "unresolved"
    if ranked and ranked[0]["current"]:
        runner = ranked[1]["score"] if len(ranked) > 1 else 0.0
        if ranked[0]["score"] > 0 and (ranked[0]["score"] - runner) / ranked[0]["score"] >= margin:
            winner, status = ranked[0], "suggested"
    return {"status": status, "winner": winner, "claims": ranked}



_SCORE_KEYS = ("vector_score", "bm25_score", "bm25_entity_score", "rrf_score", "score", "rerank_score",
               "text_score", "gnn_score", "path_confidence", "path_score", "sem_sim", "pagerank_tiebreak",
               "feedback_score", "fusion_components")


def score_components(chunk: dict) -> dict:
    """Every score a chunk accumulated, kept apart (nothing is collapsed)."""
    return {k: chunk[k] for k in _SCORE_KEYS if chunk.get(k) is not None}


def apply_chunk_trust(chunks: list[dict], *, at: datetime | None = None, enabled: bool = True,
                      authority: bool = False) -> bool:
    """Assess every chunk's evidence and fold the temporal part into its score.

    Always records ``trust`` and ``score_components``. When ``enabled``, the
    ranking score is multiplied by supersession x staleness x temporal factors
    (superseded or stale evidence ranks below current evidence instead of being
    treated as equally current); ``authority`` also applies the document
    authority factor (off by default: not yet evaluated on the golden sets).
    Returns True when any score changed, so the caller re-sorts.
    """
    changed = False
    for c in chunks:
        t = assess_chunk(c, at=at)
        base = float(c.get("final_score", c.get("rerank_score", c.get("score", 0.0))) or 0.0)
        applied = t.factors["supersession"] * t.factors["staleness"] * t.factors["temporal"]
        if authority:
            applied *= t.factors["authority"]
        c["score_components"] = {**score_components(c), "trust_factor_applied": applied,
                                 "score_before_trust": base}
        if enabled and applied != 1.0:
            c["final_score"] = base * applied
            changed = True
        c["score_components"]["final_score"] = c.get("final_score", base)
    return changed
