"""Structured, authorization-filtered explanation of an answer (plan Phase 6).

Built only from evidence the retrieval queries already returned for this
caller (every chunk/document read applies the tenant and ACL predicates), so it
cannot introduce nodes the caller may not see. Under access control the graph
parts (paths, inferences, entity resolution) are withheld entirely, matching
the retrieval safe mode, and that is stated as a limitation.

Deterministic: no LLM is involved. The answer ``confidence`` is a documented
heuristic over grounding, evidence currency and trust — not a calibrated
probability.
"""
from __future__ import annotations

import re
from statistics import mean
from typing import Any

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-./]{2,}")
_STOP = {"the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "which", "what",
         "have", "has", "not", "but", "its", "their", "there", "been", "also", "into", "than",
         "then", "they", "these", "those", "such", "per", "each", "other", "only", "more", "most",
         "can", "may", "must", "should", "would", "could", "will", "shall", "about", "under", "over"}
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_EXCERPT = 300


def _content(text: str) -> set[str]:
    return {w.lower().strip(".,;:()[]") for w in _WORD.findall(text or "")} - _STOP


def grounding_check(answer: str, evidence_texts: list[str], *, min_overlap: float = 0.5) -> dict:
    """Which answer statements are supported by the retrieved evidence text.

    A statement (sentence with >= 3 content words) is grounded when at least
    ``min_overlap`` of its content words occur in the evidence. Unsupported
    statements are listed so they are never presented as grounded.
    """
    vocab: set[str] = set()
    for t in evidence_texts:
        vocab |= _content(t)
    statements, unsupported = 0, []
    for sentence in (s.strip() for s in _SENT.split(answer or "")):
        words = _content(sentence)
        if len(words) < 3:
            continue
        statements += 1
        if len(words & vocab) / len(words) < min_overlap:
            unsupported.append(sentence[:200])
    grounded = statements - len(unsupported)
    return {"statements": statements, "grounded": grounded, "unsupported_statements": unsupported,
            "grounding_ratio": round(grounded / statements, 4) if statements else 1.0}


def _edge_view(e: dict) -> dict:
    trust = e.get("trust") or {}
    return {
        "src": e.get("src"), "src_type": e.get("src_type"), "relation": e.get("relation"),
        "tgt": e.get("tgt"), "tgt_type": e.get("tgt_type"),
        "origin": trust.get("origin") or e.get("origin"),
        "confidence_state": trust.get("confidence_state") or e.get("confidence_state"),
        "verification_status": trust.get("verification_status") or e.get("verification_status"),
        "confidence": e.get("confidence"),
        "confidence_before_trust": e.get("confidence_before_trust"),
        "trust_factors": trust.get("factors"),
        "source_document_id": e.get("source_doc_id"),
    }


def build_explanation(
    *,
    answer: str,
    route: dict | None,
    local_results: dict,
    evidence: list,
    citations: list[str],
    sufficiency: dict | None,
    schema_version: str | None,
    fallback: dict,
    acl_enforced: bool,
    router_policy: str = "observe",
) -> dict[str, Any]:
    chunks = list(local_results.get("chunks") or [])
    cited_labels = set(citations or [])
    limitations: list[dict] = []

    # ── evidence (already ACL-filtered by the retrieval queries) ─────────────
    evidence_items: list[dict] = []
    for c in chunks:
        label = c.get("source") or c.get("_doc_name") or c.get("chunk_id")
        trust = c.get("trust") or {}
        evidence_items.append({
            "id": c.get("chunk_id"),
            "kind": "chunk",
            "source": label,
            "document_id": c.get("document_id"),
            "cited": label in cited_labels,
            "excerpt": (c.get("text") or "")[:_EXCERPT],
            "origin": trust.get("origin", "EXTRACTED"),
            "current": trust.get("current"),
            "trust": trust or None,
            "score_components": c.get("score_components"),
            "retrieved_via": c.get("retrieval") or ("graph_path" if c.get("path_length") else None),
        })

    # ── graph paths, inferences, entity resolution (withheld under ACL) ──────
    graph_paths: list[dict] = []
    inferences: list[dict] = []
    entity_resolution: list[dict] = []
    edges = list(local_results.get("entity_edges") or [])
    if acl_enforced:
        if edges or local_results.get("entities"):
            limitations.append({"code": "graph_explanation_withheld",
                                "message": "Graph paths and entity details are not shown under access control."})
    else:
        for e in edges[:25]:
            graph_paths.append({"edges": [_edge_view(e)], "length": 1})
            if (e.get("source_type") == "inferred" or (e.get("trust") or {}).get("origin") == "INFERRED"):
                inferences.append({
                    "fact": f"{e.get('src')} -{e.get('relation')}-> {e.get('tgt')}",
                    "rule": e.get("inferred_by"),
                    "rule_version": e.get("rule_version"),
                    "premises": list(e.get("premise_keys") or []),
                    "derivation_recorded": bool(e.get("premise_keys")),
                })
        for c in chunks:
            if c.get("path_length"):
                graph_paths.append({"via_entity": c.get("via_entity"), "length": c.get("path_length"),
                                    "path_confidence": c.get("path_confidence"),
                                    "reached_chunk": c.get("chunk_id")})
        for ent in (local_results.get("entities") or [])[:25]:
            entity_resolution.append({
                "entity": ent.get("entity"), "type": ent.get("type"),
                "resolution_status": ent.get("resolution_status") or "unrecorded",
                "resolution_method": ent.get("resolution_method"),
            })
    if any(not i["derivation_recorded"] for i in inferences):
        limitations.append({"code": "inference_without_recorded_premises",
                            "message": "Some inferred facts predate premise recording; their derivation is the rule only."})

    # ── grounding ────────────────────────────────────────────────────────────
    texts = [c.get("text") or "" for c in chunks] + [
        f"{e.get('src')} {e.get('relation')} {e.get('tgt')}" for e in edges]
    grounding = grounding_check(answer, texts)
    if grounding["unsupported_statements"]:
        limitations.append({"code": "unsupported_statements",
                            "message": f"{len(grounding['unsupported_statements'])} statement(s) are not "
                                       "supported by the retrieved evidence and are not presented as grounded."})

    # ── trust / currency of the evidence ─────────────────────────────────────
    assessed = [i for i in evidence_items if i["trust"]]
    not_current = [i for i in assessed if i["current"] is False]
    if not_current:
        limitations.append({"code": "non_current_evidence",
                            "message": f"{len(not_current)} evidence item(s) are superseded, expired or stale."})
    unverified = [i for i in assessed if (i["trust"] or {}).get("verification_status") == "UNVERIFIED"]
    if assessed and len(unverified) == len(assessed):
        limitations.append({"code": "unverified_evidence",
                            "message": "No evidence item has been verified by a reviewer."})
    if any((g.get("edges") or [{}])[0].get("confidence_state") == "DISPUTED" for g in graph_paths):
        limitations.append({"code": "disputed_facts", "message": "The graph context includes disputed facts."})
    if not schema_version or schema_version == "platform/v1":
        limitations.append({"code": "schema_unregistered",
                            "message": "No registered schema version was active for this tenant."})

    # ── insufficient context ─────────────────────────────────────────────────
    insufficient = None
    if sufficiency and not sufficiency.get("sufficient", True):
        insufficient = {"reason": sufficiency.get("reason_code"),
                        "evidence_count": sufficiency.get("evidence_count")}
        limitations.append({"code": f"insufficient_context:{sufficiency.get('reason_code')}",
                            "message": "The retrieved evidence did not clear the sufficiency gate."})
    if fallback.get("triggered"):
        limitations.append({"code": "agentic_fallback", "message": f"Answered by the bounded agentic "
                                                                   f"fallback ({fallback.get('reason')})."})
    if route and router_policy != "enforce":
        limitations.append({"code": "route_observed_only",
                            "message": "Route recorded but retrieval used the default strategy (policy=observe)."})

    # ── score breakdown and confidence ───────────────────────────────────────
    def _mean(values):
        vals = [float(v) for v in values if v is not None]
        return round(mean(vals), 4) if vals else None

    cited = [i for i in evidence_items if i["cited"]] or evidence_items
    factors = [(i["trust"] or {}).get("factors") or {} for i in cited]
    comps = [i["score_components"] or {} for i in cited]
    score_breakdown = {
        "text": _mean(c.get("text_score", c.get("rerank_score")) for c in comps),
        "vector": _mean(c.get("vector_score") for c in comps),
        "bm25": _mean(c.get("bm25_score") for c in comps),
        "graph": _mean(c.get("gnn_score") for c in comps),
        "path": _mean(c.get("path_score") for c in comps),
        "temporal": _mean(f.get("temporal", 1.0) * f.get("staleness", 1.0) * f.get("supersession", 1.0)
                          for f in factors if f),
        "provenance": _mean(f.get("origin", 1.0) * f.get("verification", 1.0) for f in factors if f),
        "authority": _mean(f.get("authority") for f in factors if f),
        "per_evidence": {i["id"]: i["score_components"] for i in cited if i["score_components"]},
    }
    currency = (sum(1 for i in cited if i["current"] is not False) / len(cited)) if cited else 0.0
    trust_factor = _mean((i["trust"] or {}).get("trust_factor") for i in cited) or (1.0 if cited else 0.0)
    confidence = round(grounding["grounding_ratio"] * currency * trust_factor, 4) if cited else 0.0
    if insufficient:
        confidence = min(confidence, 0.25)

    return {
        "answer": answer,
        "confidence": confidence,
        "confidence_components": {"grounding": grounding["grounding_ratio"], "evidence_currency": round(currency, 4),
                                  "evidence_trust": trust_factor, "capped_by_insufficiency": bool(insufficient),
                                  "note": "heuristic, not a calibrated probability"},
        "route": (route or {}).get("route"),
        "route_reason": (route or {}).get("reason"),
        "evidence": evidence_items,
        "citations": list(citations or []),
        "graph_paths": graph_paths,
        "inferences": inferences,
        "entity_resolution": entity_resolution,
        "score_breakdown": score_breakdown,
        "grounding": grounding,
        "schema_version": schema_version,
        "fallback": {"triggered": bool(fallback.get("triggered")), "reason": fallback.get("reason")},
        "insufficient_context": insufficient,
        "limitations": limitations,
    }
