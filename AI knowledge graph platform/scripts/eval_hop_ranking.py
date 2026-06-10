"""
Eval gate for multihop semantic blending (multihop_semantic_weight).

Compares hop-chunk ranking with pure path_score (w=0) vs the semantic blend
(w from settings.yml) on the golden set, WITHOUT any LLM synthesis or judge —
this isolates the retrieval-ranking change from generation noise.

For each golden question with expected_citations, runs the full LocalSearch
pipeline (vector + BM25 + RRF + rerank + multihop + GNN) under both weights
and measures, on the final retrieved chunk set:

  - hit_rate   : >= 1 retrieved chunk belongs to an expected-citation document
  - coverage   : fraction of expected citation docs present in retrieval
  - mrr        : 1 / rank of the first chunk from an expected-citation doc
  - latency    : wall-clock per search call

Verdict: blend wins if coverage or MRR improves without hit_rate regressing.
Results written to evals/hop_ranking_eval_results.json.

Usage:
    python scripts/eval_hop_ranking.py            # full golden set
    python scripts/eval_hop_ranking.py --limit 10 # quick smoke run
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

TENANT = "aerospace"


def _norm(s: str) -> str:
    return s.lower().replace("_", "-").strip()


# Golden-set expected_citations use the seed-data doc ids; the real ingested
# corpus stores human identifiers in Document.filename. Explicit map for the
# cases where substring matching can't bridge the two namings.
CITATION_TO_FILENAME = {
    "faa-ad-2024": "FAA-AD-2024-01-02.txt",
    "faa-ad-2022": "FAA-AD-2022-03-07.txt",
    "faa-ad-2020-old": "FAA-AD-2020-05-11.txt",
    "easa-ad-2024": "EASA-AD-2024-0072.txt",
    "boeing-profile": "Boeing_company_profile.txt",
    "boeing-swcr": "Boeing_MCAS_SWChangeRecord.txt",
    "fleet-registry": "SWA_fleet_registry_2024.txt",
    "maintenance-manual": "737MAX_CMM_Engine_Mount.txt",
    "inspection-report-2024-01": "G-ABCD_inspection_2024-01.txt",
    "ad-compliance-check-2024-03": "G-ABCD_AD_compliance_2024-03.txt",
}


def _doc_matches(doc_id: str, citation: str) -> bool:
    """Match 'faa-ad-2024' against doc filenames like 'FAA-AD-2024-01-02.txt'."""
    mapped = CITATION_TO_FILENAME.get(_norm(citation))
    if mapped is not None:
        return _norm(doc_id) == _norm(mapped)
    d, c = _norm(doc_id), _norm(citation)
    return c in d or d in c


async def _chunk_doc_map(neo4j, chunk_ids: list[str]) -> dict[str, str]:
    """chunk_id -> document id via PART_OF."""
    if not chunk_ids:
        return {}
    rows = await neo4j.run(
        """
        UNWIND $ids AS cid
        MATCH (c:Chunk {id: cid})-[:PART_OF]->(d:Document)
        // d.id is a UUID — the filename carries the human doc identifier
        // that golden expected_citations reference (e.g. FAA-AD-2024-01-02.txt)
        RETURN c.id AS chunk_id, coalesce(d.filename, d.id) AS doc_id
        """,
        ids=chunk_ids,
    )
    return {r["chunk_id"]: r["doc_id"] for r in rows}


def _score_retrieval(
    ordered_chunk_ids: list[str],
    chunk_to_doc: dict[str, str],
    citations: list[str],
) -> dict:
    hit_rank = None
    found: set[str] = set()
    for rank, cid in enumerate(ordered_chunk_ids, start=1):
        doc = chunk_to_doc.get(cid, "")
        for cit in citations:
            if doc and _doc_matches(doc, cit):
                found.add(cit)
                if hit_rank is None:
                    hit_rank = rank
    return {
        "hit": hit_rank is not None,
        "coverage": len(found) / len(citations) if citations else None,
        "mrr": (1.0 / hit_rank) if hit_rank else 0.0,
    }


async def _run_arm(ls, neo4j, questions: list[dict], weight: float) -> dict:
    """Run LocalSearch over all questions at a given semantic weight."""
    # _cfg is the shared settings dict — replace with a copy so the other
    # arm (and the cached Settings singleton) is never mutated.
    ls._cfg = {**ls._cfg, "multihop_semantic_weight": weight}

    per_q, latencies = [], []
    for q in questions:
        t0 = time.perf_counter()
        result = await ls.search(q["question"], tenant=TENANT)
        latencies.append(time.perf_counter() - t0)

        chunk_ids = [c["chunk_id"] for c in result.get("chunks", [])]
        doc_map = await _chunk_doc_map(neo4j, chunk_ids)
        scores = _score_retrieval(chunk_ids, doc_map, q["expected_citations"])
        per_q.append({"id": q["id"], **scores})

    n = len(per_q)
    return {
        "weight": weight,
        "n_questions": n,
        "hit_rate": sum(1 for r in per_q if r["hit"]) / n,
        "mean_coverage": sum(r["coverage"] for r in per_q) / n,
        "mrr": sum(r["mrr"] for r in per_q) / n,
        "p50_latency_s": sorted(latencies)[n // 2],
        "per_question": per_q,
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="cap question count")
    args = parser.parse_args()

    from graphrag.core.config import get_settings
    from graphrag.graph.neo4j_client import get_neo4j
    from graphrag.retrieval.local_search import LocalSearch

    golden = json.loads(
        (Path(__file__).parents[1] / "evals" / "golden_set.json").read_text()
    )
    questions = [q for q in golden["questions"] if q.get("expected_citations")]
    if args.limit:
        questions = questions[: args.limit]
    print(f"Eval gate: {len(questions)} golden questions with expected citations\n")

    configured_w = get_settings().retrieval.get("multihop_semantic_weight", 0.5)
    ls, neo4j = LocalSearch(), get_neo4j()

    baseline = await _run_arm(ls, neo4j, questions, weight=0.0)
    print(f"[baseline w=0.0]  hit={baseline['hit_rate']:.3f}  "
          f"coverage={baseline['mean_coverage']:.3f}  mrr={baseline['mrr']:.3f}  "
          f"p50={baseline['p50_latency_s']:.2f}s")

    blended = await _run_arm(ls, neo4j, questions, weight=configured_w)
    print(f"[blend    w={configured_w}]  hit={blended['hit_rate']:.3f}  "
          f"coverage={blended['mean_coverage']:.3f}  mrr={blended['mrr']:.3f}  "
          f"p50={blended['p50_latency_s']:.2f}s")

    d_cov = blended["mean_coverage"] - baseline["mean_coverage"]
    d_mrr = blended["mrr"] - baseline["mrr"]
    d_hit = blended["hit_rate"] - baseline["hit_rate"]
    if d_hit < 0:
        verdict = "REGRESSION — blend loses whole-question hits; keep w=0"
    elif d_cov > 0 or d_mrr > 0:
        verdict = "IMPROVEMENT — blend lifts coverage/MRR with no hit regression"
    else:
        verdict = "NO CHANGE — blend neither helps nor hurts; keep for dense corpora or revert for simplicity"
    # ASCII only — Windows console (cp1252) chokes on unicode deltas
    print(f"\nd_hit={d_hit:+.3f}  d_coverage={d_cov:+.3f}  d_mrr={d_mrr:+.3f}")
    print(f"VERDICT: {verdict}")

    out = Path(__file__).parents[1] / "evals" / "hop_ranking_eval_results.json"
    out.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(),
        "tenant": TENANT,
        "baseline": baseline,
        "blended": blended,
        "deltas": {"hit_rate": d_hit, "coverage": d_cov, "mrr": d_mrr},
        "verdict": verdict,
    }, indent=2))
    print(f"\nResults -> {out}")
    return 0 if d_hit >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
