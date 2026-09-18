"""One-off measurement: query-personalized PageRank's retrieval-quality delta
on the real aerospace corpus.

Not a feature -- see docs/roadmap.md "Experimental -- benchmark before
implementation": "Query-personalized PageRank versus existing graph
expansion/PageRank." Follows the exact same pattern as
benchmark_mmr_quality.py and benchmark_splade_impact.py: real live Neo4j,
real golden-set questions, real embeddings, baseline vs. variant on the same
candidate pool, honest hit/coverage/MRR + latency reporting.

The "existing PageRank" this compares against is `run_pagerank()`
(neo4j_client.py) as currently wired into retrieval: a GLOBAL, non-seeded
PageRank used only as a disabled-by-default, low-confidence tiebreak in
local_search.py (pagerank_tiebreak_enabled=False). There is no seeded/
personalized PageRank anywhere in this codebase today. The baseline here is
therefore "current production retrieval with no PageRank involvement at
all" (fused BM25+vector RRF, top_k), which is both what ships today and a
fair apples-to-apples baseline against the MMR/SPLADE benchmarks' baseline.

Variant: seed GDS Personalized PageRank (`sourceNodes`) with the entities
MENTIONed by the top-N candidate chunks from the initial fused ranking (i.e.
query-relevant entities, not a query-independent global run), propagate
importance over the tenant's Entity/RELATES_TO graph, then rerank the full
candidate pool by blending each chunk's original relevance with the
personalized-PageRank score of the entities it mentions.

Usage:
    python scripts/benchmark_personalized_pagerank_quality.py            # full golden set
    python scripts/benchmark_personalized_pagerank_quality.py --limit 3  # smoke test
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

TENANT = "aerospace"
CANDIDATE_POOL_K = 50   # matches benchmark_mmr_quality.py / benchmark_splade_impact.py
FINAL_TOP_K = 10
SEED_FROM_TOP_N_CHUNKS = 5  # entities mentioned by the current top-5 chunks seed the personalization
BLEND_ALPHA = 0.5           # equal weight between original relevance and personalized-PageRank boost


def _norm(s: str) -> str:
    return s.lower().replace("_", "-").strip()


# Same map as benchmark_mmr_quality.py / benchmark_splade_impact.py / eval_hop_ranking.py.
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
    mapped = CITATION_TO_FILENAME.get(_norm(citation))
    if mapped is not None:
        return _norm(doc_id) == _norm(mapped)
    d, c = _norm(doc_id), _norm(citation)
    return c in d or d in c


async def _chunk_doc_map(neo4j, chunk_ids: list[str]) -> dict[str, str]:
    if not chunk_ids:
        return {}
    rows = await neo4j.run(
        """
        UNWIND $ids AS cid
        MATCH (c:Chunk {id: cid})-[:PART_OF]->(d:Document)
        RETURN c.id AS chunk_id, coalesce(d.filename, d.id) AS doc_id
        """,
        ids=chunk_ids,
    )
    return {r["chunk_id"]: r["doc_id"] for r in rows}


async def _chunk_entities(neo4j, chunk_ids: list[str]) -> dict[str, list[str]]:
    """chunk_id -> [entity_id, ...] via the real MENTIONS edge."""
    if not chunk_ids:
        return {}
    rows = await neo4j.run(
        """
        UNWIND $ids AS cid
        MATCH (c:Chunk {id: cid, tenant: $tenant})-[:MENTIONS]->(e:Entity {tenant: $tenant})
        RETURN c.id AS chunk_id, collect(e.id) AS entity_ids
        """,
        ids=chunk_ids,
        tenant=TENANT,
    )
    return {r["chunk_id"]: r["entity_ids"] for r in rows}


def _score_retrieval(ordered_chunk_ids: list[str], chunk_to_doc: dict[str, str],
                      citations: list[str]) -> dict:
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


class _PersonalizedPageRank:
    """Projects the tenant's Entity/RELATES_TO graph once via GDS and reuses
    it across many `sourceNodes`-seeded PageRank runs -- projecting per query
    would make the latency measurement mostly reflect projection cost, not
    the personalization itself.
    """

    def __init__(self, neo4j, tenant: str):
        self._neo4j = neo4j
        self._tenant = tenant
        self._graph_name = f"pp_pagerank_bench_{tenant}"

    async def __aenter__(self) -> "_PersonalizedPageRank":
        await self._neo4j.run("CALL gds.graph.drop($name, false)", name=self._graph_name)
        await self._neo4j.run(
            """
            CALL gds.graph.project.cypher(
              $name,
              'MATCH (e:Entity {tenant: $tenant}) WHERE coalesce(e.quarantined,false)=false RETURN id(e) AS id',
              'MATCH (a:Entity {tenant: $tenant})-[r:RELATES_TO {tenant: $tenant}]->(b:Entity {tenant: $tenant})
               RETURN id(a) AS source, id(b) AS target, coalesce(r.weight, r.confidence, 1.0) AS weight',
              {parameters: {tenant: $tenant}}
            )
            """,
            name=self._graph_name,
            tenant=self._tenant,
        )
        return self

    async def __aexit__(self, *exc) -> None:
        await self._neo4j.run("CALL gds.graph.drop($name, false)", name=self._graph_name)

    async def _native_ids(self, entity_ids: list[str]) -> list[int]:
        if not entity_ids:
            return []
        rows = await self._neo4j.run(
            "MATCH (e:Entity {tenant: $tenant}) WHERE e.id IN $ids RETURN id(e) AS nid",
            tenant=self._tenant, ids=entity_ids,
        )
        return [r["nid"] for r in rows]

    async def personalized_scores(self, seed_entity_ids: list[str]) -> tuple[dict[str, float], float]:
        """Returns (entity_id -> personalized score, wall-clock seconds).
        Falls back to an empty score map (i.e. no boost) when there is no
        usable seed -- a personalization with nothing to seed from
        degenerates to "no reranking", not an error.
        """
        source_nodes = await self._native_ids(seed_entity_ids)
        if not source_nodes:
            return {}, 0.0
        t0 = time.perf_counter()
        rows = await self._neo4j.run(
            """
            CALL gds.pageRank.stream($name, {
              sourceNodes: $sourceNodes,
              dampingFactor: 0.85,
              maxIterations: 20,
              relationshipWeightProperty: 'weight'
            })
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).id AS entity_id, score
            """,
            name=self._graph_name,
            sourceNodes=source_nodes,
        )
        elapsed = time.perf_counter() - t0
        return {r["entity_id"]: r["score"] for r in rows if r["entity_id"] is not None}, elapsed


def _rerank_with_personalization(
    fused: list[dict], chunk_entities: dict[str, list[str]], entity_scores: dict[str, float],
) -> list[dict]:
    if not entity_scores:
        return fused[:FINAL_TOP_K]

    raw_scores = [c.get("score", 0.0) for c in fused]
    lo, hi = min(raw_scores), max(raw_scores)
    relevance = {
        c["chunk_id"]: ((c.get("score", 0.0) - lo) / (hi - lo) if hi > lo else 1.0)
        for c in fused
    }

    raw_boosts = {
        c["chunk_id"]: max((entity_scores.get(e, 0.0) for e in chunk_entities.get(c["chunk_id"], [])), default=0.0)
        for c in fused
    }
    blo, bhi = min(raw_boosts.values()), max(raw_boosts.values())

    def _norm_boost(chunk_id: str) -> float:
        b = raw_boosts[chunk_id]
        return (b - blo) / (bhi - blo) if bhi > blo else 0.0

    blended = [
        (c, BLEND_ALPHA * relevance[c["chunk_id"]] + (1 - BLEND_ALPHA) * _norm_boost(c["chunk_id"]))
        for c in fused
    ]
    blended.sort(key=lambda pair: pair[1], reverse=True)
    return [c for c, _ in blended[:FINAL_TOP_K]]


async def _run_question(embedder, neo4j, bm25, ppr: _PersonalizedPageRank, q: dict) -> dict:
    question = q["question"]

    embedding = await embedder.embed_text(question)
    vector_chunks = await neo4j.vector_search_chunks(embedding, top_k=CANDIDATE_POOL_K, tenant=TENANT)
    fused = await bm25.search(query=question, vector_chunks=vector_chunks, top_k=CANDIDATE_POOL_K, tenant=TENANT)

    current_top = fused[:FINAL_TOP_K]  # what production returns today (no PageRank involvement)

    all_ids = [c["chunk_id"] for c in fused]
    chunk_entities = await _chunk_entities(neo4j, all_ids)

    seed_chunk_ids = [c["chunk_id"] for c in fused[:SEED_FROM_TOP_N_CHUNKS]]
    seed_entities = sorted({e for cid in seed_chunk_ids for e in chunk_entities.get(cid, [])})
    entity_scores, ppr_latency = await ppr.personalized_scores(seed_entities)

    ppr_top = _rerank_with_personalization(fused, chunk_entities, entity_scores)

    doc_map = await _chunk_doc_map(neo4j, all_ids)
    current_scores = _score_retrieval([c["chunk_id"] for c in current_top], doc_map, q["expected_citations"])
    ppr_scores = _score_retrieval([c["chunk_id"] for c in ppr_top], doc_map, q["expected_citations"])

    return {
        "id": q["id"],
        "candidate_pool_size": len(fused),
        "seed_entity_count": len(seed_entities),
        "entities_scored": len(entity_scores),
        "ppr_latency_ms": ppr_latency * 1000,
        "current": current_scores,
        "personalized_pagerank": ppr_scores,
    }


def _summarize(per_q: list[dict], key: str) -> dict:
    n = len(per_q)
    return {
        "hit_rate": sum(1 for r in per_q if r[key]["hit"]) / n,
        "mean_coverage": sum(r[key]["coverage"] for r in per_q) / n,
        "mrr": sum(r[key]["mrr"] for r in per_q) / n,
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="cap question count")
    args = parser.parse_args()

    from graphrag.graph.neo4j_client import get_neo4j
    from graphrag.ingestion.embedder import Embedder
    from graphrag.retrieval.bm25_search import HybridBM25Search

    golden = json.loads((Path(__file__).parents[1] / "evals" / "golden_set.json").read_text())
    questions = [q for q in golden["questions"] if q.get("expected_citations")]
    if args.limit:
        questions = questions[: args.limit]
    print(f"Personalized PageRank quality benchmark: {len(questions)} golden questions "
          f"(candidate pool={CANDIDATE_POOL_K}, final top_k={FINAL_TOP_K}, "
          f"seed from top {SEED_FROM_TOP_N_CHUNKS} chunks, blend alpha={BLEND_ALPHA})\n")

    embedder, neo4j, bm25 = Embedder(), get_neo4j(), HybridBM25Search()

    per_q = []
    async with _PersonalizedPageRank(neo4j, TENANT) as ppr:
        for i, q in enumerate(questions, start=1):
            result = await _run_question(embedder, neo4j, bm25, ppr, q)
            per_q.append(result)
            print(f"[{i}/{len(questions)}] {q['id']}: "
                  f"current mrr={result['current']['mrr']:.2f} "
                  f"ppr mrr={result['personalized_pagerank']['mrr']:.2f} "
                  f"seeds={result['seed_entity_count']} scored={result['entities_scored']} "
                  f"ppr_latency={result['ppr_latency_ms']:.1f}ms")

    current_summary = _summarize(per_q, "current")
    ppr_summary = _summarize(per_q, "personalized_pagerank")

    latencies = sorted(r["ppr_latency_ms"] for r in per_q)
    n = len(latencies)
    latency_summary = {
        "mean_ms": sum(latencies) / n,
        "p50_ms": latencies[n // 2],
        "p95_ms": latencies[int(n * 0.95)] if n > 1 else latencies[0],
        "max_ms": latencies[-1],
    }

    improved = sum(1 for r in per_q if r["personalized_pagerank"]["mrr"] > r["current"]["mrr"])
    regressed = sum(1 for r in per_q if r["personalized_pagerank"]["mrr"] < r["current"]["mrr"])
    tied = len(per_q) - improved - regressed

    print(f"\n[current BM25+vector RRF]      hit={current_summary['hit_rate']:.3f}  "
          f"coverage={current_summary['mean_coverage']:.3f}  mrr={current_summary['mrr']:.3f}")
    print(f"[personalized PageRank rerank] hit={ppr_summary['hit_rate']:.3f}  "
          f"coverage={ppr_summary['mean_coverage']:.3f}  mrr={ppr_summary['mrr']:.3f}")
    print(f"\nMRR: improved={improved}  regressed={regressed}  tied={tied}")
    print(f"Personalized PageRank latency per query: mean={latency_summary['mean_ms']:.1f}ms  "
          f"p50={latency_summary['p50_ms']:.1f}ms  p95={latency_summary['p95_ms']:.1f}ms  "
          f"max={latency_summary['max_ms']:.1f}ms")

    out = Path(__file__).parents[1] / "evals" / "personalized_pagerank_quality_results.json"
    out.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(),
        "tenant": TENANT,
        "n_questions": len(per_q),
        "candidate_pool_k": CANDIDATE_POOL_K,
        "final_top_k": FINAL_TOP_K,
        "seed_from_top_n_chunks": SEED_FROM_TOP_N_CHUNKS,
        "blend_alpha": BLEND_ALPHA,
        "current": current_summary,
        "personalized_pagerank": ppr_summary,
        "mrr_improved": improved,
        "mrr_regressed": regressed,
        "mrr_tied": tied,
        "ppr_latency_ms": latency_summary,
        "per_question": per_q,
    }, indent=2))
    print(f"\nResults -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
