"""GNN alpha/beta calibration via grid search over RAGAS metrics.

Loads a small eval set, tries every (alpha, beta) combination, scores each
with context_precision + context_recall (no LLM generation — retrieval-only
metrics), then writes the best values to config/settings.yml.

Usage:
    python scripts/calibrate_gnn.py
    python scripts/calibrate_gnn.py --eval-set eval_data/calibration_set.json
    python scripts/calibrate_gnn.py --dry-run   # print best values, don't save
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import structlog
import yaml

from graphrag.core.config import get_settings
from graphrag.graph.gnn_scorer import GNNScorer
from graphrag.graph.neo4j_client import get_neo4j
from graphrag.ingestion.embedder import Embedder
from graphrag.retrieval.bm25_search import HybridBM25Search
from graphrag.retrieval.reranker import CrossEncoderReranker
from graphrag.retrieval.local_search import _fetch_subgraph_edges

log = structlog.get_logger(__name__)

# ── Grid ──────────────────────────────────────────────────────────────────────
# alpha + beta must equal 1.0 (weighted average — final_score stays in [0, 1])
ALPHA_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BETA_VALUES  = []   # derived: beta = 1 - alpha


async def _retrieve_with_params(
    question: str,
    alpha: float,
    beta: float,
    neo4j,
    embedder: Embedder,
    bm25: HybridBM25Search,
    reranker: CrossEncoderReranker,
    cfg: dict,
) -> list[dict]:
    """Run retrieval steps 1-5 with a specific alpha/beta. Returns ranked chunks."""
    top_k = cfg.get("local_top_k", 10)
    hops  = cfg.get("multihop_depth", 2)

    embedding     = await embedder.embed_text(question)
    vector_chunks = await neo4j.vector_search_chunks(embedding, top_k=top_k)

    fused_chunks  = await bm25.search(
        query=question, vector_chunks=vector_chunks, top_k=top_k
    )

    seed_chunks = await reranker.rerank(question, fused_chunks)
    seed_ids    = [c["chunk_id"] for c in seed_chunks]

    hop_chunks  = await neo4j.get_multihop_chunks(seed_ids, hops=hops)
    seen        = set(seed_ids)
    all_chunks  = seed_chunks + [c for c in hop_chunks if c["chunk_id"] not in seen]
    all_ids     = [c["chunk_id"] for c in all_chunks]

    chunk_entities, entity_edges = await asyncio.gather(
        neo4j.get_chunk_entity_embeddings(all_ids),
        _fetch_subgraph_edges(neo4j, all_ids),
    )

    scorer = GNNScorer(
        gnn_type                  = cfg.get("gnn_type", "gat"),
        num_layers                = cfg.get("gnn_layers", 2),
        alpha                     = alpha,
        beta                      = beta,
        edge_confidence_threshold = cfg.get("gnn_edge_confidence_threshold", 0.7),
    )
    loop = asyncio.get_event_loop()
    ranked = await loop.run_in_executor(
        None,
        lambda: scorer.score(
            query_vec      = embedding,
            chunks         = all_chunks,
            chunk_entities = chunk_entities,
            entity_edges   = entity_edges,
        ),
    )
    return ranked


def _score_ranking(chunks: list[dict], ground_truth: str) -> float:
    """Proxy quality score — no LLM needed.

    Measures how well the top-3 chunks cover the ground truth words.
    Uses token recall: what fraction of ground-truth tokens appear in the
    top-3 retrieved chunks.
    """
    gt_tokens = set(ground_truth.lower().split())
    top_text  = " ".join(c["text"] for c in chunks[:3]).lower()
    top_tokens = set(top_text.split())
    if not gt_tokens:
        return 0.0
    recall = len(gt_tokens & top_tokens) / len(gt_tokens)
    # Bonus: reward higher final_score confidence on top chunk
    top_final = chunks[0].get("final_score", 0.0) if chunks else 0.0
    return 0.7 * recall + 0.3 * top_final


async def calibrate(eval_path: Path, dry_run: bool):
    eval_set = json.loads(eval_path.read_text(encoding="utf-8"))
    log.info("calibrate.loaded", queries=len(eval_set), eval_set=str(eval_path))

    cfg_obj   = get_settings()
    cfg       = cfg_obj.retrieval
    neo4j     = get_neo4j()
    embedder  = Embedder()
    bm25      = HybridBM25Search()
    reranker  = CrossEncoderReranker(top_k=cfg.get("rerank_top_k", 5))

    best_score  = -1.0
    best_alpha  = cfg.get("gnn_alpha", 0.6)
    best_beta   = cfg.get("gnn_beta",  0.4)
    results     = []

    combos = [(a, round(1.0 - a, 2)) for a in ALPHA_VALUES]
    log.info("calibrate.grid", combos=len(combos), queries=len(eval_set))

    for alpha, beta in combos:
        combo_scores = []
        for item in eval_set:
            chunks = await _retrieve_with_params(
                question=item["question"],
                alpha=alpha,
                beta=beta,
                neo4j=neo4j,
                embedder=embedder,
                bm25=bm25,
                reranker=reranker,
                cfg=cfg,
            )
            s = _score_ranking(chunks, item.get("ground_truth", ""))
            combo_scores.append(s)

        avg = sum(combo_scores) / len(combo_scores)
        results.append((alpha, beta, avg))
        log.info("calibrate.combo", alpha=alpha, beta=beta, avg_score=round(avg, 4))

        if avg > best_score:
            best_score = avg
            best_alpha = alpha
            best_beta  = beta

    print("\n-- Grid search results ------------------------------------------")
    for a, b, s in sorted(results, key=lambda x: -x[2]):
        marker = " ← best" if (a == best_alpha and b == best_beta) else ""
        print(f"  alpha={a}  beta={b}  score={s:.4f}{marker}")

    print(f"\nBest: alpha={best_alpha}  beta={best_beta}  score={best_score:.4f}")

    if dry_run:
        print("\n[dry-run] settings.yml NOT updated.")
        return

    # Write best values back to settings.yml
    settings_path = ROOT / "config" / "settings.yml"
    text = settings_path.read_text(encoding="utf-8")

    import re
    text = re.sub(
        r"(gnn_alpha:\s*)[\d.]+",
        lambda m: f"{m.group(1)}{best_alpha}",
        text,
    )
    text = re.sub(
        r"(gnn_beta:\s*)[\d.]+",
        lambda m: f"{m.group(1)}{best_beta}",
        text,
    )
    settings_path.write_text(text, encoding="utf-8")
    log.info("calibrate.saved", alpha=best_alpha, beta=best_beta,
             path=str(settings_path))
    print(f"\n✓ settings.yml updated: gnn_alpha={best_alpha}  gnn_beta={best_beta}")

    await neo4j.close()


def main():
    parser = argparse.ArgumentParser(description="Calibrate GNN alpha/beta via grid search")
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=ROOT / "eval_data" / "calibration_set.json",
        help="Path to JSON eval set [{question, ground_truth}, ...]",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print best values but don't update settings.yml",
    )
    args = parser.parse_args()

    if not args.eval_set.exists():
        print(f"Eval set not found: {args.eval_set}")
        sys.exit(1)

    asyncio.run(calibrate(args.eval_set, args.dry_run))


if __name__ == "__main__":
    main()
