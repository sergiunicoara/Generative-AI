# Implementation roadmap

## P0 — Query router and bounded adaptive traversal

Problem: fixed depth=2 and mandatory entity-first routing.  
Technique: basic/local/global/temporal/DRIFT-like routing with bounded beam traversal.  
Architecture change: add policy object between retrieval fusion and graph expansion.  
Files/modules affected: `graphrag/retrieval/adaptive_router.py`, `query_planner.py`, `local_search.py`, `global_search.py`, `hybrid_retriever.py`; tests in `tests/unit/test_adaptive_router.py`.  
Difficulty: medium. Expected impact: recall and context precision on multi-hop/global slices.  
Benchmark: baseline versus adaptive policy.  
Acceptance: no regression >2% on entity-local nDCG; ≥5% recall gain on multi-hop; p95 within budget.  
Rollback: feature flag to fixed baseline.

## P0 — Sufficiency and abstention gate

Problem: a fluent answer may be generated despite weak evidence.  
Technique: evidence coverage, contradiction risk, unresolved-entity count, path verification.  
Affected: `answer_policy.py`, `answer_grounding.py`, `claim_verifier.py`, `fallback_policy.py`.  
Difficulty: medium.  
Acceptance: lower unsupported-claim rate on adversarial golden cases without unacceptable abstention.  
Rollback: threshold/config rollback.

## P1 — Calibrated fusion

Problem: fixed 90/10 weighting.  
Technique: logistic/isotonic calibration or small learning-to-rank model with query features.  
Affected: `reranker.py`, `graph/gnn_scorer.py`, `evaluation/graphrag_benchmark.py`.  
Acceptance: statistically significant MRR/nDCG improvement by query class; calibration error reported.  
Rollback: retain fixed weights.

## P1 — Context Graph episode model

Problem: state and decisions are not consistently first-class temporal objects.  
Technique: episode/event/decision nodes with source, actor, valid-time, observed-time, supersession, and policy.  
Affected: `graph/bitemporal.py`, `graph/audit_trail.py`, `retrieval/session_context.py`, `enterprise/access.py`.  
Acceptance: temporal questions retrieve correct state at time T; tenant isolation and retention tests pass.  
Rollback: read-only projection behind flag.

## P2 — Community hierarchy

Problem: poor global retrieval over large corpora.  
Technique: Leiden communities, hierarchical reports, global map-reduce; borrow Microsoft GraphRAG outputs.  
Affected: `graph/community_builder.py`, `community_summarizer.py`, `retrieval/global_search.py`.  
Acceptance: global-query recall/faithfulness gain with bounded token cost.  
Rollback: basic/vector route remains available.

## P3 — PPR / LightRAG / graph-transformer experiments

Run isolated ablations. Do not make them production defaults before cost, freshness, and calibration evidence exists.

