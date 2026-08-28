# Architecture gap analysis

## Fixed depth=2 traversal

CURRENT ARCHITECTURE: RRF → cross-encoder → depth-2 traversal.  
LIMITATION: one hop budget cannot fit entity lookup, relational questions, and global questions.  
EXTERNAL TECHNIQUE: Microsoft local/global/DRIFT routing; HippoRAG Personalized PageRank; FRAG query-conditioned hop estimation.  
PROPOSED MODIFICATION: `TraversalPolicy(max_depth=4, beam_width, stop_when, query_mode)`; stop when marginal evidence gain falls below threshold or confidence saturates.  
EXPECTED BENEFIT: higher recall on multi-hop queries and less neighborhood noise on simple queries.  
COST: medium.  
RISKS: latency and graph-query explosion; enforce budgets and cache expansions.

## Static 90/10 fusion

CURRENT ARCHITECTURE: fixed text/graph weighting.  
LIMITATION: text dominates even when graph structure is the evidence.  
EXTERNAL TECHNIQUE: query-mode routing, community ranking, path confidence, PPR.  
PROPOSED MODIFICATION: learn/calibrate `alpha(query), beta(query), gamma(path_confidence)`; preserve a safe prior and monotonic calibration.  
EXPECTED BENEFIT: better ranking across entity, relationship, global, and temporal queries.  
COST: medium.  
RISKS: leakage and overfitting; split by tenant/time and require calibration tests.

## Cross-encoder before graph expansion

CURRENT ARCHITECTURE: rerank text, then traverse.  
LIMITATION: a highly relevant chunk may anchor the wrong entity neighborhood.  
EXTERNAL TECHNIQUE: joint entity/community/path reranking.  
PROPOSED MODIFICATION: retrieve broad candidates → expand bounded subgraphs → score text + relation/path + provenance → final cross-encoder on evidence bundles.  
EXPECTED BENEFIT: better evidence coherence.  
COST: medium-high.  
RISKS: latency; keep baseline path as fallback.

## Entity-first retrieval

CURRENT ARCHITECTURE: entity linking is early and central.  
LIMITATION: global and underspecified queries may have no reliable entity anchor.  
EXTERNAL TECHNIQUE: Microsoft basic/global/DRIFT search.  
PROPOSED MODIFICATION: query router chooses entity-local, community-global, vector-basic, or hybrid mode.  
EXPECTED BENEFIT: fewer false anchors and better corpus-level answers.  
COST: medium.  
RISKS: routing errors; evaluate per query class.

## Static knowledge graph

CURRENT ARCHITECTURE: graph + bitemporal/provenance features exist.  
LIMITATION: user state, decisions, interactions, and evolving relationships are not uniformly modeled as episodes.  
EXTERNAL TECHNIQUE: context/memory graph ideas in the group; HippoRAG memory retrieval.  
PROPOSED MODIFICATION: typed event/episode layer with validity, observation time, actor, source, confidence, and supersession.  
EXPECTED BENEFIT: context-aware and auditable answers.  
COST: medium-high.  
RISKS: privacy, retention, stale state; enforce tenant/policy filters.

