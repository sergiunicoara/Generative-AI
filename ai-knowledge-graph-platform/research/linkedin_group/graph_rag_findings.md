# GraphRAG findings

## Post

Author: David Geddes  
Date: 1w  
URL: not exposed in the normal UI  
### Core idea

Knowledge representation and an operator model make an agent adaptive and interactive.

### Technical mechanism

Represent operator goals, preferences, constraints, and context as first-class graph entities.

### Why it matters

This supports a Context Graph rather than a static fact graph.

### Difference from our architecture

The current pipeline has session context, but operator state should be explicit, versioned, permissioned, and retrievable.

### Evidence

LinkedIn practitioner claim; evidence strength 1.

### Implementation opportunity

Add `OperatorState`, `Goal`, `Preference`, `Decision`, and `Episode` nodes with validity intervals and provenance. Retrieve operator state only when policy allows.

### Priority score

59/100.

## Post

Author: David Geddes  
Date: 1mo  
### Core idea

MBSE/executable knowledge structures and real-time adaptation.

### Technical mechanism

Separate declarative domain concepts from executable workflows and system constraints.

### Difference from our architecture

The existing graph reasoning and engineering workflows can be joined by typed “procedure enables action” edges with preconditions and postconditions.

### Evidence

LinkedIn claim plus linked Defense News article; evidence strength 1 for the architecture inference.

### Implementation opportunity

Prototype a read-only action planner over validated graph paths; keep execution behind existing tool policy and human approval gates.

### Priority score

65/100.

## Post

Author: David Geddes  
Date: 3mo  
### Core idea

“Script graphs” connect graph state to code execution.

### Difference from our architecture

The repo already has workflow orchestration; the missing boundary is a typed, auditable graph-to-workflow contract.

### Evidence

LinkedIn claim; evidence strength 1.

### Implementation opportunity

Represent `Trigger`, `Precondition`, `Action`, `Postcondition`, and `Rollback` nodes. Never execute an LLM-generated arbitrary Cypher/action; compile only allowlisted operations.

### Priority score

70/100.

## Post

Author: David Geddes  
Date: 4mo  
### Core idea

Use probabilistic systems for decision support and deterministic graph-backed systems for high-risk actions.

### Difference from our architecture

This validates the current hybrid direction but recommends explicit risk routing instead of a single answer path.

### Evidence

Plausible practitioner opinion, evidence strength 2.

### Implementation opportunity

Route by query risk and answer policy: probabilistic retrieval for exploration; verified evidence paths plus abstention for consequential answers.

### Priority score

75/100.

## Post

Author: David Geddes  
Date: 1yr  
### Core idea

Prefer semi-autonomous, human-centered agents over claims of full autonomy.

### Difference from our architecture

Strengthens the existing fallback, policy, and grounding layers; do not remove human approval.

### Evidence

LinkedIn claim; evidence strength 2.

### Implementation opportunity

Expose evidence coverage, uncertainty, and proposed actions separately; require confirmation for state-changing tools.

### Priority score

72/100.

## Verified external techniques

Microsoft GraphRAG documents global, local, DRIFT, and basic search, Leiden community hierarchy, community reports, and source-text/entity fusion. These are directly relevant to the current entity-first, depth-2 design.

LightRAG and HippoRAG are experiments, not drop-in replacements: LightRAG motivates dual-level retrieval and incremental updates; HippoRAG motivates Personalized PageRank over extracted graph memory.

New technique: risk-conditioned deterministic boundary. It is not a new retrieval algorithm, but it is a useful architectural control plane suggested by the group’s repeated “probabilistic vs deterministic” distinction.

