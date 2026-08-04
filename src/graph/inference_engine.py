# Forked from ai-knowledge-graph-platform (graphrag/graph/inference_engine.py) — only the
# InferenceRule dataclass is ported. The full ForwardChainingEngine (Datalog-style
# transitivity/symmetry/inverse/composition forward-chaining, ~450 lines) is NOT forked:
# nothing in docs/plan.md requires KG completion inference for this vertical slice, and
# domain_ontology.build_inference_rules_from_ontology() only needs this dataclass to
# construct InferenceRule instances from YAML — it never calls the engine itself.
# If a future phase needs forward-chaining, fork ForwardChainingEngine then, against a
# real test, rather than carrying unused machinery now.

"""InferenceRule — a single forward-chaining rule definition.

See ai-knowledge-graph-platform/graphrag/graph/inference_engine.py for the full
ForwardChainingEngine this dataclass was designed to feed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InferenceRule:
    """A single forward-chaining rule."""
    name:              str
    rule_type:         str          # "transitivity" | "symmetry" | "inverse" | "composition"
    relation:          str          # head relation (LHS of =>)
    derived_relation:  str = ""     # what relation to derive (defaults to same as `relation`)
    body_relation_2:   str = ""     # for composition: the second body relation
    max_depth:         int = 3      # transitivity only: max chain length
    confidence_decay:  float = 0.9  # per-hop confidence multiplier
