# Sales Context Graph

CRM data (Salesforce-shaped) + sales-call transcripts (Gong-shaped) → Neo4j
knowledge graph → query-specific context subgraphs for sales teams.

## Status

Scaffolding only. Nothing runs yet. See `docs/plan.md` — it contains:

- a defect review of the original spec (Parts 1–2)
- research grounding on Showpad/Salesforce/Gong data shapes (Part 3)
- the phase plan, P0 through P4.5 (Part 4)
- reuse-vs-greenfield assessment against `ai-knowledge-graph-platform` (Part 5)
- scalability and operational gaps (Part 6)
- **8 open decisions that must be settled before implementation starts** (Part 7)
- the Codex implementation prompt (Part 8)

**Do not hand Part 8 to a code generator until Part 7 is resolved.** Items
2, 3, and 5 in particular (CRM choice, transcript source, `workspace_id` vs.
Showpad `Division`) will otherwise be answered by invention.

## What's already here

`src/graph/` — eight modules ported from `../ai-knowledge-graph-platform`
(a sibling repo, different domain — compliance/regulatory GraphRAG). Each
carries a one-line attribution comment. These are the pieces judged reusable
after inventorying that codebase directly, not assumed reusable:

| Module | What it gives you |
|---|---|
| `alias_registry.py` | Multi-stage entity resolution: exact → normalized → fuzzy (rapidfuzz) → embedding-cosine. Has a `TODO` marking the aerospace-specific regulatory-prefix block for removal. |
| `review_queue.py` | The human-review path for ambiguous resolutions — this is most of `/unresolved-mentions/{id}/resolve` already built. |
| `reification.py` | Reified-triple pattern — this is the Claim layer, under a different name. |
| `bitemporal.py` | `valid_from/to` + `transaction_from/to` queries (as-of, history, diff). |
| `contradiction_detector.py` + `contradiction_strategies.py` | Conflict detection as first-class `:Conflict` nodes, pluggable per-conflict-class strategies. |
| `ontology_registry.py` + `domain_ontology.py` | Config-driven domain model — `config/ontologies/sales.yml.template` (copied from `marketing_adtech.yml`, closest existing ontology in shape) is the starting point for a real `sales.yml`. |

**Not ported, and not to be reused as-is** (see `docs/plan.md` Part 5):
extraction (`extractor.py` in the source repo swallows `JSONDecodeError`
with no retry — the new spec's typed/retrying/fake-extractor design is a
real improvement, build it fresh), and the tenancy pattern (a `WHERE`
predicate repeated per-query, audited post-hoc — P1 here mandates a single
`GraphSession` choke point instead).

`docker-compose.yml` — Neo4j + API only. No queue, no cache, no KPI store.
Add those when there's a measured reason (see the comment in the file), not
speculatively — this mirrors the source repo's own "when to add X" roadmap
convention.

## Next step

Resolve `docs/plan.md` Part 7, then run Part 8 through Codex (or implement
P0 directly — it's small: Pydantic models + deterministic ID functions,
no DB, no LLM).
