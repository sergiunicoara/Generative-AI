# Research-derived roadmap

The full product roadmap remains in `docs/roadmap.md`. This concise view records
only decisions derived from the 2026-08-26 LinkedIn Knowledge Graph group
research; it avoids duplicating capabilities already delivered.

| Priority | Item | Exit criterion | Status |
|---|---|---|---|
| P0 | Fix retrieval-query interpolation and metadata-schema tenant denial | Valid-time, transaction-time, tenant-edge, hop-depth and semantic-score query guards pass | Implemented |
| P1 | Capture standard and agentic retrieval trajectories | Every enabled retrieval result exposes bounded route/evidence steps without changing ranking | Implemented |
| P1 | Score route, evidence, graph-edge recall, answer quality, and efficiency | API and golden runner produce deterministic scores; legacy cases are unchanged | Implemented |
| P2 | Add structural expectations to representative single-hop, multi-hop, contradiction, and negative golden cases | Expert-confirmed expected surfaces/evidence/edges and stable release threshold | Planned |
| P3 | Prototype epistemic stance on claims | A real dataset proves that stance changes correctness beyond existing confidence/contradiction provenance | Conditional |
| P3 | Benchmark a low-memory Bolt backend | Independent parity on tenancy, temporal semantics, backup/restore, recovery, and latency/cost | Conditional |
| Research | Learned graph routing / structural reward optimization | Sufficient expert-labelled trajectories; offline policy beats deterministic routing with no safety regression | Deferred |

Detailed source analysis and corpus limitations: `docs/archive/research/linkedin-group-research.md`.
Machine-readable decisions: `research/linkedin_findings.json`.
