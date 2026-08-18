# Ontology changelog

Tracks version changes to the per-tenant domain ontologies in
`config/ontologies/` and the SHACL shapes in `ontology/shapes/`. See
[`README.md`](README.md) for the process a change should follow.

Format: one entry per released version bump, newest first. An entry records
what changed and why — not a copy of the diff, which `git log` already gives
you.

## Current baseline — 2026-08-18

All 7 shipped domain ontologies are at `1.0.0`, `status: active`, with empty
`deprecated_types`/`deprecated_relations` — no ontology has been through a
deprecation cycle yet. This is the first point this changelog exists; it
records the starting baseline rather than reconstructed history, per the
project's evidence-over-inference convention (see `tasks/lessons.md`, A154).

| Ontology | `ontology.id` | Version |
|---|---|---|
| `aerospace_regulatory.yml` | `aerospace-regulatory` | 1.0.0 |
| `automotive_iatf.yml` | `automotive-iatf` | 1.0.0 |
| `marketing_adtech.yml` | `marketing-adtech` | 1.0.0 |
| `pharma_commercial.yml` | `synthetic-pharma-commercial` | 1.0.0 |
| `sustainability_supply_chain.yml` | `synthetic-sustainability-supply-chain` | 1.0.0 |
| `synthetic_large.yml` | `synthetic-large` | 1.0.0 |
| `telecom_oss.yml` | `telecom-oss` | 1.0.0 |

### SHACL shapes — 2026-08-18

Moved from inline Python strings in `graphrag/graph/shacl_validator.py` to
version-controlled `ontology/shapes/export.shapes.ttl` and
`ontology/shapes/ingestion.shapes.ttl` (the inline strings remain as a
last-resort fallback, kept byte-identical). Shapes renamed from anonymous
`[]` blank nodes to stable IRIs so SHACL validation results can be
meaningfully grouped by shape. Added explicit `sh:severity` on every
constraint, distinguishing `sh:Violation` (fails validation) from
`sh:Warning` (visible, does not fail) — previously every constraint failed
validation uniformly regardless of how serious the issue actually was. Added
one grounded warning-tier constraint: an `owl:Axiom`'s confidence annotation
may legitimately be absent (older data, or an extraction path that never set
it), so its absence is now a warning rather than indistinguishable from a
genuinely malformed confidence value. See
`docs/context_graph_gap_plan.md` (Priority 1 — SHACL validation) for the full
rationale and `graphrag/graph/shacl_validator.py`'s module docstring for the
implementation detail.

## Template for future entries

```
## <ontology-id> <old-version> -> <new-version> — YYYY-MM-DD

**Changed:** what entity types / relations / inference rules were added,
removed, or redefined.

**Why:** the concrete gap or incorrect behavior this fixes — link a golden-
eval question (see README.md's Competency Questions) or a specific
production observation, not a hypothetical.

**Migration:** for a removal/rename, the `migration_map` entry added, and
whether `graphrag/graph/ontology_migration.py` was run against already-
ingested data (and if so, against which tenant, when).

**Verified:** which golden-eval questions were re-run and their result
before vs. after, or "none affected" with the reasoning why.
```
