# Ontology governance

This directory is the SHACL-shapes half of the platform's ontology system —
see [`docs/ontology-model.md`](../docs/ontology-model.md) for the technical
model (type hierarchy, relation domain/range rules, inference rules,
bitemporal/confidence semantics, design decisions). This document is the
process half: naming conventions, lifecycle, and how to propose, review, and
release a change without breaking a live tenant.

## What lives where

| Location | Contains | Enforced by |
|---|---|---|
| `config/ontologies/*.yml` | Per-tenant domain ontologies — entity type hierarchy, relation domain/range rules, inference rules, deprecation state | `graphrag/graph/domain_ontology.py`, `graphrag/graph/ontology_registry.py` |
| `ontology/shapes/*.ttl` | SHACL shapes that validate the platform's RDF representations (export + relational-ingestion mutation gate) | `graphrag/graph/shacl_validator.py` |

Two directories, not one, because they answer different questions: a domain
ontology YAML defines *what a tenant's graph is allowed to contain*
(`config/ontologies/`); a SHACL shape defines *what shape a specific RDF
representation of that graph must have* (`ontology/shapes/`). The domain
ontology can be extended per-tenant; the SHACL shapes are shared structural
invariants (a label must exist, a confidence must be a valid float) that
don't vary by tenant.

## Naming conventions and stable identifiers

Enforced by `graphrag/graph/domain_ontology.py:validate_ontology_document`,
not just convention:

- **Entity type names**: `UPPER_SNAKE_CASE` (e.g. `AIRWORTHINESS_DIRECTIVE`).
- **Relation names**: `UPPER_SNAKE_CASE`, matched against `_RELATION_RE` at
  write time (`graphrag/graph/ontology_registry.py`) — a malformed name is
  auto-corrected, not silently accepted.
- **`ontology.id`**: a stable slug for the tenant/domain (e.g.
  `aerospace-regulatory`), used as the file's own self-identifier.
- **`ontology.version`**: strict semver (`_SEMVER_RE`). Compatibility across
  versions is currently a **major-version equality check**
  (`domain_ontology.py:159-161`) — `1.x.y` is compatible with any other
  `1.x.y`, a `2.0.0` is not. `compatible_with` is a required field but is
  presence-checked only, not parsed against the actual version — a known,
  documented limitation, not a silent gap.

## Lifecycle

Every ontology file declares `ontology.status` — one of `draft`, `active`,
`deprecated` — checked at load time.

**Deprecating a type or relation never deletes it.** Add it to
`ontology.deprecated_types` / `ontology.deprecated_relations`; the validator
(`domain_ontology.py:92-97`) **requires** a matching entry in
`ontology.migration_map` for every deprecated name, so a deprecation without
a stated replacement fails validation rather than shipping silently. This is
enforced, not just documented practice — `tests/unit/test_ontology_lifecycle.py`
gates every shipped ontology file on it.

## Process: proposing, reviewing, migrating, releasing a change

This is a small-team process matching how the repo actually operates today —
not an invented governance board. Follow it in this order:

1. **Propose**: edit the tenant's `config/ontologies/{tenant}_*.yml` on a
   branch. Bump `ontology.version` per semver — a breaking change (removing
   a type/relation without a `migration_map` entry, or changing a relation's
   domain/range in a way that would reclassify existing edges) is a major
   bump; additive changes (new type, new relation, widened domain/range) are
   minor.
2. **Validate locally**: `pytest tests/unit/test_ontology_lifecycle.py
   tests/unit/test_domain_ontology.py -v`, or the broader
   `make test-shacl` target, before opening a PR. The lifecycle test globs
   `config/ontologies/*.yml` — every shipped file is gated automatically,
   there is no separate step to remember for a newly added ontology.
3. **Review**: the diff itself is the review artifact — a `git diff` on the
   YAML shows exactly what entity types, relations, and deprecations changed.
   For a non-additive change (removing/renaming), the PR description should
   state which existing golden-eval questions (see Competency Questions
   below) the change could affect, and whether they were re-run.
4. **Migrate**: if the change renames a relation, `ontology_registry.py`'s
   `_migration_map` (loaded from `settings.ontology.migration_map` and the
   file's own `migration_map`) rewrites the old name to the new one at
   extraction-validation time — existing graph data is **not** rewritten
   automatically; `graphrag/graph/ontology_migration.py` provides the tooling
   for that when a migration needs to touch already-ingested data.
5. **Release**: merge. `OntologyRegistry.load()` re-reads the file and
   upserts a new `OntologyVersion` node (per-tenant since F13) the next time
   the tenant's registry loads — there is no separate "publish" step.

## Competency questions

A competency question is a question the ontology must be able to answer —
if extraction and retrieval can't produce a correct, cited answer to it, the
ontology (or its wiring into extraction/retrieval) is incomplete. These are
drawn directly from the live golden-eval suites (`evals/golden_set.json`,
`data/eval_golden/queries_*.json`), not invented for this document — they are
run, not aspirational, and their current pass/fail status is tracked in
`docs/audit-2026-08-13.md` and `evals/last_run.json`.

**Aerospace** (`aerospace_regulatory.yml`):
- *Which directive supersedes AD-2022?* — exercises `SUPERSEDES` and the
  `supersedes_transitivity` inference rule.
- *Which regulation body mandated FAA-AD-2024-01-02?* — exercises
  `MANDATED_BY` and its `mandated_by_inverse` rule.
- *What aircraft type does FAA-AD-2024-01-02 apply to?* — exercises
  `APPLIES_TO` domain/range (`AIRWORTHINESS_DIRECTIVE` → `AIRCRAFT_TYPE`).

**Automotive** (`automotive_iatf.yml`):
- *Sub ce rată de neconformitate trebuie să se situeze un furnizor PlastiAuto?*
  ("Under what non-conformance rate must a PlastiAuto supplier stay?") —
  exercises supplier-classification relations and IATF 16949 threshold
  entities.
- *Ce consecințe apar pentru un furnizor PlastiAuto clasificat ca CRITIC?*
  ("What consequences apply to a supplier classified as CRITICAL?") —
  multi-hop, exercises the classification → consequence chain.

**Marketing** (`marketing_adtech.yml`):
- *Which ad categories are strictly excluded from Nova Beverages Global
  campaigns?* — exercises `CATEGORY_EXCLUDED` and negative-knowledge
  modeling.
- *What consent requirement applies to behavioral/interest-based targeting
  in the EU?* — multi-hop, exercises jurisdiction-scoped policy relations.

Adding a new domain ontology should add at least one single-hop and one
multi-hop competency question to its tenant's golden set — an ontology with
zero questions exercising it has no evidence it actually supports retrieval,
only that it parses.

## Automated checks

`graphrag/graph/domain_ontology.py:validate_ontology_document` runs on every
load and detects:

- semver format violations;
- an invalid `status` value;
- a deprecated name with no `migration_map` entry;
- a cycle in the type hierarchy;
- a malformed (non-`UPPER_SNAKE_CASE`) relation name;
- an inference rule with a missing required field;
- a major-version incompatibility against the previous loaded version.

It does **not** currently detect duplicate relation definitions across two
ontology files, or an orphaned type (declared but never used in any relation
rule) — both are real gaps, not silently claimed as covered here.

## SHACL shapes (`ontology/shapes/`)

See `graphrag/graph/shacl_validator.py`'s module docstring for the full
design (named shapes for stable failures-by-shape grouping, explicit
`sh:severity` distinguishing `sh:Violation` from `sh:Warning`, the
machine-readable `ShaclReport` API). In short: `export.shapes.ttl` validates
the RDF graph `scripts/export_rdf.py` produces; `ingestion.shapes.ttl` is a
live pre-write mutation gate for the relational-ingestion path
(`graphrag/ingestion/relational.py`) — a non-conformant relational mapping is
rejected before any Neo4j write, not merely logged.

**Known gap, not covered by SHACL today**: the LLM-extraction ingestion path
(`graphrag/ingestion/extractor.py` → `ontology_registry.validate_extraction`)
coerces ontology violations (unknown type → `CONCEPT`, invalid relation →
`RELATED_TO`) rather than rejecting or quarantining them. SHACL only gates
the relational-import path today. Extending it to the LLM path is tracked
separately (see `docs/context_graph_gap_plan.md`, finding F5) — it would be a
new mutation gate on the main ingestion pipeline and needs its own
live-verified iteration, not a documentation-only fix.
