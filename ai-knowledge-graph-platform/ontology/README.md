# Ontology governance

## Canonical Energy semantic model

The compiler also emits `generated/energy/diagnostics.json`, a read-only,
machine-readable capability-loss report, and `generated/energy/diagnostics.md`,
the same report rendered as a human-readable Markdown table — one row per
canonical rule × target, its fidelity (`preserved`/`approximated`/
`unenforceable`), the runtime control that compensates, and its source-model
location. Both are generated during compilation and must not be edited from
the dashboard or by hand. RDF/SHACL remains the authoritative governed
semantic layer; Neo4j/LPG is a rebuildable projection with different native
constraint capabilities. `tests/unit/test_semantic_model.py::
test_no_rule_disappears_silently_across_targets` cross-checks a synthetic
model exercising every tracked rule category (abstractness, mixins, plain
inheritance, key/required/datatype/cardinality properties, relation endpoints
and cardinality, closed-world validation) against every target lacking that
capability, so a future compiler change can't silently stop reporting one.

The Energy domain now has one storage-independent, reviewable source of truth:
[`models/energy-asset-intelligence.yaml`](models/energy-asset-intelligence.yaml).
It declares classes, single inheritance, reusable mixins, properties,
relationships, datatypes, keys, cardinalities, LPG labels, and lifecycle
metadata. The following files are generated projections and must not be edited
directly:

- `energy/energy-asset-intelligence.ttl` — RDFS/OWL semantics;
- `shapes/energy-asset-intelligence.shapes.ttl` — closed-world SHACL checks;
- `generated/energy/neo4j-constraints.cypher` — Neo4j-enforceable keys;
- `generated/energy/diagnostics.json` — explicit target capability losses and
  their runtime controls (machine-readable);
- `generated/energy/diagnostics.md` — the same capability losses, rendered as
  a human-readable table;
- `generated/energy/erd.mmd` — a reviewable ERD/frame view (entities,
  properties, inheritance, mixins, typed relations, cardinality) rendered as
  a Mermaid `classDiagram`, viewable natively in GitHub/GitLab Markdown with
  no separate tool.

Regenerate them with:

```text
python -m graphrag.semantic_model compile ontology/models/energy-asset-intelligence.yaml
```

CI uses the same command with `--check`, so a pull request fails if committed
artifacts drift from the YAML. Generation is deterministic and each output is
atomically replaced only after the complete model validates. R2RML and RML
files remain separate source-to-model mapping contracts; they consume this
vocabulary rather than competing with it.

The normal compiler succeeds when a target has a known limitation, but prints
the corresponding diagnostic and writes it to JSON and Markdown. Deployments
that permit no runtime-only rules can add `--fail-on-unenforceable`; this
fails before writing any artifact. `--fail-on-error` is the equally strict,
independent gate on `severity` rather than `fidelity` — it fails before
writing if any diagnostic is `error`-severity, while still allowing reviewed
`warning`s and `info`-level downgrades through; no diagnostic is `error`-
severity today, so this gate is currently dormant, not currently enforced.
The migration is parity-gated by `tests/unit/test_semantic_model.py`,
which parses both generated Turtle targets, exercises the real SHACL validator,
checks established Energy vocabulary, verifies deterministic output and drift,
and proves shared-write rejection. Existing Energy publication tests continue
to exercise R2RML/RML → candidate RDF → SHACL → quarantine/publication → SPARQL.

This directory is the SHACL-shapes half of the platform's ontology system —
see [`docs/ontology-model.md`](../docs/ontology-model.md) for the technical
model (type hierarchy, relation domain/range rules, inference rules,
bitemporal/confidence semantics, design decisions). This document is the
process half: naming conventions, lifecycle, and how to propose, review, and
release a change without breaking a live tenant.

## What lives where

| Location | Contains | Enforced by |
|---|---|---|
| `ontology/models/*.yaml` | Canonical storage-independent domain semantics compiled to RDF and LPG targets | `graphrag/semantic_model/`; drift-gated by `make semantic-model-check` |
| `config/ontologies/*.yml` | Per-tenant domain ontologies — entity type hierarchy, relation domain/range rules, inference rules, deprecation state | `graphrag/graph/domain_ontology.py`, `graphrag/graph/ontology_registry.py` |
| `ontology/shapes/*.ttl` | SHACL shapes that validate the platform's RDF representations (export + relational-ingestion mutation gate) | `graphrag/graph/shacl_validator.py` |
| `ontology/mappings/*.r2rml.ttl` | R2RML mappings from a relational source (SQLite/PostgreSQL/Excel) to this ontology's entity/relation shape, e.g. `supply-chain.r2rml.ttl` | `graphrag/ingestion/r2rml.py` (`r2rml_to_mapping`), driven by `scripts/ingest_r2rml.py`, supports the Neo4j-shaped ingestion path. The Energy path executes the mapping directly into RDF via `graphrag/ingestion/r2rml_rdf.py`'s `materialize_r2rml()` (RML sources: `graphrag/ingestion/rml_rdf.py`'s `materialize_rml()`), validates and publishes RDF first, then optionally projects that published graph one-way to Neo4j for GraphRAG. |

Three directories, not one, because they answer different questions: a domain
ontology YAML defines *what a tenant's graph is allowed to contain*
(`config/ontologies/`); a SHACL shape defines *what shape a specific RDF
representation of that graph must have* (`ontology/shapes/`); an R2RML
mapping defines *how to derive that graph from a relational table* that
was never authored as RDF in the first place (`ontology/mappings/`). The
domain ontology can be extended per-tenant; the SHACL shapes are shared
structural invariants (a label must exist, a confidence must be a valid
float) that don't vary by tenant; an R2RML mapping is source-specific — one
per relational schema, validated against the domain ontology it targets
before any row is ingested (`RelationalGraphIngestor.validate()`, called by
`scripts/ingest_r2rml.py --validate-only`).

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

## Onboarding a new domain

Roadmap "P1 — reusable domain onboarding and governance pack": before
writing a new `config/ontologies/<tenant>_*.yml` from scratch, fill in
these templates (`docs/templates/domain-onboarding-*.md`,
`docs/templates/domain-profile-template.md`) — they turn stakeholder
discovery into reviewable artifacts instead of decisions that only ever
existed in a meeting:

- `domain-onboarding-business-glossary-template.md` — the business
  vocabulary the ontology's type/relation names will be derived from.
- `domain-onboarding-sme-workshop-notes-template.md` — one file per SME
  session.
- `domain-onboarding-entity-relationship-decisions-template.md` — the
  append-only decision log explaining *why* the YAML looks the way it does.
- `domain-onboarding-unresolved-questions-template.md` — tracks an open
  question from raised to closed.
- `domain-onboarding-ownership-and-deprecation-template.md` — who owns the
  domain and this domain's deprecation SLA, on top of the platform's
  already-enforced `migration_map` requirement.
- `domain-profile-template.md` — identifiers, naming, provenance, retention,
  access, temporal semantics, validation and publication rules, each
  marked as platform-enforced (with the enforcing code cited) or
  policy-only (not currently checked by anything).

These are templates, not automation — nothing currently validates that a
domain filled them in. Roadmap bullets 3–4 of the same item (a full
regulated-domain example demonstrating this pack end to end, and the
curation-queue integration) remain open.

## Competency questions

A competency question is a question the ontology must be able to answer —
if extraction and retrieval can't produce a correct, cited answer to it, the
ontology (or its wiring into extraction/retrieval) is incomplete. These are
drawn directly from the live golden-eval suites (`evals/golden_set.json`,
`data/eval_golden/queries_*.json`), not invented for this document — they are
run, not aspirational, and their current pass/fail status is tracked in
`evals/last_run.json`.

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

It also detects duplicate relation definitions across two ontology files
(conflicting domain/range for the same relation name) and orphaned types
(declared but never used in any relation rule) — added in commit `46fa83f`,
surfaced as non-fatal warnings; see `tests/unit/test_domain_ontology_lint.py`.

### Competency-query migration gate

The "adding a new domain ontology should add ... competency questions" rule
above was, until now, an honor-system PR-description ask — nothing actually
ran those questions before a migration landed. `graphrag/graph/
competency_gate.py` makes it executable: `run_competency_suite()` runs a
pluggable set of `CompetencyQuestion`s and reports pass/fail, and
`OntologyRegistry.apply_ontology_migration()` accepts an optional
`competency_report` that blocks the migration (raising
`OntologyMigrationBlockedError`, no graph write attempted) when it fails,
unless explicitly `force`d. Only one concrete, fully-executable
implementation ships today —
`energy_maintenance_review_competency_questions()`, reusing `evals/
energy_demo/sparql/maintenance_review.rq` against the Energy tenant's RDF
graph — since it's the one competency question in this repo that needs no
LLM or live service to run deterministically.

The aerospace/automotive/marketing questions above are now wired too, as
real executable Cypher (`aerospace_regulatory_competency_questions()`,
`automotive_iatf_competency_questions()`,
`marketing_adtech_competency_questions()` — query text lives in
`evals/<domain>/cypher/*.cypher`, one file per question, mirroring the
Energy/SPARQL layout). Be precise about the evidence level: unlike the
Energy question, these have **not been run against a live Neo4j graph** —
there is no Neo4j instance in this development environment. What *is* true
and checked on every `make ci` (`competency-questions-check` /
`scripts/check_competency_questions.py`): every relation/type name each
question's Cypher references is verified to still exist in that domain's
current `config/ontologies/*.yml` — a static regression gate that catches a
renamed/removed type or relation immediately, without needing Neo4j, even
though it can't prove the query returns real rows. `run_competency_suite_async`
plus `neo4j_cypher_query_fn` (`graphrag/graph/competency_gate.py`) is the
real execution path once a live Neo4j instance is available; running it and
confirming a pass is the step that would move these from "unit-tested" to
"live-validated" in this doc's own evidence-level convention.

`graphrag/graph/ontology_migration.py`'s `find_affected_competency_questions()`
is the complementary "model-change impact analysis" direction: given a
`MigrationReport`'s added/removed/renamed names and a domain id, it reports
which of that domain's competency questions reference a name the migration
removes or renames away from — so a reviewer can see *which specific
question* a breaking change affects, not just that the class/property diff
is structurally unmapped. Honestly scoped: it only inspects competency-
question query text, not R2RML/RML mappings, SHACL shapes, generated Neo4j
constraints, or API contracts — a real but partial answer to "what does this
ontology change affect."

## SHACL shapes (`ontology/shapes/`)

See `graphrag/graph/shacl_validator.py`'s module docstring for the full
design (named shapes for stable failures-by-shape grouping, explicit
`sh:severity` distinguishing `sh:Violation` from `sh:Warning`, the
machine-readable `ShaclReport` API). In short: `export.shapes.ttl` validates
the RDF graph `scripts/export_rdf.py` produces; `ingestion.shapes.ttl` is a
live pre-write mutation gate for the relational-ingestion path
(`graphrag/ingestion/relational.py`) — a non-conformant relational mapping is
rejected before any Neo4j write, not merely logged.

The LLM-extraction path first rejects ontology-invalid entity types and
domain/range violations, then runs this same SHACL batch gate before any graph
write. Rejected facts are recorded as an `extraction_shacl_rejected` schema
event; they are not coerced into plausible graph facts.
