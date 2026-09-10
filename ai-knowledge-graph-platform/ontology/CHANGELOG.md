# Ontology changelog

Tracks version changes to the per-tenant domain ontologies in
`config/ontologies/` and the SHACL shapes in `ontology/shapes/`. See
[`README.md`](README.md) for the process a change should follow.

Format: one entry per released version bump, newest first. An entry records
what changed and why — not a copy of the diff, which `git log` already gives
you.

## Exported vocabulary `1.0.0` — 2026-09-10

The RDF export's own `base:`/`annot:` vocabulary now declares a version. It is
deliberately separate from the per-tenant domain ontologies below: those version
*what the graph is about*, this versions *the interchange terms the export
mints*. Tracked as `ONTOLOGY_VERSION` in `scripts/export_rdf.py`; bump it here
and there together whenever a `base:` class/property changes meaning.

Emitted as `owl:versionIRI` + `owl:versionInfo` on the ontology resource, with
`rdfs:isDefinedBy` on every minted `base:` term — previously those terms were
generated per export with no way for a consumer to resolve where they were
defined or which release defined them.

Also in this entry, `1.0.0` being the first version rather than a change to an
existing one:

- Entity aliases are published as `skos:altLabel` (they were resolved
  internally by `graphrag/graph/alias_registry.py` but never exported, while
  `cross_ontology_linker.py` consumed that same predicate from *external*
  ontologies).
- Chunks carry `prov:specializationOf` to their source document alongside
  `prov:wasDerivedFrom`, matching PROV-DM's distinction between "derived from"
  and "same thing, finer granularity".
- SHACL validation moved ahead of serialisation, with `--strict` refusing to
  write a non-conformant export at all. This closed a real defect: artifact
  extraction activities were read from the wrong result key
  (`source_chunk_id` vs the projected `source_chunk_ids`), so every one of
  them was exported with no `prov:used` — a violation of this project's own
  shapes that post-hoc, opt-in validation never surfaced.

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
`graphrag/graph/shacl_validator.py`'s module docstring for the implementation
detail.

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
