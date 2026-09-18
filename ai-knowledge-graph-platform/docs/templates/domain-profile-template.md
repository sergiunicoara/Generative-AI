# Domain Profile Template

Roadmap "P1 — reusable domain onboarding and governance pack", bullet 2. One
of these per domain, filled in *before* writing
`config/ontologies/<tenant>_*.yml` — it's the policy decisions that YAML
then encodes, kept as a reviewable document rather than something a reader
has to reverse-engineer from the schema. For each field, this template
states whether the platform already enforces it (cite the code) or whether
it's a stated policy this domain commits to that nothing currently checks —
be honest about which is which; don't imply enforcement that isn't there.

- Domain / tenant id: `<e.g. insurance-underwriting — becomes ontology.id>`

## Identifiers

- Entity/relation naming: **enforced** — `UPPER_SNAKE_CASE`, checked by
  `graphrag/graph/domain_ontology.py:validate_ontology_document` and
  `graphrag/graph/ontology_registry.py`'s `_RELATION_RE` at write time.
- Business-key convention for this domain: `<e.g. policy number format>` —
  policy statement only; the platform's `assetId`-style key fields
  (`ontology/models/*.yaml`'s `PropertySpec.key`) are per-model, not
  domain-wide.

## Naming policy

- Type/relation vocabulary source: link to this domain's
  `domain-onboarding-business-glossary-template.md`.
- Deprecation naming (`_deprecated`, versioned suffix, etc.), if any beyond
  the platform's `deprecated_types`/`migration_map` mechanism: `<policy>`.

## Provenance

- Source systems this domain's data originates from: `<list>`.
- Provenance the platform already carries end to end for RDF-track domains
  (Energy): `prov:wasDerivedFrom`, ingestion activity, software agent —
  see `graphrag/provenance/prov_o.py`. **State explicitly** whether this
  domain uses that RDF/PROV-O track or the LPG/`domain_ontology.py` track
  (aerospace/automotive/marketing/pharma today) — the latter has provenance
  as `SystemRepresentation`/`ContextualAssertion` nodes
  (`graph_writer.py`, `neo4j_client.py`), not PROV-O RDF.

## Retention

- Retention period for this domain's data: `<policy — not currently a
  platform-enforced field; document it here>`.
- GDPR erasure applicability: `graphrag/graph/gdpr.py` provides cascade
  erasure; state whether this domain's data falls under it.

## Access

- Who can read this domain's data (tenant scoping is enforced by
  `(name, type, tenant)` identity + `ToolPolicy`; this is about roles
  *within* the tenant): `<policy>`.
- Any field-level access restriction beyond tenant scoping: `<policy — not
  currently platform-enforced; document it here if it exists>`.

## Temporal semantics

- Does this domain need valid-time/transaction-time (bitemporal) tracking?
  `<yes/no>` — if yes, the platform mechanism is
  `graphrag/graph/bitemporal.py`; if the domain is LPG-track only, state how
  "as of" queries are expected to work (or that they aren't supported yet).

## Validation and publication rules

- Which SHACL shapes (if RDF-track) or `relation_rules`/domain-range checks
  (if LPG-track, `graphrag/graph/domain_ontology.py`) gate a write.
- Publication gate, if this domain has a staged-candidate → validated
  → published pipeline like Energy's (`graphrag/domains/energy/
  publication.py`): `<describe, or "not applicable — writes go straight to
  the graph">`. Be explicit rather than silent if there is no gate — a
  reader should not have to guess whether one exists.
