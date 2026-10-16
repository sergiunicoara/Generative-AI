# Ontology Governance Loop

The active ontology is a controlled contract, not a by-product of LLM
extraction. Unknown entity types, unknown relation predicates, and invalid
domain/range pairs are rejected before graph persistence.

Rejected candidates create a tenant-scoped `OntologyProposal` only after their
source chunk has been stored. Each proposal retains links to the source
`Document`, `Chunk`, and active `OntologyVersion`, and aggregates repeat
evidence through a stable fingerprint and `seen_count`.

## Review flow

1. Ingestion validates extracted entities and relations against the active
   ontology and SHACL gate.
2. Unapproved schema candidates are excluded from active entities/edges.
3. `GET /kg/ontology/proposals` lists pending proposals for the authenticated
   tenant.
4. A reviewer calls `POST /kg/ontology/proposals/{id}/approve` or `/reject`.
5. Approval records a governance decision only. A maintainer still changes the
   versioned YAML ontology or applies the existing `/kg/ontology/migration`
   workflow before the vocabulary becomes active.

This separation prevents a reviewer click from silently changing production
schema rules without a versioned migration, while still preserving the evidence
needed to make that decision.

## Entity identity

`Entity.id` is extraction-local and links relations returned in the same LLM
response. It is not a durable graph identifier. The canonical entity identity
is the tenant-scoped `(canonical_name, canonical_type)` natural key resolved by
the alias registry. The former ambiguous `Entity.canonical_id` field has been
removed.

## Alias matching scale

Fuzzy alias comparison is still performed by RapidFuzz with the existing
thresholds. Before comparison, candidates are indexed by normalized string
length and filtered only when their theoretical maximum score cannot reach the
review threshold. This preserves automatic matches and ambiguous-review
behaviour while avoiding a scan of every alias for obviously incompatible
lengths.

## Derived artifacts and keeping them in sync

The change process itself (propose, validate, review, migrate, release) is in `ontology/README.md`. These artifacts
are derived from the domain YAML files, so a YAML change has to be followed by regenerating them.

| Artifact | Made by | Where | Gate |
| --- | --- | --- | --- |
| OWL classes and properties, SHACL shapes (all domains except automotive, energy and `synthetic_large`, which has no compiled model) | `python scripts/export_ontology_owl.py --all ontology/from_config` | `ontology/from_config/<id>.owl.ttl`, `<id>.shapes.ttl` | `tests/unit/test_ontology_owl_export.py` fails when a committed file differs from what the generator produces now |
| SKOS concept scheme (hierarchy, definitions, synonyms) | `python -m graphrag.graph.skos_export <yaml> [out.ttl]` | generated on demand, not committed | `tests/unit/test_skos_export.py` |
| Competency questions with asserted result sets | hand-written | `evals/competency/cases.json` | `tests/unit/test_sparql_competency_questions.py` |

What the generators emit and what they leave out:

* OWL: classes, `rdfs:subClassOf`, object properties with domain and range (a union when a rule lists several types),
  `owl:inverseOf`, `owl:FunctionalProperty`, and `owl:TransitiveProperty` for `transitivity` inference rules.
  `exclusive_state_pairs` and the other inference rule types have no faithful OWL form and are not emitted.
* SHACL: the subject of a relation must be an instance of a domain type, every object an instance of a target type,
  and a functional relation has at most one value. Shapes are not generated for automotive or energy; those keep
  their hand-written files in `ontology/shapes/`.
* SKOS: equivalence links to another scheme come from `add_label_mappings` (exact label match gives `exactMatch`,
  an alternative-label match gives `closeMatch`, nothing is guessed from similarity) or from mappings passed in
  explicitly. Review them before use: a concept can match several external concepts.
* `vocabulary` sections (definitions and synonyms) exist for automotive and aerospace. The definitions restate the
  notes already in each YAML and were not reviewed by a domain expert. Other domains have none yet.

Reasoning and validation are also reachable over the API: `POST /kg/validate` (SHACL, against a named shapes file from
`ontology/shapes`) and `POST /kg/reason` (OWL-RL on an in-memory copy; reports contradictions including disjointness
violations). Both read the tenant's RDF export and write nothing.
