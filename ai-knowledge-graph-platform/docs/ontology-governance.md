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
