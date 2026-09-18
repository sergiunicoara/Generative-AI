# Ownership and Deprecation Template

Roadmap "P1 — reusable domain onboarding and governance pack". One of these
per domain (`config/ontologies/<tenant>_*.yml`), answering "who owns this
and what happens when something in it needs to go away" — the two questions
that otherwise get answered ad hoc, once, under time pressure, the first
time someone actually needs to deprecate a type.

## Ownership

- Domain / tenant: `<e.g. insurance-underwriting>`
- Owning team: `<team>`
- Primary contact: `<name/role>`
- Escalation path: `<who decides a disputed modeling change>`
- Review cadence: `<e.g. quarterly, or "on every SME workshop">`

## Deprecation policy

The mechanics are already enforced, not just documented — this section is
where a domain states *how it will use* that enforcement, not new tooling:

- `graphrag/graph/domain_ontology.py:validate_ontology_document` requires a
  `migration_map` entry for every name in `ontology.deprecated_types` /
  `ontology.deprecated_relations` — a deprecation with no stated replacement
  fails validation (`tests/unit/test_ontology_lifecycle.py` gates this on
  every shipped ontology file).
- State this domain's own deprecation SLA here, since the platform doesn't
  enforce a *timeline*, only that a replacement is named:
  - Minimum notice before a `deprecated` type/relation is actually removed:
    `<e.g. one minor version, or "N days">`.
  - Who is notified: `<downstream consumers, e.g. specific dashboards/queries>`.
  - Where deprecations are announced: `<changelog, Slack channel, etc.>`.
