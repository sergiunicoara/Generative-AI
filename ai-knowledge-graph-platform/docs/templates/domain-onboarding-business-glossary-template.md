# Business Glossary Template

Roadmap "P1 — reusable domain onboarding and governance pack". Use this to
capture a new domain's business vocabulary *before* it becomes
`config/ontologies/<tenant>_*.yml` type/relation names — the glossary is the
human-readable source; the YAML is the machine-enforced projection of it
(`graphrag/graph/domain_ontology.py:validate_ontology_document` enforces
`UPPER_SNAKE_CASE` type/relation names — decide the mapping here, not there).

| Business term | Definition (plain language) | Maps to entity/relation type | Business owner | Status |
|---|---|---|---|---|
| `<term>` | `<one or two sentences, no jargon>` | `<ENTITY_TYPE or RELATION_NAME, or "n/a — not modeled">` | `<name/role>` | `draft \| reviewed \| active` |

## Notes

- A term with no `Maps to` entry is a deliberate scope decision, not an
  oversight — record it as `n/a` with a one-line reason, not a blank cell.
- Two rows should never map to the same type/relation without a documented
  reason (a synonym list, not two competing definitions of one thing).
- When a term's definition changes materially, bump the row's status back to
  `draft` and route it through the same review the ontology YAML change
  itself gets (`ontology/README.md`'s "Process: proposing, reviewing,
  migrating, releasing a change").
