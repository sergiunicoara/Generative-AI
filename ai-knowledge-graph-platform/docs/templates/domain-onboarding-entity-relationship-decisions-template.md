# Entity and Relationship Decisions Log Template

Roadmap "P1 — reusable domain onboarding and governance pack". The
reviewable record of *why* a domain's `config/ontologies/<tenant>_*.yml`
looks the way it does — a `git blame` on the YAML shows *what* changed, not
the business reasoning behind it. Keep one row per decision, appended to,
never rewritten (mirrors the ontology's own deprecate-don't-delete rule:
`ontology/README.md`'s "Lifecycle" section).

| Date | Entity/relation | Decision | Rationale | Deciders | Ontology PR/commit |
|---|---|---|---|---|---|
| `<YYYY-MM-DD>` | `<TYPE_NAME or RELATION_NAME>` | `<e.g. "modeled as its own type, not a property">` | `<why — cite the SME workshop note if one exists>` | `<names>` | `<link>` |

## Notes

- A decision to *not* model something is still a decision — record it the
  same way, with `Entity/relation` naming the candidate that was rejected.
- If a later decision reverses an earlier one, add a new row referencing the
  old row's date rather than editing the old row — the history of changing
  your mind is itself useful context for the next reviewer.
