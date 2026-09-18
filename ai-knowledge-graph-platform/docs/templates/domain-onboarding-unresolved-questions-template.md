# Unresolved Questions Log Template

Roadmap "P1 — reusable domain onboarding and governance pack". Tracks a
domain-onboarding question from "raised" to "closed" so nothing raised in an
SME workshop quietly disappears. A question stays a row here — never
deleted — until it has a `Resolution`; then it moves to
`domain-onboarding-entity-relationship-decisions-template.md` if it produced
a modeling decision, or stays here marked `closed — no modeling change` if
it didn't.

| ID | Question | Raised by / date | Blocks | Status | Resolution |
|---|---|---|---|---|---|
| `Q-001` | `<the actual open question, in the SME's own terms>` | `<name, YYYY-MM-DD>` | `<type/relation this blocks modeling, or "none yet">` | `open \| in_progress \| closed` | `<how it was resolved, or blank while open>` |

## Notes

- `Blocks` is what makes this actionable rather than a wishlist: a
  `status: open` row with a non-empty `Blocks` is a real reason a modeling
  decision can't be finalized yet, and should show up in review the same
  way an unmapped ontology removal does
  (`graphrag/graph/ontology_migration.py::plan_migration`'s `warnings`).
- Prefer closing a question with "we deliberately chose not to model this"
  over leaving it open indefinitely — an explicit non-decision is more
  useful to the next person than silence.
