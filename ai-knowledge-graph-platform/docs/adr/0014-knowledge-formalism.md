# ADR-0014: Formalism for contextual knowledge

**Status:** Accepted (2026-11-03); amended by ADR-0015 (2026-11-04): published decision tables may gate ontology actions. Knowledge evaluation still does not execute them.

## Context

Contextual knowledge is evaluated with bounded predicates: a requirement is a
conjunction of at most 20 predicates over named, evidenced facts, with a fixed
operator set, three-valued applicability and explicit missing-fact semantics
(`graphrag/context_graph/knowledge.py`). That covers constraints and
guidelines. It does not cover knowledge that maps inputs to outputs
(transformations, "which template or procedure applies"), and the choice of
formalism was never recorded.

## Options

| Option | Strengths | Failure modes |
| --- | --- | --- |
| **Bounded predicates (current)** | Total, terminating, explainable per predicate; missing facts are explicit (`unknown` or `violation`); easy to render for reviewers; no hidden evaluation order | No disjunction, arithmetic or cross-fact comparison: such rules stay unexecutable and escalate. Cannot express input-to-output mappings |
| **DMN-style decision tables** | Reviewable grid; hit policy makes ambiguity explicit; gaps and overlaps can be checked statically over finite domains | Full DMN brings FEEL expressions (a language to sandbox); without declared domains, completeness cannot be checked; tables grow combinatorially |
| **Datalog** | Recursion and derived facts (transitive scope, part-of chains) with clear semantics and termination | Closed-world negation is easy to misuse with incomplete facts; every derivation needs provenance to be explainable; harder for reviewers to read than a table |
| **Production rule engine (Rete, e.g. Drools)** | Expressive, mature, fast at scale | Conflict-resolution and agenda order make outcomes depend on rule order and salience; rules can loop; mutation of working memory hides why a result happened; the hardest to audit and replay |

## Decision

1. Keep **bounded predicates** for requirements (constraints, guidelines).
2. Add **DMN-style decision tables** as a bounded item kind for
   transformations and procedures expressed as input-to-output mappings
   (`graphrag/context_graph/decision_tables.py`): enum or numeric inputs with
   declared domains, conditions limited to `any`, `eq`, `in` and numeric
   `range`, and an explicit hit policy (`unique`, `first`, `any`, `collect`).
   Tables are **represented and validated only**: overlaps, gaps and
   conflicting outputs are checked over a finite partition of the domains and
   block publication when they are errors under the hit policy. They are not
   executed by the evaluator.
3. Content outside these vocabularies is **refused at load time**, never
   approximated: no FEEL, no expressions, no undeclared domains.
4. **Datalog** may be revisited for derived facts (for example transitive
   applicability) once a concrete case needs it; it would sit beside the
   predicates, with derivation provenance, not replace them.
5. **No production rule engine.** Order-dependent conflict resolution conflicts
   with exact replay and with "every applicable requirement is evaluated".

## Consequences

* A reviewer can read every executable rule (predicates are rendered in
  controlled language; tables are grids).
* Procedures that need branching, loops or external calls remain out of
  scope and must go to human review rather than be encoded approximately.
* Executing tables (and recording which rule fired in a decision) is a
  separate, later decision; until then a published table is reference
  knowledge with a validated shape, not an executed rule.
