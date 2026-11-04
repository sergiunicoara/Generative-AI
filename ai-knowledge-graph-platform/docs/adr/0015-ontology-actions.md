# ADR-0015: Ontology actions and functions

**Status:** Accepted (2026-11-04)

## Context

Agents could read and answer from the graph but had no governed way to change
it: the only writes an agent could reach were the work-order commands. The
decision trace recorded actions after the fact (`CGAction`, `ActionNode`), but
nothing in the canonical model declared an operation that could be proposed,
checked against the model and approved before it happened. Decision tables
(ADR-0014) were validated and published but never executed.

## Decision

1. **Actions and functions are declared in the canonical model YAML**
   (`actions:` and `functions:` in `ontology/models/*.yaml`, validated at load).
   An action names the object type it applies to (which must have exactly one
   key property), typed parameters, submission criteria over the object's
   properties, parameters and functions, effects that set properties of that
   object (to a literal, a parameter or the action time), an optional decision
   gate, whether approval is required and the scope it needs. A function is
   read-only: a property of the object or a count of one relation in one
   direction. Actions and functions do not change the compiled OWL, SHACL or
   Neo4j artifacts.
2. **One flow** (`graphrag/semantic_model/action_service.py`): read the object
   by label, key and tenant; run the pure check
   (`graphrag/semantic_model/actions.py`): parameters, criteria, decision gate,
   and the post-change state through `SemanticMutationValidator`; record the
   proposal with the snapshot, parameters, verdict, model digest and action
   time; apply at once when no approval is declared, otherwise wait for a
   different person to approve (four-eyes). Before writing, re-read the object;
   if a property the decision depended on changed, or the action no longer
   passes, the proposal becomes `stale`. The write is one Cypher statement
   guarded on those properties, inside a `CorpusMutation`.
3. **Every transition is logged**: the proposal record carries its events and
   an append-only event table keeps them (`GovernanceStore`). `replay` re-runs
   the recorded check and must reproduce the verdict, violations and changes.
4. **Decision tables may gate actions.** A gate names a published table of a
   knowledge-pack domain, maps its input columns to subjects, and requires one
   output to be `true`. `evaluate_table` applies the hit policy (`collect` is not
   a verdict and is refused). A missing or unpublished table fails closed. This
   is the only execution path for decision tables: knowledge evaluation still
   does not execute them (ADR-0014 stands for that path).
5. **Exposure** goes through the capability registry only: `kg.action.propose`,
   `kg.action.decide`, `kg.action.list` and `kg.function.call`, so MCP, the
   stateless MCP transport and the LangGraph tools get the same scopes, tenant
   binding and audit. The action's own scope is checked in addition to the
   capability scope. GraphQL stays query-only and does not expose them.

## Consequences

* Agents can change the graph only through declared, typed operations whose
  checks are visible in the model and whose outcomes are attributable.
* Effects are limited to properties of the target object. Creating nodes or
  relations needs a later extension with its own guard semantics.
* Relation counts used by criteria are re-checked before the write but are not
  part of the atomic guard; a relation added between the re-check and the write
  is not detected. Property guards are atomic.
* The proposal log is an audit trail and is not erased (erasure registry
  entry `governance.ontology_actions`); free-text parameters must not carry
  personal data.
* Verified offline only (temporary SQLite store, a fake graph); the guarded
  write has not been run against a live Neo4j.
