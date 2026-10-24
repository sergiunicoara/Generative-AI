# Contextual knowledge evaluation

A shared platform capability that lets agents and deterministic services ask,
for a given business context:

* which knowledge **applies** (applies / does_not_apply / unknown);
* which applicable requirements **can be checked**, and the result of each
  (pass / fail / insufficient_evidence / conflict / not_evaluated);
* whether the **evidence** supports the result (every predicate cites the facts
  it used, every requirement cites its source);
* when **conflicts, ambiguity or missing evidence** require human review.

Demonstrated on the Energy maintenance path and on a small **synthetic**
packaging/labeling pack. The packaging rules are invented demonstration rules,
not FDA, EMA or QRD requirements, and nothing here has had domain-expert review.

## Architecture and extension points

| Layer | Module | Notes |
| --- | --- | --- |
| Contracts | `graphrag/context_graph/knowledge.py` | `KnowledgeItem`, `KnowledgeException`, `BusinessContext`, `Fact`, `Predicate`/`Constraint` |
| Applicability | `graphrag/context_graph/applicability.py` | Three-valued, no authorization logic |
| Evaluation | `graphrag/context_graph/knowledge_evaluation.py` | Authorization filter, supersession, exceptions, validation, conflicts, decision |
| Releases | `graphrag/context_graph/knowledge_release.py` | Packs, review helpers, publication gate, regression cases |
| Persistence | `graphrag/domains/energy/governance_store.py` | `knowledge_releases` / `knowledge_active_releases` in the existing durable governance store |
| Serving + cache | `graphrag/context_graph/knowledge_service.py` | Active-release resolution, decision cache |
| MCP | `mcp_server/capabilities/knowledge_evaluate.py` | `kg.knowledge.evaluate@1.0.0` |
| HTTP | `api/routes/knowledge.py` | `POST /knowledge/evaluate` (same registry call path) |
| Energy | `graphrag/domains/energy/knowledge.py`, `answers.py` | Bulletin projection, per-asset evaluation on `maintenance_review` |
| Domain pack | `config/knowledge_packs/packaging_labeling.yml` | Synthetic demonstration pack |

Reused rather than duplicated:

* **Identity and access**: tenant from the signed token (`get_tenant`), caller
  principals from `AccessContext.from_claims`, and a Python twin of
  `document_access_predicate` (`knowledge_evaluation.can_read`, always
  fail-closed).
* **Constraint vocabulary**: the Context Graph `ConditionOperator` set and its
  matcher; no `eval`, no expressions.
* **Escalation**: a declarative `PolicyVersion`
  (`knowledge-evaluation-escalation@1.1.0`) evaluated by the existing policy
  engine. The first-match engine decides only the overall escalation; it is
  not used to evaluate simultaneous obligations.
* **Releases**: the durable governance store's append-only rows plus active
  pointer and rollback, as already used for Energy publications.
* **Extraction**: `ObligationDraft` is unchanged;
  `KnowledgeItem.from_obligation_draft` adapts one into a draft.

New domain: add a pack under `config/knowledge_packs/`, regression cases under
`evals/knowledge_eval/`, and publish it with `publish_pack`. A domain that
already has governed data (like Energy) can instead project it into
`KnowledgeItem`s, as `graphrag/domains/energy/knowledge.py` does.

### Authoritative representation and derived projections

The **authoritative** representation of served knowledge is a published
knowledge release: the content-addressed snapshot of a pack's published items
and exceptions, stored in the durable governance store and referenced by every
decision (`release.release_id`, `release.content_hash`). The pack YAML is the
version-controlled curation surface; decisions, compact Energy summaries and
MCP payloads are derived views.

For Energy, the authoritative source remains the published RDF graph (SHACL
gated). Knowledge items are a deterministic projection of its guidance
revisions, and the release reference is the Energy publication `version_id`.

**Not implemented**: an RDF/SHACL projection of knowledge items through the
canonical semantic-model compiler (`graphrag/semantic_model`). Item validation
is currently Pydantic-only. This is tracked in the roadmap.

## Separate axes

Source authority, normative modality, specificity, confidence and temporal
validity are separate fields and never substitute for each other:

* `authority` records who issued an item (regulatory, customer, brand,
  internal, extracted).
* `modality` records how strongly it binds: mandatory, recommended, permitted
  or prohibited.
* Specificity is how narrow `applicability` is. It never authorises an
  override.
* `confidence` is extraction confidence. It is informational only, and a
  confidence of 0.99 does not make text executable.
* `valid_from`/`valid_to` hold valid time and `recorded_at` holds transaction
  time.

## Applicability versus validation

Applicability asks whether a requirement binds in this context and view.
Validation asks whether the evidenced facts satisfy it. They are computed and
reported separately: "does not apply" is `not_evaluated` with reason
`not_applicable`, never a pass, and "applies but fails" is `fail`.

### Applicability semantics

| Situation | Result |
| --- | --- |
| Dimension scope `any` (unrestricted) | Satisfied without any context value |
| Dimension restricted, request does not state it | `unknown` (`context_missing`). Missing context never broadens. |
| Item declares no scope for a dimension | `unknown` (`applicability_metadata_unknown`). Unknown metadata is never "everywhere". |
| Several context values (e.g. two markets) | Applies if any value is in scope; the matched values and `partial_scope_match` are reported |
| One dimension definitely excludes | `does_not_apply`, even if another dimension is unknown |
| Recorded after `known_as` | `does_not_apply` (`not_recorded_at_view`) |
| Outside the valid window at `as_of` | `does_not_apply` (`not_yet_valid`/`expired`) |
| Window present but no `as_of` | `unknown` (`as_of_missing`) |
| Superseded by a later revision that applies and covers the whole bound context | `does_not_apply` (`superseded`). Supersession is transitive (C supersedes B supersedes A means C supersedes A, even if B is not in force) and independent of input order. |
| A later revision whose applicability is unknown | `unknown` (`supersession_undetermined`) |
| A later revision covering only part of the bound context (e.g. one of two markets) | stays `applies` (`partially_superseded`); the pair is ordered, so it is not an `ambiguous_revision` conflict |
| Cyclic supersession lineage | `unknown` (`supersession_cycle`); the release gate also rejects cycles |

Dimensions are orthogonal: customer, brand, market, product, product family
and workflow. No hierarchy is assumed (a market is not a child of a customer).

## Normative semantics and exception handling

* Mandatory and prohibited requirements are **blocking**. A recommendation is
  reported, never blocking, and never cancels a mandatory requirement.
* For a prohibition, the constraint describes the forbidden state: if it holds,
  the result is `fail`.
* A permission has no compliance check. A permission and a prohibition on the
  same subject form a **conflict**: a permission never overrides a prohibition.
* `overrides` on an item records intent only. Without an authorised exception
  it yields an `unauthorized_override` conflict, and both requirements stay in
  force.
* Authorised exceptions (`KnowledgeException`) are scoped, time-bounded,
  cited, and carry `granted_by` and `approval_ref`:
  * `waiver`: the target does not bind within the exception's scope.
  * `precedence`: an approved resolution; the winner prevails over the targets.
  * **Coverage.** An exception changes a requirement only if it applies *and*
    covers every context value the requirement binds (per dimension, the
    requirement's bound values must be a subset of the exception's). A waiver
    for market A on a carton sold in A and B leaves the requirement in force
    and evaluated; the exception is listed in `partial_exception_refs`.
    Several partial waivers are not unioned (documented limitation).
  * **Effective winner.** Precedence suppresses a target only when the winner
    is visible to the caller, applies, is not waived or superseded, covers the
    target's bound context, and is not itself the target of another applicable
    precedence. A winner that definitely does not apply, or is fully waived,
    makes the precedence moot: the target stays in force without review. A
    winner that is unknown, superseded, partly waived, inaccessible (reported
    without naming it) or contradicted by another precedence leaves the target
    in force and escalates (`unresolved_exception`). Results do not depend on
    exception order.
  * An exception whose own applicability is unknown, with a target in play,
    also escalates.
* Conflict kinds:
  * `mandatory_vs_prohibited` and `permission_vs_prohibition`.
  * `incompatible_requirements`: provably unsatisfiable together, meaning
    different required values, disjoint allowed sets, or numeric bounds with an
    empty intersection. Bounds in the same direction, such as ≥ 6 and ≥ 8, are
    compatible.
  * `ambiguous_revision`: two revisions of one series in force with no
    supersession edge.
* Facts: a missing fact, or one without evidence, makes its predicate
  `unknown` (`insufficient_evidence`). It counts as a violation only when the
  predicate declares `when_missing: violation`, and then it fails under every
  modality, including prohibitions (`absence_is_violation`): absence is never
  read as "the forbidden state is absent".
* Fact types are compared strictly, never coerced: numeric operators need a
  finite number; `eq`/`ne` need the same scalar type; `in` needs a scalar of a
  type present in the allowed list; `contains` needs a list or a string.
  `bool` is not a number. A mismatch (e.g. `"7"` against `gte 6`) gives
  predicate reason `fact_type_mismatch` and outcome `insufficient_evidence`,
  never a crash or a pass. NaN and infinities are rejected by the `Fact` and
  `Predicate` contracts (HTTP 422 / `invalid_request`).
* Extracted text never becomes an executable rule by itself. A constraint on an
  approved or published item, or on any extracted item, needs
  `constraint_provenance` (a human author and an approval reference).
* Escalation (first match in the declarative policy) runs in this order:
  1. Unresolved conflict: escalate.
  2. Unresolved exception (unconfirmed precedence, or an exception whose scope
     cannot be determined): escalate.
  3. Blocking failure: deny, giving `non_compliant`.
  4. Blocking item with unknown applicability: escalate.
  5. Insufficient evidence on a blocking item: escalate.
  6. A blocking requirement that is not machine-checkable: escalate.
  7. No applicable knowledge when knowledge was expected: escalate.

  `human_review_required` lists every reason, not only the first.

The decision is inspectable and contains no hidden reasoning:

* the context and the facts used;
* knowledge IDs and versions;
* per-requirement applicability, outcome, predicates and citations;
* missing evidence;
* conflicts, and the exceptions or resolutions used;
* the escalation result, and the human-review reasons;
* the release used.

### Authorization

Items and exceptions are filtered to the token tenant, to `published` status
and to what the caller may read **before** applicability, validation or
conflict detection. Unauthorized knowledge never appears as a candidate,
explanation, conflict or citation. A test checks the serialized decision for
the restricted item's ID and source. Business context can narrow what applies
but never grants access.

### Semantic retrieval

Semantic similarity does not decide applicability. The evaluator considers
every published, authorized requirement of the release. `semantic_candidates`
only orders the results. The regression case KE-13 ranks an attractive advisory
item first and checks that the applicable mandatory rule is still evaluated
and fails.

## Lifecycle and release guarantees

* **Drafts are reviewable**: `review_queue(pack)` lists draft and in-review
  entries with what they still need. `approve_item` creates the next version
  with the reviewer identity, which must come from the authenticated `sub`, and
  re-runs contract validation.
* **Only published entries are served.** Draft, in-review, approved and
  withdrawn entries are not.
* **One effective version per id.** The evaluator takes the highest version of
  each item and exception id *before* the status and access filters, so an
  older published version never stands in for a newer withdrawn or
  inaccessible one, regardless of input order.
* **Publication runs a gate** (`publish_pack`):
  * contract validation;
  * version checks: at most one published version per item or exception id,
    and it must be the latest version of that id;
  * referential checks: supersession, override and exception targets exist; no
    supersession cycles;
  * the domain's regression cases against the candidate release. Cases are
    mandatory: publishing with none is refused, so `regression: passed` is
    never recorded for an empty gate.

  Any failure raises `KnowledgePublicationError` and leaves the active release
  unchanged. The failed-gate test checks this.
* **Releases are append-only.** Identical content is idempotent, rollback
  appends a new version that points at earlier content, and history is kept.
  The content hash covers pack metadata (tenant, domain, pack version,
  description) and the disclaimer as well as items and exceptions, so a
  disclaimer-only change is a new audited release.
* **Decisions name their release.** Energy answers now also carry
  `publication_version_id`.
* **Caching**: the decision cache key includes the tenant, the caller's access
  fingerprint, the domain, release id plus content hash, the full business
  context including `as_of`/`known_as`, the facts, the candidate order and the
  schema version. `QueryCacheContext` gained `knowledge_release` and
  `business_context_fingerprint`; its schema version moved from 3 to 4.

Not implemented: an HTTP route or review UI for curating and publishing packs.
Publication is library- and script-driven.

## API and MCP usage

```http
POST /knowledge/evaluate        (Bearer token with read scope; tenant from the token)
{
  "domain": "packaging_labeling",
  "context": {"market": "NORTHLAND", "customer": "CORNERSHOP", "brand": "OTHERBRAND",
              "product": "PX-200", "product_family": "otc_tablets", "workflow": "label_review",
              "as_of": "2026-08-01T00:00:00Z"},
  "facts": {"label.min_font_pt": {"value": 7, "evidence": [{"source_id": "ARTWORK-PX200-R3"}]}}
}
```

* **MCP:** call `kg.knowledge.evaluate@1.0.0` with the same `domain`, `context`
  and `facts`. The stdio tool is `evaluate_contextual_knowledge`.
* **Errors:**
  * a registry denial (`missing_scope`, `tenant_mismatch`, `invalid_arg`)
    maps to 403, 403 and 422;
  * `invalid_request` maps to 422;
  * `no_published_release` maps to 404.
* **Consumer example:** `agent_action()` in `scripts/knowledge_eval_demo.py`
  acts on the result:
  * `non_compliant`: block release and list the failed requirements (plus any
    review reasons);
  * approve only when the decision is `compliant` with no review flag;
  * everything else (`needs_review`, `no_applicable_knowledge`, any review
    flag): route to human review. Incomplete or unresolved decisions never fall
    through to approval.

## Commands

```bash
# deterministic evaluation set (writes evals/knowledge_eval/last_run.json)
python scripts/run_knowledge_eval.py

# walkthrough: publish the synthetic pack to a temp store, evaluate six contexts
python scripts/knowledge_eval_demo.py

# targeted tests
python scripts/run_pytest.py tests/unit/context_graph/test_contextual_knowledge.py \
    tests/unit/test_knowledge_capability.py tests/unit/test_energy_contextual_knowledge.py \
    tests/unit/context_graph/test_contextual_knowledge_audit.py
```

## Limitations and safe claims

Safe to say:

* "A deterministic contextual-knowledge evaluator with three-valued
  applicability, normative modality, authorised exceptions, conflict detection
  and gated, versioned releases."
* "Exposed through the authenticated API and MCP registry."
* "Wired into the Energy maintenance-review answer."
* "Measured on a 26-case synthetic engineering evaluation set written by the
  developer." (KE-23 to KE-26 were added by the 2026-10-24 audit, with
  expectations written before the fixes.)

Do not say:

* That it checks regulatory compliance or encodes real FDA, EMA or QRD rules.
* That domain experts validated it.
* That it is deployed or operated in production.
* That the evaluation numbers measure real-world accuracy. They measure
  agreement with the developer's own expectations.

Known limitations:

* Constraints are conjunctions of bounded predicates. Disjunction, arithmetic,
  units and cross-fact comparisons are out of scope, and such requirements stay
  non-executable and escalate.
* Conflict detection is pairwise over shared facts and subjects. It does not
  prove the joint satisfiability of more than two requirements.
* Energy guidance is projected as *recommended* (advisory review), matching the
  answer path. The Energy answer text and abstentions remain decided by its own
  temporal selection, and a test checks that both agree.
* There is no knowledge-item RDF/SHACL projection yet, and no pack
  curation/publish HTTP route.
* Verification is unit-level and in-process (temporary SQLite, synthetic pack).
  Nothing has been live-validated against Neo4j, a deployed API or a real MCP
  client.

## Audit 2026-10-24

Eight reported defects were reproduced against the pre-fix code and fixed. Each
has a regression test in
`tests/unit/context_graph/test_contextual_knowledge_audit.py`:

1. partial-context exceptions suppressed obligations outside their scope;
2. precedence suppressed its target without an effective winner;
3. a prohibition with `when_missing: violation` passed when the fact was absent;
4. accepted fact types (`"7"`, lists) crashed numeric comparison;
5. supersession chains depended on input order;
6. publishing with zero regression cases recorded `regression: passed`;
7. several published versions of one exception were all applied;
8. disclaimer-only release updates were deduplicated.

Known limitations that remain: partial waivers are not unioned; a partially
superseded revision is reported with its full matched context; the evaluation
set is still developer-authored.
