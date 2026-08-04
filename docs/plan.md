# Sales Context Graph — Updated Implementation Plan and Codex Prompt

## 1. Objective

Build a production-oriented vertical slice that combines structured Salesforce CRM data, Gong-shaped sales-call transcripts, and Showpad-style content engagement into a Neo4j knowledge graph.

The first release must prove one differentiating workflow end to end:

> Given an opportunity, identify the objection raised by a stakeholder in the latest relevant call and recommend an appropriate content asset that the buyer has not already viewed, with exact evidence and an explainable entity-resolution decision.

The system is not a general sales assistant yet. It first establishes trustworthy ingestion, identity resolution, provenance, conflict handling, tenant isolation, and bounded context selection.

---

## 2. Architecture principles

1. CRM records provide canonical commercial identity, but are not assumed to be perfectly clean.
2. Transcript-derived information is stored as evidence-backed assertions, not unquestionable graph facts.
3. The persistent Knowledge Graph and the query-specific Context Graph are separate concepts.
4. Entity resolution is deterministic where identifiers are truly unique and probabilistic otherwise.
5. Similarity alone never silently auto-links an ambiguous entity.
6. Human review is asynchronous and does not block ingestion completion.
7. Identical ingestion is idempotent; changed and deleted source data is reconciled explicitly.
8. Every tenant-sensitive read and write is verified with adversarial cross-workspace tests.
9. LLMs extract typed information. They never resolve identities, calculate business scores, or write to Neo4j.
10. Scalability-sensitive interfaces are batch-shaped and versioned from the beginning.

---

## 3. Scope

### Included in the vertical slice

- Salesforce-shaped CRM contracts and ingestion
- Gong-shaped transcript contracts and ingestion
- Showpad-style content and engagement fixtures
- deterministic source identity and source-version reconciliation
- transcript windowing and typed extraction
- speaker resolution
- entity candidate generation and explainable resolution
- Claim, Mention, evidence, conflict, supersession, and review models
- asynchronous unresolved-mention review API
- tenant-safe Neo4j repositories
- one bounded Context Graph builder
- one evidence-backed content-recommendation use case
- evaluation of blocking recall, entity linking, provenance, and grounding
- Docker Compose, tests, documentation, and demo script

### Explicitly deferred

- complete production UI
- every CRM provider
- all ten sales-assistant use cases
- autonomous seller actions
- automatic CRM writeback
- enterprise SSO implementation
- database-per-customer deployment
- advanced graph algorithms and GNNs
- large-scale distributed ingestion infrastructure

---

## 4. Technical decisions

### Initial adapters

- CRM: Salesforce-shaped adapter
- transcript source: Gong-shaped adapter
- sales content: Showpad-style adapter and fixtures

Every adapter is behind a typed interface. Adding Dynamics, HubSpot, Chorus, Zoom, or another provider must not require changing the canonical domain model.

### Stack

- Python 3.12+
- FastAPI
- Pydantic v2
- Neo4j 2026.01 or newer, pinned to an exact supported release
- official Neo4j Python driver
- LangGraph only for durable workflow state and resumable processing where it adds value
- RapidFuzz
- pluggable batch embedding provider
- pluggable batch extraction provider
- structlog
- OpenTelemetry API/SDK and FastAPI/Neo4j instrumentation
- pytest and Hypothesis
- Docker Compose

The Neo4j version is pinned because tenant-prefiltered vector retrieval depends on filterable vector-index properties and the `SEARCH` clause available in Neo4j 2026.01+.

### Workspace and Division

For the vertical slice:

```text
workspace_id = security and data-isolation boundary
division_id  = Showpad organizational/permission dimension inside a workspace
```

They are not interchangeable. Every stored node carries `workspace_id`; Showpad-derived nodes may additionally carry `division_id`.

---

## 5. Canonical data model

### CRM entities

- Account
- Contact
- Lead
- Opportunity
- Seller
- Meeting
- Activity
- OpportunityContactRole
- SourceRecord
- SourceSnapshot

Salesforce conversions and merges are explicit:

```text
Lead -[:CONVERTED_TO]-> Contact|Account|Opportunity
Account -[:MERGED_INTO]-> Account
```

The resolver treats identifiers belonging to converted or merged records as aliases of the surviving canonical entity.

### Conversation entities

- Conversation
- Participant
- TranscriptSegment
- ExtractionWindow
- Mention
- SpeakerResolution

`TranscriptSegment` represents the immutable source-level sentence or utterance. `ExtractionWindow` is a derived grouping used only to reduce extraction cost. Evidence offsets always remain relative to the immutable source segment.

### Sales knowledge entities

- Product
- Feature
- Objection
- PainPoint
- Blocker
- BuyingSignal
- ActionItem
- Commitment
- ContentAsset
- Share
- AssetView

### Assertion and audit entities

- Claim
- ExtractionRun
- ResolutionDecision
- ReviewDecision
- Conflict
- ErasureEvent

### Stakeholder assignments

Do not store transcript-derived stakeholder roles only as mutable properties on `HAS_STAKEHOLDER`.

Use:

```text
(Opportunity)-[:HAS_ASSIGNMENT]->(StakeholderAssignment)-[:ASSIGNS]->(Contact)
```

`StakeholderAssignment` contains the materialized current view. Source-specific role, influence, sentiment, and authority remain Claims with provenance. Conflicting Claims can coexist.

---

## 6. Identity, versioning, and idempotency

### Stable source identities

```text
crm_entity_id = hash(workspace | source_system | object_type | external_id)
conversation_id = hash(workspace | source_system | external_call_id)
segment_id = hash(conversation_id | source_segment_index)
mention_id = hash(segment_id | char_start | char_end | normalized_surface | entity_type)
```

### Source versions

Every ingested source record includes:

```text
source_record_id
source_version
content_hash
ingestion_run_id
first_seen_at
last_seen_at
source_status = ACTIVE | SUPERSEDED | DELETED
```

Identical content is a no-op. Changed content creates a new source snapshot and triggers reconciliation. Deleted or missing source records are tombstoned only when the adapter supplies a trustworthy deletion or complete-snapshot signal.

### Assertion identity versus extraction execution

Do not include the extractor version in the stable assertion identity.

```text
assertion_id = hash(
  workspace |
  source_segment_id |
  evidence_char_start |
  evidence_char_end |
  canonical_subject |
  predicate |
  normalized_object |
  polarity
)

extraction_run_id = hash(
  provider |
  model |
  prompt_version |
  extractor_version |
  run_nonce
)
```

An identical assertion found by a newer extractor links to the existing Claim and adds a new extraction observation. A materially different interpretation creates a new Claim and may `SUPERSEDE` or `CONTRADICT` the prior Claim.

### Claim fields

```text
claim_id
workspace_id
subject_id
predicate
object_id | object_value
polarity = AFFIRMED | NEGATED | HYPOTHETICAL
source_type
source_record_id
source_segment_id
evidence_char_start
evidence_char_end
source_timestamp
speaker_id
speaker_role = BUYER | SELLER | UNKNOWN
confidence
valid_from
valid_to
transaction_from
transaction_to
is_superseded
adjudication_status = UNREVIEWED | ACCEPTED | DISPUTED | REJECTED
retention_class
erasure_status
created_at
```

Multiple non-superseded contradictory Claims may coexist. `is_superseded=false` means that a Claim has not been replaced; it does not mean the Claim is accepted truth.

---

## 7. Transcript extraction

### Window construction

1. Persist every source segment before extraction.
2. Accumulate consecutive speaker turns until a topic boundary, 60–90 seconds, or a configurable token budget is reached.
3. Add a small configurable overlap between adjacent windows.
4. Never discard the closing portion of a call.
5. Deduplicate assertions created from overlapping windows by stable `assertion_id`.

### Extraction provider

Use a batch interface:

```python
extract(windows: list[ExtractionWindow]) -> list[ExtractionResult]
```

Provide:

- a deterministic fixture extractor for tests;
- a real LLM adapter;
- strict Pydantic validation;
- bounded retry and repair attempts;
- explicit permanent failure after retries;
- provider timeout and rate-limit handling;
- no database access from the provider.

The transcript is untrusted input. The extraction prompt must delimit it as data, explicitly reject instructions contained inside it, expose no tools, and enforce input/window size limits.

### Optional extraction filtering

Filtering is disabled by default until measured against a `NullFilter` baseline.

Measure each filter tier as:

```text
extraction calls saved (%) versus claim recall (%)
```

Persist skipped segments with the filter version. Filtering affects only extraction, never source storage or Context Graph retrieval.

---

## 8. Entity and speaker resolution

### Stage A — deterministic matches

A deterministic rule auto-links only when it returns exactly one eligible entity inside the workspace:

```text
A1 exact external ID in the same source system
A2 exact normalized unique email for Contact or Seller
A3 exact canonical normalized name when unique and entity-type compatible
A4 exact approved alias when unique and entity-type compatible
```

Email-domain equality is not deterministic. It is relational evidence only.

### Candidate generation

Generate candidates before fuzzy scoring using tenant-safe:

- normalized prefix and full-text lookup;
- trigram or fuzzy-capable name lookup;
- vector retrieval with `workspace_id` configured as an in-index filterable property;
- relationship-based candidates from participants, meetings, opportunities, and products.

Union and deduplicate candidates. Start with a configurable cap of 50, but measure `blocking_recall@10`, `@25`, and `@50`. If the expected entity is not generated, report `candidate_generation_miss` separately from an ordinary unresolved result.

Full-text retrieval must also be tenant-safe; do not obtain a global top-k and merely discard other workspaces afterward.

### Probabilistic scoring

```text
lexical  = normalized fuzzy similarity
semantic = cosine similarity over contextual entity representations
base     = configured blend(lexical, semantic)
rel_bonus = capped sum of independent relational signals
final    = min(base + rel_bonus, 1.0)
margin   = top1_final - top2_final
```

Possible relational signals:

- known participant belongs to candidate Account;
- seller owns an open Opportunity for candidate Account;
- mentioned Product appears on that Opportunity;
- participant email domain agrees with the Account domain;
- temporally nearby Meeting or Activity references the candidate.

Each signal fires at most once and appears in the explanation.

### Decision policy

Initial configurable defaults, to be calibrated:

```text
unique deterministic match
  → AUTO_LINKED

base >= 0.75
AND final >= 0.90
AND relational_signals >= 1
AND top1_margin >= 0.08
  → AUTO_LINKED

final >= 0.55
  → PENDING_REVIEW

otherwise
  → UNRESOLVED
```

Similarity alone never auto-links. Domain equality alone never auto-links. Missing runner-up margin never auto-links.

### Speaker resolution

Resolve opaque speaker IDs before calculating Claim authority, using participant email, seller directory, meeting invitees, self-introduction evidence, and relationship context. Failed speaker resolution produces `speaker_role=UNKNOWN`; it does not discard the Claim.

---

## 9. Asynchronous review and reconciliation

An ambiguous Mention does not pause the entire ingestion.

```text
ingest → persist resolved and unresolved results → complete_with_review
      → reviewer resolves later → targeted reconciliation
```

A manual decision records:

- reviewer identity;
- timestamp;
- selected entity or rejection;
- candidate set shown;
- original scores and explanation;
- optional reviewer reason;
- affected Claims and materialized relationships;
- previous decision if overridden.

The review endpoint is API-only in this phase. No review UI is required.

---

## 10. Neo4j persistence and tenant isolation

### Repository rules

- parameterized Cypher only;
- repository/service separation;
- explicit transaction boundaries;
- managed transaction functions for retryable operations;
- bounded retry for transient failures;
- no retry for validation or constraint violations;
- deterministic lock ordering within batched writes;
- no direct Claim or Mention fan-out from Account;
- route evidence through Conversation, TranscriptSegment, and Opportunity.

### Tenant safety

Every application query must scope every matched endpoint by `workspace_id`, not merely include a parameter named `$workspace_id`.

The database execution wrapper has separate modes:

```text
tenant_query       requires workspace context
schema_query       allowlisted for migrations only
operational_query  allowlisted for health/index metadata only
```

Add integration tests containing two workspaces with intentionally identical names and external IDs. Tests must prove that reads, writes, relationships, full-text retrieval, vector retrieval, review operations, and evidence lookups cannot cross the boundary.

### Indexes

Prefer composite indexes matching tenant-scoped access patterns:

```text
(workspace_id, id)
(workspace_id, external_id)
(workspace_id, canonical_name)
(workspace_id, normalized_email)
(workspace_id, resolution_status)
(workspace_id, started_at)
(workspace_id, adjudication_status, is_superseded)
```

Create full-text and versioned vector indexes. Verify that every required index is online in readiness checks.

### Embedding/index versioning

Store:

```text
embedding_model
embedding_version
embedding_dimension
embedded_at
source_content_hash
```

Use versioned vector-index names and support backfill plus temporary dual-read during model migration.

---

## 11. Ingestion execution model

The API returns an ingestion ID instead of holding a long request open:

```text
POST /api/v1/ingestions/crm
POST /api/v1/ingestions/transcripts
POST /api/v1/ingestions/content-assets
GET  /api/v1/ingestions/{id}
```

States:

```text
ACCEPTED
NORMALIZING
EXTRACTING
RESOLVING
PERSISTING
COMPLETED
COMPLETED_WITH_REVIEW
FAILED_RETRYABLE
FAILED_PERMANENT
```

For the MVP, an in-process bounded worker may be used, but its interface must support later replacement with a durable queue. Define queue capacity, backpressure behavior, retry count, failure reason, and idempotency key.

Extraction may run concurrently. Graph writes are batched and partitioned by `workspace_id + conversation_id`, with deterministic lock ordering. Do not require a single global writer. If implementation evidence shows conflicting writes, serialize only the affected partition.

LangGraph may coordinate the workflow, but the authoritative status is persisted outside process memory. Do not use a memory-only checkpointer while claiming restart safety.

---

## 12. Context Graph builder

Inputs:

```text
question
workspace_id from authenticated context, never request body
seller_id
optional account_id
optional opportunity_id
optional conversation_id
time range
max_nodes
max_tokens
```

Selection pipeline:

1. deterministic scope filters;
2. tenant-safe full-text/vector candidate retrieval;
3. bounded directed traversal with configured relationship allowlist and maximum depth;
4. scoring by relevance, source authority, confidence, recency, and adjudication status;
5. greedy selection by score/token cost;
6. diversity caps per conversation, predicate, and subject-predicate;
7. preserve both sides of a relevant conflict.

Every query has a timeout and maximum database-result budget. Avoid N+1 repository calls.

Response:

```text
selected nodes
selected relationships
Claims
evidence references with segment-relative spans
unresolved Mentions
Conflicts
selection score
selection reason per item
budget usage
truncation indicators
```

### First wired use case

Given an Opportunity:

1. find the most recent relevant call;
2. identify an affirmed, non-rejected Objection raised by a buyer stakeholder;
3. identify ContentAssets that address the Objection through curated tags or explicit mappings;
4. exclude assets already viewed by that buyer;
5. rank remaining assets;
6. return the recommendation, exact transcript evidence, Claim IDs, and ranking explanation.

The mapping between ContentAsset and Objection must have an explicit source. It may come from a curated taxonomy or reviewed Claim; it must not be invented at query time.

---

## 13. Security, privacy, and lifecycle

- secrets from environment only;
- authentication dependency and authorization-policy interface;
- `workspace_id` derived from authenticated context;
- division/team/opportunity authorization hooks;
- request and transcript size limits;
- PII-safe logs and traces;
- no transcript text, email, or access token in INFO logs or metric labels;
- audit events for automatic and manual resolution decisions;
- prompt-injection fixtures for extraction;
- retention class, legal-hold state, and erasure state;
- erasure propagation to text, embeddings, search indexes, caches, derived summaries, and Claims;
- erasure audit record without retaining erased personal content;
- documented backup-retention limitation.

The vertical slice is not described as production-authorized until a real identity provider and policy implementation exist.

---

## 14. Observability

Emit one span per workflow stage with `ingestion_id`, `workspace_id`, provider, duration, outcome, and retry count.

Emit structured resolution events containing IDs, candidate scores, signal names, margin, and decision status. Do not emit raw transcript text or email.

Metrics:

- ingestion count and duration by status;
- extraction windows and provider calls;
- extraction failures and retries;
- candidate-generation latency;
- blocking recall in evaluation runs;
- auto-link, review, unresolved, and rejection counts;
- Claims created, superseded, conflicted, and erased;
- Context Graph latency, result count, and budget truncation;
- queue depth and oldest-job age.

Avoid unbounded-cardinality metric labels. Workspace-level operational detail should be available through controlled logs/traces or bounded reporting, not arbitrary metric labels at large tenant counts.

---

## 15. Evaluation and acceptance criteria

### Identity and ingestion

- stable IDs across processes;
- second identical ingest changes zero graph counts;
- corrected source record supersedes or updates the prior snapshot without leaving active stale Claims;
- deletion/tombstone fixture invalidates derived data;
- no cross-workspace read, write, edge, vector result, full-text result, or review action;
- true candidate appears in the blocking set at the selected cap.

### Entity resolution

- `Volks Wagen` resolves to `Volkswagen Group` with at least two named relational signals;
- the same mention without relational evidence remains pending review;
- `Volkswagen Financial Services` is present as a distractor;
- a weak-base candidate cannot auto-link only through bonuses;
- insufficient top1/top2 margin forces review;
- domain equality alone never auto-links;
- duplicate exact names do not deterministic-link;
- report entity-link accuracy, auto-link precision, review rate, unresolved recall, and blocking recall.

### Extraction and provenance

- deterministic fake extraction is byte-stable;
- negated and hypothetical variants remain distinct Claims;
- window overlap does not duplicate Claims;
- evidence spans map to exact source segments;
- opaque speaker IDs still produce Claims with appropriate authority;
- invalid LLM output fails explicitly after bounded retries;
- prompt-injection transcript fixture cannot change extractor instructions;
- provenance completeness is 100% for persisted transcript-derived Claims.

### Context and grounding

- every factual response item cites a Claim served in the Context Graph;
- no cited Claim may be outside the served Context Graph;
- conflicting relevant Claims survive selection;
- hard node/token/query budgets are enforced;
- already-viewed content is excluded;
- recommendation cites the Objection evidence and mapping source.

Operational definitions:

```text
grounded factual item = factual item citing at least one served Claim
hallucinated item = factual item with no served Claim or an invalid citation
context recall = expected Claims served / expected Claims in golden set
```

---

## 16. Implementation phases

| Phase | Scope | Exit criterion |
|---|---|---|
| P0 | Contracts, stable IDs, source snapshots, assertion/extraction identity | Models round-trip; ID properties pass; change/delete semantics tested without DB |
| P1 | Neo4j schema, tenant-safe execution modes, repositories, indexes | Adversarial two-workspace integration suite passes |
| P2 | CRM/content ingestion and source reconciliation | Identical, changed, merged, converted, archived, and deleted fixtures behave correctly |
| P3 | Transcript persistence, windowing, fake/real extraction adapters, speaker resolution | Polarity, overlap, opaque speakers, invalid output, and prompt injection tests pass |
| P4 | Candidate generation, entity resolution, asynchronous review | VW positive/negative/distractor/margin tests and blocking-recall evaluation pass |
| P4.5 | Context Graph and content recommendation | End-to-end demo returns unviewed content with exact evidence and selection reasons |

Do not widen scope until P4.5 is executable from a clean checkout.

---

# Codex implementation prompt

You are a Staff AI Engineer and Knowledge Graph architect. Implement the first trustworthy vertical slice of **Sales Context Graph** according to this document.

## Working method

1. Inspect the complete repository, its instructions, dependency files, migrations, tests, and current git status before editing.
2. Report a concise repository-specific implementation plan mapped to P0–P4.5.
3. Reuse working code only after verifying it with tests. Do not assume that paths or capabilities mentioned in historical documents exist.
4. Preserve unrelated user changes.
5. Implement one phase at a time and run the relevant tests after each phase.
6. Do not claim a capability unless executable code and a meaningful test demonstrate it.

## Non-negotiable implementation rules

- No `assert True`, hardcoded demo result, silent exception swallowing, or placeholder implementation presented as complete.
- LLM adapters extract typed data only. They never resolve entities, score candidates, authorize access, or write to Neo4j.
- All Cypher values are parameterized.
- `workspace_id` comes from trusted request/authentication context, not a user-controlled body field.
- Every tenant-sensitive matched node is scoped; parameter presence alone is not considered enforcement.
- Identical ingestion is a no-op; source changes and deletions reconcile derived records.
- Claim identity includes polarity and evidence span but not extractor version.
- Domain matching is corroboration only.
- Auto-linking requires calibrated textual evidence, relational corroboration, and a sufficient runner-up margin.
- Ambiguous Mentions persist for asynchronous review and do not block ingestion completion.
- Batch-shaped provider interfaces are required.
- Every source segment is persisted even when extraction is skipped.
- No Claim or Mention is attached directly to Account.
- Vector and full-text retrieval are tenant-safe before top-k truncation.
- Pin Neo4j and all important dependencies; document compatibility assumptions.
- Use a durable workflow/status store or explicitly describe and test the MVP limitation. Never claim restart safety with process memory only.

## Required API for this slice

```text
POST /api/v1/ingestions/crm
POST /api/v1/ingestions/transcripts
POST /api/v1/ingestions/content-assets
GET  /api/v1/ingestions/{id}
GET  /api/v1/unresolved-mentions
POST /api/v1/unresolved-mentions/{id}/resolve
POST /api/v1/context/build
GET  /api/v1/claims/{id}/evidence
GET  /health
GET  /ready
```

`/health` is process liveness. `/ready` checks Neo4j connectivity, schema migration state, and required online indexes.

## Required fixtures

- two workspaces with deliberately overlapping names and external IDs;
- `Volkswagen Group` and distractor `Volkswagen Financial Services`;
- transcript mention `Volks Wagen`;
- Elena Popescu as a Contact and participant;
- seller-owned open Opportunity;
- Showpad Genie and Shared Spaces product references;
- security blocker, pricing objection, negated variants, and hypothetical variants;
- action item with deadline;
- at least two ContentAssets addressing the objection;
- AssetView showing that one candidate asset was already viewed;
- opaque speaker IDs;
- duplicate exact Account names;
- shared email-domain case that must not auto-link;
- corrected transcript version and deleted transcript fixture;
- prompt-injection text inside a transcript;
- overlapping tenant data used by every isolation test.

## Required deliverables

- working source code;
- migrations/constraints/index definitions;
- Docker Compose;
- `README.md` with setup and Mermaid architecture diagram;
- `architecture.md`;
- `ontology.md`;
- `entity-resolution.md` matching the implemented algorithm;
- `security-and-tenancy.md`;
- `evaluation.md` with metric definitions and real results;
- sample data;
- Makefile or equivalent commands;
- example curl commands;
- `demo_volkswagen.py` printing candidates, component scores, relational signals, top-two margin, final status, recommended asset, and evidence IDs;
- unit, integration, isolation, and end-to-end tests.

## Completion report

At the end, report accurately:

- files added and modified;
- architecture decisions made;
- commands executed;
- formatting, static-analysis, and test results with real counts;
- measured resolution and grounding metrics;
- known limitations and deferred work;
- next recommended milestone.

If a requirement cannot be implemented in the repository or environment, state the exact blocker. Do not replace it silently with a stub.
