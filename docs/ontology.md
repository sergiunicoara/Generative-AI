# Ontology

The canonical domain model, as actually implemented in `src/domain/*.py`
(Pydantic v2 — these are the source of truth; this document describes them,
it doesn't define anything new).

## CRM entities (`src/domain/crm.py`)

| Entity | Key fields | Notes |
|---|---|---|
| `SourceRecord` | `source_record_id, workspace_id, source_system, object_type, external_id, source_status, first_seen_at, last_seen_at, current_snapshot_id` | Frozen. `source_record_id` == the entity's own id (both are `crm_entity_id(...)` — see `entity-resolution.md`'s identity section). |
| `SourceSnapshot` | `snapshot_id, source_record_id, source_version, content_hash, ingestion_run_id, captured_at, superseded` | Frozen. One per content change; never overwritten, only superseded. |
| `Account` | `account_id, workspace_id, source_record_id, name, domain, merged_into_account_id` | `merged_into_account_id` implements `Account -[:MERGED_INTO]-> Account`. |
| `Contact` | `contact_id, workspace_id, source_record_id, account_id, name, email` | `.normalized_email` is a computed property (lowercased/stripped), not a stored validated type — real CRM exports contain malformed-but-real addresses. |
| `Lead` | `lead_id, ..., converted_to_type, converted_to_id` | `converted_to_type ∈ {Contact, Account, Opportunity}`, implements `Lead -[:CONVERTED_TO]->`. A real Salesforce conversion can produce all three simultaneously; this model holds one (Contact > Account > Opportunity priority — see `src/ingestion/adapters/salesforce.py::parse_lead`). |
| `Seller` | `seller_id, workspace_id, source_record_id, name, email` | |
| `Opportunity` | `opportunity_id, ..., account_id, seller_id, name, stage, is_open` | |
| `OpportunityContactRole` | `role_id, opportunity_id, contact_id, role` | |
| `Meeting`, `Activity` | `..., opportunity_id, occurred_at` | |

## Conversation entities (`src/domain/conversation.py`)

| Entity | Key fields | Notes |
|---|---|---|
| `Conversation` | `conversation_id, workspace_id, source_record_id, source_system, external_call_id, occurred_at, opportunity_id, account_id` | |
| `Participant` | `participant_id, conversation_id, speaker_label, contact_id, seller_id, role` | `speaker_label` is the opaque source-provided id; `contact_id`/`seller_id`/`role` are filled in by speaker resolution, not at parse time. |
| `TranscriptSegment` | `segment_id, conversation_id, source_segment_index, speaker_label, text, start_ms, end_ms` | **Frozen** — the immutable source-level sentence/utterance. Persisted before any extraction runs, even if extraction is skipped. |
| `ExtractionWindow` | `window_id, conversation_id, segment_ids, start_segment_index, end_segment_index` | A derived grouping to reduce extraction cost — never authoritative for evidence offsets, and not persisted to the graph (extraction-pipeline-internal only). |
| `Mention` | `mention_id, segment_id, char_start, char_end, surface_text, normalized_surface, entity_type, resolved_entity_id, resolution_status` | Self-validates `0 <= char_start < char_end`; `mention_within_segment()` is a separate cross-entity check against the actual `TranscriptSegment`. |
| `SpeakerResolution` | `resolution_id, conversation_id, speaker_label, resolved_contact_id, resolved_seller_id, role, evidence` | Failed resolution still produces a record (`role=UNKNOWN`) — never silently dropped. |

## Sales knowledge entities (`src/domain/knowledge.py`)

`Product`, `Feature`, `Objection`, `PainPoint`, `Blocker`, `BuyingSignal`,
`ActionItem`, `Commitment`, `ContentAsset` (carries `division_id` and `tags` —
`tags` is the curated mapping surface the recommendation use case matches
against, see `entity-resolution.md`), `Share`, `AssetView`.

Note: in the implemented extraction pipeline, an objection **claim**
(`Claim.predicate == "RAISED_OBJECTION"`) is what actually gets extracted and
persisted — a separate `Objection` graph node is not currently materialized or
repository-backed. `Objection` the Pydantic model exists in the domain package
for API/future use but has no corresponding repository yet.

## Assertion and audit entities (`src/domain/assertion.py`)

`Claim` — the full §6 field list, with three `model_validator`s: exactly one of
`object_id`/`object_value`; `evidence_char_end > evidence_char_start`;
`0.0 <= confidence <= 1.0`. `ExtractionRun`, `ResolutionDecision` (the full
component-score breakdown for one Mention), `ReviewDecision` (reviewer
identity, candidate set shown, original scores, affected Claims, previous
decision if overridden), `Conflict`, `ErasureEvent`.

## Stakeholder assignment (`src/domain/stakeholder.py`)

`StakeholderAssignment` — the materialized-view record for
`(Opportunity)-[:HAS_ASSIGNMENT]->(StakeholderAssignment)-[:ASSIGNS]->(Contact)`.
Source-specific role/influence/sentiment/authority remain Claims with
provenance; this is the current-best-view snapshot, not the source of truth.

## Graph edges materialized by the repository layer

Beyond the property-level foreign keys above, these edges are actually written
by `src/graph/repositories/*.py` (not modeled as Pydantic fields — edges are a
graph/repository concern, per §10's routing principle):

- `(Conversation)-[:HAS_SEGMENT]->(TranscriptSegment)`
- `(Conversation)-[:HAS_PARTICIPANT]->(Participant)`
- `(Conversation)-[:HAS_SPEAKER_RESOLUTION]->(SpeakerResolution)`
- `(TranscriptSegment)-[:HAS_CLAIM]->(Claim)` — the routing path
  `list_claims_for_conversation` traverses; a Claim's `source_segment_id`
  property alone is never used for tenant-scoped lookup.
- `(Mention)-[:HAS_RESOLUTION_DECISION]->(ResolutionDecision)`
- `(Mention)-[:HAS_REVIEW_DECISION]->(ReviewDecision)`
- `(SourceRecord)-[:HAS_SNAPSHOT]->(SourceSnapshot)`

## Ontology YAML (`config/ontologies/sales.yml`)

Consumed by the legacy `src/graph/domain_ontology.py` / `ontology_registry.py`
(kept working, not yet wired into a live `OntologyRegistry.load()` call site).
Defines `type_hierarchy`, and `relation_rules` including
`ADDRESSES_OBJECTION` (domain `[CONTENT_ASSET, PRODUCT]`, target `[OBJECTION]`)
— the ontology-level documentation of the same curated-tag mapping
`src/usecases/objection_content_recommendation.py` implements at the data
level. Replaces a leftover ad-tech-industry (advertiser/campaign/publisher)
template that was in this file from the initial project scaffold.
