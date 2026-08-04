"""§10 — this repo's own composite/full-text/versioned-vector indexes.

This is NOT a fork of the legacy neo4j_client.py's init_schema() (that encodes
graphrag's own Document/Chunk/Community RAG-ingestion shape, entirely irrelevant
here — see Increment 1's fork-scope notes). Every statement below targets this
repo's own sales-domain labels.

(workspace_id, external_id) from §10's index list is deliberately not created:
every CRM entity id in this repo (account_id, contact_id, ...) is itself
`crm_entity_id(workspace, source_system, object_type, external_id)` — a
deterministic hash of external_id — so looking an entity up by external_id never
needs an index scan; the caller just recomputes the hash and looks up by id
directly (see src/domain/identity.py). Add a denormalized external_id property +
index only if a future phase needs raw-external-id lookup without the
source_system on hand to recompute the hash.
"""

from __future__ import annotations

INDEX_STATEMENTS: list[str] = [
    # (workspace_id, id) per label — id lookups are the hot path for every repository.
    "CREATE INDEX account_workspace_id IF NOT EXISTS FOR (n:Account) ON (n.workspace_id, n.account_id)",
    "CREATE INDEX contact_workspace_id IF NOT EXISTS FOR (n:Contact) ON (n.workspace_id, n.contact_id)",
    "CREATE INDEX opportunity_workspace_id IF NOT EXISTS FOR (n:Opportunity) ON (n.workspace_id, n.opportunity_id)",
    "CREATE INDEX conversation_workspace_id IF NOT EXISTS FOR (n:Conversation) ON (n.workspace_id, n.conversation_id)",
    "CREATE INDEX claim_workspace_id IF NOT EXISTS FOR (n:Claim) ON (n.workspace_id, n.claim_id)",
    "CREATE INDEX mention_workspace_id IF NOT EXISTS FOR (n:Mention) ON (n.workspace_id, n.mention_id)",
    "CREATE INDEX content_asset_workspace_id IF NOT EXISTS FOR (n:ContentAsset) ON (n.workspace_id, n.content_asset_id)",
    # (workspace_id, canonical_name) — candidate-generation prefix lookups (P4).
    "CREATE INDEX account_workspace_name IF NOT EXISTS FOR (n:Account) ON (n.workspace_id, n.name)",
    "CREATE INDEX contact_workspace_name IF NOT EXISTS FOR (n:Contact) ON (n.workspace_id, n.name)",
    # (workspace_id, normalized_email) — Contact repository denormalizes the
    # Pydantic-computed normalized_email onto the node at write time (P4/P2).
    "CREATE INDEX contact_workspace_normalized_email IF NOT EXISTS FOR (n:Contact) ON (n.workspace_id, n.normalized_email)",
    # (workspace_id, resolution_status)
    "CREATE INDEX mention_workspace_resolution_status IF NOT EXISTS FOR (n:Mention) ON (n.workspace_id, n.resolution_status)",
    # (workspace_id, started_at) — this repo's Conversation field is `occurred_at`.
    "CREATE INDEX conversation_workspace_occurred_at IF NOT EXISTS FOR (n:Conversation) ON (n.workspace_id, n.occurred_at)",
    # (workspace_id, adjudication_status, is_superseded)
    "CREATE INDEX claim_workspace_adjudication IF NOT EXISTS FOR (n:Claim) ON (n.workspace_id, n.adjudication_status, n.is_superseded)",
]

FULLTEXT_STATEMENTS: list[str] = [
    "CREATE FULLTEXT INDEX account_contact_names IF NOT EXISTS "
    "FOR (n:Account|Contact) ON EACH [n.name]",
]

# Versioned per §10 ("Use versioned vector-index names and support backfill plus
# temporary dual-read during model migration"). Dimension 1536 is a provisional
# placeholder — no embedding provider is pinned yet (pyproject.toml's own open
# item), so this index exists but stays unpopulated until that's decided.
VECTOR_STATEMENTS: list[str] = [
    "CREATE VECTOR INDEX contact_embeddings_v1 IF NOT EXISTS "
    "FOR (n:Contact) ON n.embedding "
    "OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}",
]

ALL_STATEMENTS: list[str] = [*INDEX_STATEMENTS, *FULLTEXT_STATEMENTS, *VECTOR_STATEMENTS]

# Names as reported by `SHOW INDEXES YIELD name` — used by the readiness check.
ALL_INDEX_NAMES: list[str] = [
    "account_workspace_id",
    "contact_workspace_id",
    "opportunity_workspace_id",
    "conversation_workspace_id",
    "claim_workspace_id",
    "mention_workspace_id",
    "content_asset_workspace_id",
    "account_workspace_name",
    "contact_workspace_name",
    "contact_workspace_normalized_email",
    "mention_workspace_resolution_status",
    "conversation_workspace_occurred_at",
    "claim_workspace_adjudication",
    "account_contact_names",
    "contact_embeddings_v1",
]
