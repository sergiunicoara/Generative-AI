-- ── Core node constraints ──────────────────────────────────────────────────────
CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE;
-- NOTE: if upgrading from a single-tenant schema, run first:
--   DROP CONSTRAINT entity_name_type IF EXISTS;
-- The new constraint scopes entity identity per tenant.
CREATE CONSTRAINT entity_name_type_tenant IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type, e.tenant) IS UNIQUE;
CREATE INDEX entity_tenant IF NOT EXISTS FOR (e:Entity) ON (e.tenant);
CREATE CONSTRAINT community_id IF NOT EXISTS FOR (c:Community) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT alias_id IF NOT EXISTS FOR (a:Alias) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT changelog_id IF NOT EXISTS FOR (c:ChangeLog) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT canonical_part_id IF NOT EXISTS FOR (p:CanonicalPart) REQUIRE p.part_number IS UNIQUE;

-- ── Vector indexes ─────────────────────────────────────────────────────────────
CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
  FOR (c:Chunk) ON (c.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}};
CREATE VECTOR INDEX community_embeddings IF NOT EXISTS
  FOR (c:Community) ON (c.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}};
CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
  FOR (e:Entity) ON (e.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}};

-- ── Property indexes ──────────────────────────────────────────────────────────
CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX chunk_doc IF NOT EXISTS FOR (c:Chunk) ON (c.document_id);
CREATE INDEX entity_source_type IF NOT EXISTS FOR (e:Entity) ON (e.source_type);
CREATE INDEX relation_source_type IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.source_type);
CREATE INDEX relation_valid_from IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.valid_from);
CREATE INDEX doc_authority IF NOT EXISTS FOR (d:Document) ON (d.authority_level);

-- ── Fulltext indexes ───────────────────────────────────────────────────────────
CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text];
CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description];
CREATE FULLTEXT INDEX alias_fulltext IF NOT EXISTS FOR (a:Alias) ON EACH [a.value];

-- ── New node constraints (gaps implementation) ────────────────────────────────
CREATE CONSTRAINT conflict_id IF NOT EXISTS FOR (c:Conflict) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT ontology_version_id IF NOT EXISTS FOR (o:OntologyVersion) REQUIRE o.id IS UNIQUE;
CREATE CONSTRAINT ontology_migration_id IF NOT EXISTS FOR (m:OntologyMigration) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT community_snapshot_id IF NOT EXISTS FOR (s:CommunitySnapshot) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT quarantine_log_id IF NOT EXISTS FOR (q:QuarantineLog) REQUIRE q.id IS UNIQUE;
CREATE CONSTRAINT graph_health_snapshot_id IF NOT EXISTS FOR (h:GraphHealthSnapshot) REQUIRE h.id IS UNIQUE;

-- ── Tenant + quarantine indexes ───────────────────────────────────────────────
CREATE INDEX chunk_tenant IF NOT EXISTS FOR (c:Chunk) ON (c.tenant);
CREATE INDEX doc_tenant IF NOT EXISTS FOR (d:Document) ON (d.tenant);
CREATE INDEX entity_quarantined IF NOT EXISTS FOR (e:Entity) ON (e.quarantined);
CREATE INDEX entity_source_doc IF NOT EXISTS FOR (e:Entity) ON (e.source_doc_id);
CREATE INDEX relation_valid_to IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.valid_to);
CREATE INDEX relation_source_doc_ids IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.source_doc_ids);
CREATE INDEX relation_tenant IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.tenant);
CREATE INDEX conflict_status IF NOT EXISTS FOR (c:Conflict) ON (c.status);
CREATE INDEX conflict_tenant IF NOT EXISTS FOR (c:Conflict) ON (c.tenant);
