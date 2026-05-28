-- ── Core node constraints ──────────────────────────────────────────────────────
CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT entity_name_type IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE;
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
