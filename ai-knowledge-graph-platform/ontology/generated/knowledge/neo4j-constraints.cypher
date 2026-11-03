# DO NOT EDIT — generated from contextual-knowledge.yaml.
// Neo4j schema can enforce keys/indexes; see diagnostics.json for runtime-only semantics.
CREATE CONSTRAINT energy_knowledge_exception_knowledgekey_unique IF NOT EXISTS FOR (n:KNOWLEDGE_EXCEPTION) REQUIRE n.knowledgeKey IS UNIQUE;
CREATE CONSTRAINT energy_knowledge_exception_tenant_knowledgeid_version_unique IF NOT EXISTS FOR (n:KNOWLEDGE_EXCEPTION) REQUIRE (n.tenant, n.knowledgeId, n.version) IS UNIQUE;
CREATE CONSTRAINT energy_knowledge_item_knowledgekey_unique IF NOT EXISTS FOR (n:KNOWLEDGE_ITEM) REQUIRE n.knowledgeKey IS UNIQUE;
CREATE CONSTRAINT energy_knowledge_item_tenant_knowledgeid_version_unique IF NOT EXISTS FOR (n:KNOWLEDGE_ITEM) REQUIRE (n.tenant, n.knowledgeId, n.version) IS UNIQUE;
CREATE CONSTRAINT energy_knowledge_release_releaseid_unique IF NOT EXISTS FOR (n:KNOWLEDGE_RELEASE) REQUIRE n.releaseId IS UNIQUE;
CREATE CONSTRAINT energy_knowledge_source_sourceid_unique IF NOT EXISTS FOR (n:KNOWLEDGE_SOURCE) REQUIRE n.sourceId IS UNIQUE;
