# DO NOT EDIT — generated from automotive-iatf-quality.yaml.
// Neo4j schema can enforce keys/indexes; see diagnostics.json for runtime-only semantics.
CREATE CONSTRAINT energy_automotive_component_partnumber_unique IF NOT EXISTS FOR (n:AUTOMOTIVE_COMPONENT) REQUIRE n.partNumber IS UNIQUE;
CREATE CONSTRAINT energy_client_requirement_documentid_unique IF NOT EXISTS FOR (n:CLIENT_REQUIREMENT) REQUIRE n.documentId IS UNIQUE;
CREATE CONSTRAINT energy_form_record_documentid_unique IF NOT EXISTS FOR (n:FORM_RECORD) REQUIRE n.documentId IS UNIQUE;
CREATE CONSTRAINT energy_internal_procedure_documentid_unique IF NOT EXISTS FOR (n:INTERNAL_PROCEDURE) REQUIRE n.documentId IS UNIQUE;
CREATE CONSTRAINT energy_kpi_name_unique IF NOT EXISTS FOR (n:KPI) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT energy_quality_document_documentid_unique IF NOT EXISTS FOR (n:QUALITY_DOCUMENT) REQUIRE n.documentId IS UNIQUE;
CREATE CONSTRAINT energy_quality_manual_documentid_unique IF NOT EXISTS FOR (n:QUALITY_MANUAL) REQUIRE n.documentId IS UNIQUE;
CREATE CONSTRAINT energy_standard_standardid_unique IF NOT EXISTS FOR (n:STANDARD) REQUIRE n.standardId IS UNIQUE;
CREATE CONSTRAINT energy_supplier_supplierid_unique IF NOT EXISTS FOR (n:SUPPLIER) REQUIRE n.supplierId IS UNIQUE;
