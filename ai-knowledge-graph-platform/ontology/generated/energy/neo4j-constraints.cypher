# DO NOT EDIT — generated from energy-asset-intelligence.yaml.
// Neo4j schema can enforce keys/indexes; see diagnostics.json for runtime-only semantics.
CREATE CONSTRAINT energy_asset_assetid_unique IF NOT EXISTS FOR (n:ASSET) REQUIRE n.assetId IS UNIQUE;
CREATE CONSTRAINT energy_document_revision_documentid_unique IF NOT EXISTS FOR (n:DOCUMENT_REVISION) REQUIRE n.documentId IS UNIQUE;
CREATE CONSTRAINT energy_evidence_request_requestid_unique IF NOT EXISTS FOR (n:EVIDENCE_REQUEST) REQUIRE n.requestId IS UNIQUE;
CREATE CONSTRAINT energy_wind_turbine_assetid_unique IF NOT EXISTS FOR (n:WIND_TURBINE) REQUIRE n.assetId IS UNIQUE;
CREATE CONSTRAINT energy_work_order_workorderid_unique IF NOT EXISTS FOR (n:WORK_ORDER) REQUIRE n.workOrderId IS UNIQUE;
