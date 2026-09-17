// Competency question: "Which ad categories are strictly excluded from
// {{ADVERTISER}} campaigns?" -- exercises EXCLUDES_CATEGORY and negative-
// knowledge modeling (config/ontologies/marketing_adtech.yml). GOVERNS and
// EXCLUDES_CATEGORY share the same governing-document domain
// (STATEMENT_OF_WORK / DATA_PRIVACY_POLICY / BRAND_SAFETY_POLICY), so a
// policy document governing this advertiser is the join point.
MATCH (policy:Entity {tenant: '{{TENANT}}'})-[:RELATES_TO {relation: 'GOVERNS'}]->(:Entity {tenant: '{{TENANT}}', name: '{{ADVERTISER}}', type: 'ADVERTISER'})
MATCH (policy)-[:RELATES_TO {relation: 'EXCLUDES_CATEGORY'}]->(category:Entity {tenant: '{{TENANT}}'})
RETURN DISTINCT category.name AS excludedCategory
