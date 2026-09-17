// Competency question: "What consent requirement applies to a campaign
// that complies with {{REGULATION}}?" -- multi-hop, exercises
// jurisdiction-scoped policy relations: COMPLIES_WITH and REQUIRES_CONSENT
// share CAMPAIGN as a common domain (config/ontologies/marketing_adtech.yml).
MATCH (campaign:Entity {tenant: '{{TENANT}}'})-[:RELATES_TO {relation: 'COMPLIES_WITH'}]->(:Entity {tenant: '{{TENANT}}', name: '{{REGULATION}}', type: 'PRIVACY_REGULATION'})
MATCH (campaign)-[:RELATES_TO {relation: 'REQUIRES_CONSENT'}]->(signal:Entity {tenant: '{{TENANT}}'})
RETURN DISTINCT campaign.name AS campaign, signal.name AS consentSignal
