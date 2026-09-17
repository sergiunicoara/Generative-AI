// Competency question: "What is {{SUPPLIER}} classified as, and what does
// that classification imply?" -- exercises CLASSIFIED_AS
// (config/ontologies/automotive_iatf.yml), grounding MH-01 in
// data/eval_golden/queries_automotive.json ("consequences for a PlastiAuto
// supplier classified as CRITIC").
MATCH (supplier:Entity {tenant: '{{TENANT}}', name: '{{SUPPLIER}}'})-[:RELATES_TO {relation: 'CLASSIFIED_AS'}]->(classification:Entity {tenant: '{{TENANT}}'})
RETURN classification.name AS classification, classification.type AS classificationType
