// Competency question: "At what frequency must {{SUPPLIER}} be
// reevaluated?" -- exercises REEVALUATED_AT
// (config/ontologies/automotive_iatf.yml), grounding the corpus's C03/C05
// ground-truth contradictions (reevaluation frequency: semestrial for
// CRITICAL suppliers vs. annual for general suppliers).
MATCH (supplier:Entity {tenant: '{{TENANT}}', name: '{{SUPPLIER}}'})-[:RELATES_TO {relation: 'REEVALUATED_AT'}]->(frequency:Entity {tenant: '{{TENANT}}', type: 'REEVALUATION_FREQUENCY'})
RETURN frequency.name AS reevaluationFrequency
