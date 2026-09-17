// Competency question: "What aircraft type does {{DIRECTIVE}} apply to?"
// Exercises APPLIES_TO domain/range (AIRWORTHINESS_DIRECTIVE -> AIRCRAFT_TYPE,
// config/ontologies/aerospace_regulatory.yml).
MATCH (directive:Entity {tenant: '{{TENANT}}', name: '{{DIRECTIVE}}'})-[:RELATES_TO {relation: 'APPLIES_TO'}]->(target:Entity {tenant: '{{TENANT}}', type: 'AIRCRAFT_TYPE'})
RETURN target.name AS aircraftType
