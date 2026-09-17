// Competency question: "Which regulation body mandated {{DIRECTIVE}}?"
// Exercises MANDATED_BY and its mandated_by_inverse rule
// (config/ontologies/aerospace_regulatory.yml).
MATCH (directive:Entity {tenant: '{{TENANT}}', name: '{{DIRECTIVE}}'})-[:RELATES_TO {relation: 'MANDATED_BY'}]->(authority:Entity {tenant: '{{TENANT}}'})
RETURN authority.name AS mandatingAuthority, authority.type AS authorityType
