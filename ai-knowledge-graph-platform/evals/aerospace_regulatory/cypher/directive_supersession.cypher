// Competency question: "Which directive supersedes {{DIRECTIVE}}?"
// Exercises SUPERSEDES (config/ontologies/aerospace_regulatory.yml) and the
// supersedes_transitivity inference rule -- "what is the current authority
// on this component?"
MATCH (newer:Entity {tenant: '{{TENANT}}'})-[:RELATES_TO {relation: 'SUPERSEDES'}]->(older:Entity {tenant: '{{TENANT}}', name: '{{DIRECTIVE}}'})
RETURN newer.name AS supersedingDirective, newer.type AS supersedingType
