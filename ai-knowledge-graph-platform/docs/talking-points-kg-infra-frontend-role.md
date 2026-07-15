# Talking Points — Knowledge Graph Developer (Infrastructure/Frontend Focus, Metaphactory/AWS)

Prep notes for a recruiter/hiring-manager call on this specific JD. Goal: be
upfront about the two real gaps early, then spend the rest of the call on
genuine overlap so the conversation doesn't stay anchored on what's missing.

## The gaps — say this first, don't let them discover it

- **Metaphactory**: "I haven't used Metaphactory specifically. My knowledge
  graph work is on Neo4j — building the platform, not configuring a
  commercial KG frontend on top of one. If Metaphactory is a hard
  requirement for week one, I'm not your candidate. If there's ramp time,
  the underlying concepts (SPARQL, RDF, ontology-driven UI, graph
  exploration views) transfer directly — I'd just need onboarding on the
  product itself."
- **AWS as platform infrastructure**: "My AWS exposure is SageMaker for ML
  workloads, not EC2/ECS/networking/IAM-level platform ops. I've run
  production infra on GCP Cloud Run and Fly.io with the same DevOps
  discipline — Docker, CI/CD, observability — so the operational instincts
  are there, the AWS-specific tooling isn't yet."

Naming these unprompted signals honesty and self-awareness — better than
having the interviewer find it in a follow-up question.

## What genuinely overlaps — lead with these once gaps are on the table

1. **Semantic web stack is real, not padding.** rdflib + owlrl + pyshacl are
   live in the codebase: OWL ontology export, SPARQL query bridge, and a
   SHACL validator with real shapes (entity label/type completeness, axiom
   reification, confidence-range constraints) running against 49K+ live
   triples. Can screen-share `scripts/export_rdf.py --validate` and the
   passing SHACL report on request.
2. **Neo4j → Neptune is a narrower gap than it looks.** Cypher queries in
   this project are openCypher-standard — the same query language Neptune
   speaks. The gap is Neptune's AWS-specific deployment/ops model, not the
   query layer.
3. **Frontend is real, not theoretical.** Built a tabbed demo UI (Chat /
   Knowledge Graph) with vanilla HTML5/CSS/JS, iframe-embedded interactive
   graph visualization, tenant-aware routing — exactly the "dashboards,
   search views, application components" language in the JD, just not
   inside Metaphactory's component model.
4. **DevOps/API/system integration — full match.** Docker, CI/CD, Git,
   FastAPI, gRPC, WebSocket, RabbitMQ/Kafka/Redis integration. This is the
   deepest overlap in the JD and worth the most airtime.
5. **Product Owner collaboration.** Every project in the portfolio was
   spec'd, scoped, and delivered solo end-to-end — translating a fuzzy
   requirement into a shipped feature is the daily default, not a novel
   skill to demonstrate.

## Questions to ask them (shows engagement, also surfaces real fit)

- "How much of the role is Metaphactory configuration vs. building custom
  KG tooling around it? Is there ramp time for someone new to the product?"
- "Is the AWS work closer to 'deploy and maintain what's there' or
  'design the infrastructure from scratch'?" (Signals whether SageMaker +
  Cloud Run/Fly.io experience is close enough or a real stretch.)
- "Is Neptune already the graph store, or is that aspirational/roadmap?"

## One-line close

"I'm not a Metaphactory developer today, but I am a knowledge-graph and
semantic-web engineer who ships production RDF/OWL/SHACL pipelines and
builds the frontends on top of them — the product-specific tooling is the
smallest part of this role to learn."
