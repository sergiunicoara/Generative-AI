"""Re-run forward-chaining inference on the existing graph — no LLM calls needed."""
import asyncio
from pathlib import Path


async def main():
    from graphrag.graph.neo4j_client import get_neo4j
    from graphrag.graph.inference_engine import ForwardChainingEngine, InferenceRule
    from graphrag.graph.domain_ontology import load_domain_ontology

    neo4j = get_neo4j()
    ontology = load_domain_ontology(Path("config/ontologies/aerospace_regulatory.yml"))

    engine = ForwardChainingEngine(neo4j)
    for rule_cfg in ontology.get("inference_rules", []):
        engine.add_rule(InferenceRule(
            name=rule_cfg["name"],
            rule_type=rule_cfg["rule_type"],
            relation=rule_cfg["relation"],
            derived_relation=rule_cfg.get("derived_relation", ""),
            body_relation_2=rule_cfg.get("body_relation_2", ""),
            max_depth=rule_cfg.get("max_depth", 3),
            confidence_decay=rule_cfg.get("confidence_decay", 0.9),
        ))

    print("Running forward-chaining inference on aerospace tenant...")
    report = await engine.run(tenant="aerospace")
    print(f"  Total inferred edges written: {report['total_inferred']}")
    print(f"  By rule: {report['by_rule']}")

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
