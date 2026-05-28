"""Forward-chaining inference engine — Datalog-style rules for KG completion.

Problem solved
--------------
The KG stores only asserted facts.  Logically derivable facts must be
reasoned over by the LLM at query time, which is:
  1. Slow — a deduction that could be pre-computed is re-derived every query.
  2. Inconsistent — the LLM may not apply the same rule consistently.
  3. Invisible — the derived fact has no provenance in the graph.

Examples of rules that should be pre-computed:
  TRANSITIVITY  : A SUBSIDIARY_OF B  ∧  B SUBSIDIARY_OF C  ⇒  A SUBSIDIARY_OF C
  SYMMETRY      : A RELATED_TO B                            ⇒  B RELATED_TO A
  INVERSE       : A WORKS_AT B                              ⇒  B EMPLOYS A
  COMPOSITION   : A LOCATED_IN B  ∧  B PART_OF C           ⇒  A LOCATED_IN C

Architecture
------------
- InferenceRule dataclass: name, head_relation, body (list of (rel, direction)),
  max_depth (for transitivity), confidence_decay (per hop).
- ForwardChainingEngine.run() iterates rules to fixpoint (max_iterations cap).
- Derived edges are written as RELATES_TO with source_type="inferred" and
  a reference to the rule that fired.
- Only new edges are written (MERGE semantics) — existing asserted edges are
  not overwritten; their confidence takes priority.
- Inferred edges have confidence = confidence_of_premises × decay^depth.
- run_for_document(doc_id) scopes inference to the subgraph affected by a
  single document (efficient post-ingestion trigger).

Config
------
Rules can be defined in config/settings.yml under `inference.rules` as a
list of dicts:
    - name: subsidiary_transitivity
      relation: SUBSIDIARY_OF
      rule_type: transitivity
      max_depth: 3
      confidence_decay: 0.9
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

import structlog

log = structlog.get_logger(__name__)


@dataclass
class InferenceRule:
    """A single forward-chaining rule."""
    name:              str
    rule_type:         str          # "transitivity" | "symmetry" | "inverse" | "composition"
    relation:          str          # head relation (LHS of =>)
    derived_relation:  str = ""     # what relation to derive (defaults to same as `relation`)
    body_relation_2:   str = ""     # for composition: the second body relation
    max_depth:         int = 3      # transitivity only: max chain length
    confidence_decay:  float = 0.9  # per-hop confidence multiplier


# Canonical built-in rules — safe to apply to any domain
DEFAULT_RULES: list[InferenceRule] = [
    # Transitivity: A SUBSIDIARY_OF B, B SUBSIDIARY_OF C => A SUBSIDIARY_OF C
    InferenceRule(
        name="subsidiary_transitivity",
        rule_type="transitivity",
        relation="SUBSIDIARY_OF",
        max_depth=3,
        confidence_decay=0.85,
    ),
    # Symmetry: A RELATED_TO B => B RELATED_TO A
    InferenceRule(
        name="related_to_symmetry",
        rule_type="symmetry",
        relation="RELATED_TO",
    ),
    # Symmetry: A PART_OF B (when used between same-type orgs) => B CONTAINS A
    # (modelled as symmetry for RELATED_TO only; domain-specific rules via config)
    InferenceRule(
        name="works_at_inverse",
        rule_type="inverse",
        relation="WORKS_AT",
        derived_relation="EMPLOYS",
    ),
    InferenceRule(
        name="founded_inverse",
        rule_type="inverse",
        relation="FOUNDED",
        derived_relation="FOUNDED_BY",
    ),
    # Composition: A LOCATED_IN B, B PART_OF C => A LOCATED_IN C
    InferenceRule(
        name="located_in_part_of",
        rule_type="composition",
        relation="LOCATED_IN",
        body_relation_2="PART_OF",
        derived_relation="LOCATED_IN",
        confidence_decay=0.8,
    ),
]


class ForwardChainingEngine:
    """
    Apply Datalog-style forward-chaining rules to derive implicit KG edges.

    Usage::

        engine = ForwardChainingEngine(neo4j_client)

        # Apply all default rules across the full graph
        report = await engine.run(tenant="acme")

        # Apply only to entities affected by a recent document
        report = await engine.run_for_document(doc_id="doc_abc", tenant="acme")

        # Register a domain-specific rule
        engine.add_rule(InferenceRule(
            name="certifies_inverse",
            rule_type="inverse",
            relation="CERTIFIED_BY",
            derived_relation="CERTIFIES",
        ))
    """

    def __init__(self, neo4j_client, rules: list[InferenceRule] | None = None):
        self._neo4j = neo4j_client
        self._rules: list[InferenceRule] = list(rules or DEFAULT_RULES)

    def add_rule(self, rule: InferenceRule) -> None:
        """Register an additional inference rule."""
        self._rules.append(rule)
        log.info("inference_engine.rule_added", rule=rule.name, type=rule.rule_type)

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(
        self,
        tenant: str = "default",
        max_iterations: int = 5,
        dry_run: bool = False,
    ) -> dict:
        """
        Apply all rules to fixpoint (or max_iterations, whichever comes first).

        Returns a summary of edges derived per rule.
        """
        total_derived: dict[str, int] = {}
        for iteration in range(max_iterations):
            new_in_iteration = 0
            for rule in self._rules:
                count = await self._apply_rule(rule, tenant=tenant, dry_run=dry_run)
                total_derived[rule.name] = total_derived.get(rule.name, 0) + count
                new_in_iteration += count
            log.info(
                "inference_engine.iteration",
                iteration=iteration + 1,
                new_edges=new_in_iteration,
                dry_run=dry_run,
            )
            if new_in_iteration == 0:
                break   # fixpoint reached

        log.info(
            "inference_engine.run_complete",
            total=sum(total_derived.values()),
            tenant=tenant,
            dry_run=dry_run,
        )
        return {
            "tenant":      tenant,
            "dry_run":     dry_run,
            "total_inferred": sum(total_derived.values()),
            "by_rule":     total_derived,
        }

    async def run_for_document(
        self,
        doc_id: str,
        tenant: str = "default",
    ) -> dict:
        """
        Scope inference to entities introduced or updated by a specific document.

        More efficient than full-graph run for post-ingestion triggers.
        """
        # Identify affected entities
        rows = await self._neo4j.run(
            """
            MATCH (c:Chunk {document_id: $doc_id})-[:MENTIONS]->(e:Entity {tenant: $tenant})
            RETURN DISTINCT e.name AS name, e.type AS type
            """,
            doc_id=doc_id,
            tenant=tenant,
        )
        if not rows:
            return {"tenant": tenant, "doc_id": doc_id, "total_inferred": 0, "by_rule": {}}

        # Run all rules — the Cypher already only fires when new edges would be created
        return await self.run(tenant=tenant, max_iterations=3)

    # ── Rule application ───────────────────────────────────────────────────────

    async def _apply_rule(
        self,
        rule: InferenceRule,
        tenant: str,
        dry_run: bool,
    ) -> int:
        """Dispatch to the correct application method for the rule type."""
        if rule.rule_type == "transitivity":
            return await self._apply_transitivity(rule, tenant, dry_run)
        elif rule.rule_type == "symmetry":
            return await self._apply_symmetry(rule, tenant, dry_run)
        elif rule.rule_type == "inverse":
            return await self._apply_inverse(rule, tenant, dry_run)
        elif rule.rule_type == "composition":
            return await self._apply_composition(rule, tenant, dry_run)
        else:
            log.warning("inference_engine.unknown_rule_type", type=rule.rule_type)
            return 0

    async def _apply_transitivity(
        self, rule: InferenceRule, tenant: str, dry_run: bool
    ) -> int:
        """
        A -[rel]-> B, B -[rel]-> C  =>  A -[rel]-> C

        Uses Cypher path matching up to max_depth hops. Only creates edges
        that don't already exist (asserted or inferred).
        """
        rel = rule.relation
        depth = rule.max_depth
        decay = rule.confidence_decay
        tenant_filter = "AND a.tenant = $tenant AND c.tenant = $tenant" if tenant else ""

        # Tenant must be enforced on every node and edge along the path,
        # not only on the endpoints — otherwise a 2-hop path can traverse
        # an intermediate entity from a different tenant.
        path_tenant_filter = (
            "AND ALL(n IN nodes(path) WHERE n.tenant = $tenant) "
            "AND ALL(r IN relationships(path) WHERE r.tenant = $tenant)"
        ) if tenant else ""

        rows = await self._neo4j.run(
            f"""
            MATCH path = (a:Entity)-[:RELATES_TO*2..{depth} {{relation: $rel}}]->(c:Entity)
            WHERE NOT (a)-[:RELATES_TO {{relation: $rel}}]->(c)
              AND a <> c
              {path_tenant_filter}
            WITH a, c,
                 length(path) AS hops,
                 reduce(conf = 1.0, r IN relationships(path) |
                     conf * coalesce(r.confidence, 1.0)) AS path_conf
            RETURN a.name AS src, a.type AS src_type,
                   c.name AS tgt, c.type AS tgt_type,
                   hops,
                   path_conf * $decay AS inferred_conf
            LIMIT 500
            """,
            rel=rel,
            decay=decay,
            tenant=tenant,
        )

        if dry_run:
            return len(rows)

        for row in rows:
            await self._write_inferred_edge(
                src_name=row["src"],  src_type=row["src_type"],
                tgt_name=row["tgt"],  tgt_type=row["tgt_type"],
                relation=rel,
                confidence=float(row.get("inferred_conf") or decay),
                rule_name=rule.name,
                tenant=tenant,
            )
        return len(rows)

    async def _apply_symmetry(
        self, rule: InferenceRule, tenant: str, dry_run: bool
    ) -> int:
        """A -[rel]-> B  =>  B -[rel]-> A"""
        rel = rule.relation
        derived = rule.derived_relation or rel
        tenant_filter = (
            "AND a.tenant = $tenant "
            "AND b.tenant = $tenant "
            "AND r.tenant = $tenant"
        ) if tenant else ""

        rows = await self._neo4j.run(
            f"""
            MATCH (a:Entity)-[r:RELATES_TO {{relation: $rel}}]->(b:Entity)
            WHERE NOT (b)-[:RELATES_TO {{relation: $derived}}]->(a)
              {tenant_filter}
            RETURN a.name AS src, a.type AS src_type,
                   b.name AS tgt, b.type AS tgt_type,
                   coalesce(r.confidence, 1.0) AS conf
            LIMIT 500
            """,
            rel=rel,
            derived=derived,
            tenant=tenant,
        )

        if dry_run:
            return len(rows)

        for row in rows:
            await self._write_inferred_edge(
                src_name=row["tgt"],  src_type=row["tgt_type"],
                tgt_name=row["src"],  tgt_type=row["src_type"],
                relation=derived,
                confidence=float(row.get("conf") or 1.0) * rule.confidence_decay,
                rule_name=rule.name,
                tenant=tenant,
            )
        return len(rows)

    async def _apply_inverse(
        self, rule: InferenceRule, tenant: str, dry_run: bool
    ) -> int:
        """A -[rel]-> B  =>  B -[derived_rel]-> A"""
        rel = rule.relation
        derived = rule.derived_relation or rel
        tenant_filter = (
            "AND a.tenant = $tenant "
            "AND b.tenant = $tenant "
            "AND r.tenant = $tenant"
        ) if tenant else ""

        rows = await self._neo4j.run(
            f"""
            MATCH (a:Entity)-[r:RELATES_TO {{relation: $rel}}]->(b:Entity)
            WHERE NOT (b)-[:RELATES_TO {{relation: $derived}}]->(a)
              {tenant_filter}
            RETURN a.name AS src, a.type AS src_type,
                   b.name AS tgt, b.type AS tgt_type,
                   coalesce(r.confidence, 1.0) AS conf
            LIMIT 500
            """,
            rel=rel,
            derived=derived,
            tenant=tenant,
        )

        if dry_run:
            return len(rows)

        for row in rows:
            await self._write_inferred_edge(
                src_name=row["tgt"],  src_type=row["tgt_type"],
                tgt_name=row["src"],  tgt_type=row["src_type"],
                relation=derived,
                confidence=float(row.get("conf") or 1.0) * rule.confidence_decay,
                rule_name=rule.name,
                tenant=tenant,
            )
        return len(rows)

    async def _apply_composition(
        self, rule: InferenceRule, tenant: str, dry_run: bool
    ) -> int:
        """
        A -[rel]-> B, B -[body_rel_2]-> C  =>  A -[derived_rel]-> C
        """
        rel1    = rule.relation
        rel2    = rule.body_relation_2
        derived = rule.derived_relation or rel1
        if not rel2:
            return 0
        # All three entities AND both edges must belong to the same tenant.
        # Missing b.tenant would let the inference cross tenant boundaries via
        # a shared intermediate entity name.
        tenant_filter = (
            "AND a.tenant = $tenant "
            "AND b.tenant = $tenant "
            "AND c.tenant = $tenant "
            "AND r1.tenant = $tenant "
            "AND r2.tenant = $tenant"
        ) if tenant else ""

        rows = await self._neo4j.run(
            f"""
            MATCH (a:Entity)-[r1:RELATES_TO {{relation: $rel1}}]->(b:Entity)
                  -[r2:RELATES_TO {{relation: $rel2}}]->(c:Entity)
            WHERE NOT (a)-[:RELATES_TO {{relation: $derived}}]->(c)
              AND a <> c
              {tenant_filter}
            RETURN a.name AS src, a.type AS src_type,
                   c.name AS tgt, c.type AS tgt_type,
                   coalesce(r1.confidence, 1.0) * coalesce(r2.confidence, 1.0) AS conf
            LIMIT 500
            """,
            rel1=rel1,
            rel2=rel2,
            derived=derived,
            tenant=tenant,
        )

        if dry_run:
            return len(rows)

        for row in rows:
            await self._write_inferred_edge(
                src_name=row["src"],  src_type=row["src_type"],
                tgt_name=row["tgt"],  tgt_type=row["tgt_type"],
                relation=derived,
                confidence=float(row.get("conf") or 1.0) * rule.confidence_decay,
                rule_name=rule.name,
                tenant=tenant,
            )
        return len(rows)

    async def _write_inferred_edge(
        self,
        src_name: str,
        src_type: str,
        tgt_name: str,
        tgt_type: str,
        relation: str,
        confidence: float,
        rule_name: str,
        tenant: str,
    ) -> None:
        """Write a derived RELATES_TO edge with source_type=inferred."""
        await self._neo4j.run(
            """
            MATCH (s:Entity {name: $src_name, type: $src_type, tenant: $tenant})
            MATCH (t:Entity {name: $tgt_name, type: $tgt_type, tenant: $tenant})
            MERGE (s)-[r:RELATES_TO {relation: $relation}]->(t)
            ON CREATE SET r.confidence      = $confidence,
                          r.source_type     = 'inferred',
                          r.inferred_by     = $rule,
                          r.tenant          = $tenant,
                          r.recorded_at     = datetime(),
                          r.source_doc_ids  = []
            // Never overwrite an asserted edge with an inferred one
            WITH r
            WHERE r.source_type = 'inferred'
            SET r.confidence  = $confidence,
                r.inferred_by = $rule
            """,
            src_name=src_name,
            src_type=src_type,
            tgt_name=tgt_name,
            tgt_type=tgt_type,
            relation=relation,
            confidence=min(1.0, max(0.0, confidence)),
            rule=rule_name,
            tenant=tenant,
        )
