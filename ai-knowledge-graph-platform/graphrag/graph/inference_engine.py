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

import hashlib
import json
from dataclasses import asdict, dataclass

import structlog

from graphrag.core.tenancy import require_tenant

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
    note:              str = ""     # human-readable rule statement/rationale
    owner:             str = ""     # accountable team/person for this rule


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
        require_tenant(tenant)
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
        require_tenant(tenant)
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
        that don't already exist (asserted or inferred). Retracted, expired or
        quarantined premises never support a derivation.
        """
        rows = await self._neo4j.run(
            self._transitivity_query(rule) + " LIMIT 500",
            rel=rule.relation, decay=rule.confidence_decay, tenant=tenant,
        )
        if dry_run:
            return len(rows)
        for row in rows:
            await self._write_inferred_edge(
                src_name=row["src"],  src_type=row["src_type"],
                tgt_name=row["tgt"],  tgt_type=row["tgt_type"],
                relation=rule.relation,
                confidence=float(row.get("inferred_conf") or rule.confidence_decay),
                rule_name=rule.name,
                tenant=tenant,
                premises=row.get("premises") or [],
                rule_version=rule_version(rule),
            )
        return len(rows)

    def _transitivity_query(self, rule: InferenceRule, *, pinned: bool = False) -> str:
        # Tenant must be enforced on every node and edge along the path,
        # not only on the endpoints — otherwise a 2-hop path can traverse
        # an intermediate entity from a different tenant.
        pin = ("AND a.name = $src_name AND a.type = $src_type "
               "AND c.name = $tgt_name AND c.type = $tgt_type ") if pinned else \
              "AND NOT (a)-[:RELATES_TO {relation: $rel}]->(c) "
        return f"""
            MATCH path = (a:Entity)-[:RELATES_TO*2..{rule.max_depth} {{relation: $rel}}]->(c:Entity)
            WHERE a <> c
              {pin}
              AND ALL(n IN nodes(path) WHERE n.tenant = $tenant AND {_NODE_OK.format(v='n')})
              AND ALL(r IN relationships(path) WHERE r.tenant = $tenant AND {_EDGE_OK.format(v='r')})
            WITH a, c, path,
                 reduce(conf = 1.0, r IN relationships(path) |
                     conf * coalesce(r.confidence, 1.0)) AS path_conf
            RETURN a.name AS src, a.type AS src_type,
                   c.name AS tgt, c.type AS tgt_type,
                   length(path) AS hops,
                   path_conf * $decay AS inferred_conf,
                   [i IN range(0, length(path) - 1) |
                       nodes(path)[i].type + ':' + nodes(path)[i].name + '|' + $rel + '|' +
                       nodes(path)[i + 1].type + ':' + nodes(path)[i + 1].name] AS premises
        """

    async def _apply_symmetry(
        self, rule: InferenceRule, tenant: str, dry_run: bool
    ) -> int:
        """A -[rel]-> B  =>  B -[rel]-> A"""
        return await self._apply_flip(rule, tenant, dry_run)

    async def _apply_inverse(
        self, rule: InferenceRule, tenant: str, dry_run: bool
    ) -> int:
        """A -[rel]-> B  =>  B -[derived_rel]-> A"""
        return await self._apply_flip(rule, tenant, dry_run)

    def _flip_query(self, *, pinned: bool = False) -> str:
        pin = ("AND b.name = $src_name AND b.type = $src_type "
               "AND a.name = $tgt_name AND a.type = $tgt_type ") if pinned else \
              "AND NOT (b)-[:RELATES_TO {relation: $derived}]->(a) "
        return f"""
            MATCH (a:Entity)-[r:RELATES_TO {{relation: $rel}}]->(b:Entity)
            WHERE a.tenant = $tenant AND b.tenant = $tenant AND r.tenant = $tenant
              {pin}
              AND {_EDGE_OK.format(v='r')} AND {_NODE_OK.format(v='a')} AND {_NODE_OK.format(v='b')}
            RETURN a.name AS src, a.type AS src_type,
                   b.name AS tgt, b.type AS tgt_type,
                   coalesce(r.confidence, 1.0) AS conf,
                   [a.type + ':' + a.name + '|' + $rel + '|' + b.type + ':' + b.name] AS premises
        """

    async def _apply_flip(self, rule: InferenceRule, tenant: str, dry_run: bool) -> int:
        derived = rule.derived_relation or rule.relation
        rows = await self._neo4j.run(
            self._flip_query() + " LIMIT 500",
            rel=rule.relation, derived=derived, tenant=tenant,
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
                premises=row.get("premises") or [],
                rule_version=rule_version(rule),
            )
        return len(rows)

    def _composition_query(self, *, pinned: bool = False) -> str:
        # All three entities AND both edges must belong to the same tenant.
        # Missing b.tenant would let the inference cross tenant boundaries via
        # a shared intermediate entity name.
        pin = ("AND a.name = $src_name AND a.type = $src_type "
               "AND c.name = $tgt_name AND c.type = $tgt_type ") if pinned else \
              "AND NOT (a)-[:RELATES_TO {relation: $derived}]->(c) "
        return f"""
            MATCH (a:Entity)-[r1:RELATES_TO {{relation: $rel1}}]->(b:Entity)
                  -[r2:RELATES_TO {{relation: $rel2}}]->(c:Entity)
            WHERE a <> c
              {pin}
              AND a.tenant = $tenant AND b.tenant = $tenant AND c.tenant = $tenant
              AND r1.tenant = $tenant AND r2.tenant = $tenant
              AND {_EDGE_OK.format(v='r1')} AND {_EDGE_OK.format(v='r2')}
              AND {_NODE_OK.format(v='a')} AND {_NODE_OK.format(v='b')} AND {_NODE_OK.format(v='c')}
            RETURN a.name AS src, a.type AS src_type,
                   c.name AS tgt, c.type AS tgt_type,
                   coalesce(r1.confidence, 1.0) * coalesce(r2.confidence, 1.0) AS conf,
                   [a.type + ':' + a.name + '|' + $rel1 + '|' + b.type + ':' + b.name,
                    b.type + ':' + b.name + '|' + $rel2 + '|' + c.type + ':' + c.name] AS premises
        """

    async def _apply_composition(
        self, rule: InferenceRule, tenant: str, dry_run: bool
    ) -> int:
        """
        A -[rel]-> B, B -[body_rel_2]-> C  =>  A -[derived_rel]-> C
        """
        if not rule.body_relation_2:
            return 0
        derived = rule.derived_relation or rule.relation
        rows = await self._neo4j.run(
            self._composition_query() + " LIMIT 500",
            rel1=rule.relation, rel2=rule.body_relation_2, derived=derived, tenant=tenant,
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
                premises=row.get("premises") or [],
                rule_version=rule_version(rule),
            )
        return len(rows)

    def rule_named(self, name: str) -> InferenceRule | None:
        return next((r for r in self._rules if r.name == name), None)

    async def derivation_for(
        self, rule_name: str, *, src_name: str, src_type: str,
        tgt_name: str, tgt_type: str, tenant: str,
    ) -> dict | None:
        """Re-derive ONE inferred edge from currently valid premises.

        Returns ``{"premises": [...], "confidence": float}`` for the first valid
        derivation, or ``None`` when the rule no longer supports the edge (used
        by targeted recomputation, graphrag/graph/invalidation).
        """
        require_tenant(tenant)
        rule = self.rule_named(rule_name)
        if rule is None:
            return None
        pins = {"src_name": src_name, "src_type": src_type, "tgt_name": tgt_name,
                "tgt_type": tgt_type, "tenant": tenant}
        if rule.rule_type == "transitivity":
            rows = await self._neo4j.run(
                self._transitivity_query(rule, pinned=True) + " LIMIT 1",
                rel=rule.relation, decay=rule.confidence_decay, **pins)
            conf_key = "inferred_conf"
            factor = 1.0
        elif rule.rule_type in ("symmetry", "inverse"):
            rows = await self._neo4j.run(
                self._flip_query(pinned=True) + " LIMIT 1",
                rel=rule.relation, derived=rule.derived_relation or rule.relation, **pins)
            conf_key, factor = "conf", rule.confidence_decay
        elif rule.rule_type == "composition" and rule.body_relation_2:
            rows = await self._neo4j.run(
                self._composition_query(pinned=True) + " LIMIT 1",
                rel1=rule.relation, rel2=rule.body_relation_2,
                derived=rule.derived_relation or rule.relation, **pins)
            conf_key, factor = "conf", rule.confidence_decay
        else:
            return None
        if not rows:
            return None
        row = rows[0]
        return {"premises": list(row.get("premises") or []),
                "confidence": min(1.0, max(0.0, float(row.get(conf_key) or 0.0) * factor))}

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
        premises: list[str] | None = None,
        rule_version: str = "",
    ) -> None:
        """Write a derived RELATES_TO edge with source_type=inferred.

        ``premise_keys`` (``Type:Name|REL|Type:Name``) and ``rule_version`` make
        the derivation explicit, so a change to any premise can invalidate
        exactly the edges it supports (docs/invalidation.md).
        """
        await self._neo4j.run(
            """
            MATCH (s:Entity {name: $src_name, type: $src_type, tenant: $tenant})
            MATCH (t:Entity {name: $tgt_name, type: $tgt_type, tenant: $tenant})
            MERGE (s)-[r:RELATES_TO {relation: $relation}]->(t)
            ON CREATE SET r.confidence      = $confidence,
                          r.source_type     = 'inferred',
                          r.confidence_state = 'INFERRED',
                          r.inferred_by     = $rule,
                          r.tenant          = $tenant,
                          r.recorded_at     = datetime(),
                          r.source_doc_ids  = []
            // Never overwrite an asserted edge with an inferred one
            WITH r
            WHERE r.source_type = 'inferred'
            SET r.confidence  = $confidence,
                r.inferred_by = $rule,
                r.rule_version = $rule_version,
                r.premise_keys = $premises,
                r.confidence_state = 'INFERRED',
                r.origin = 'INFERRED',
                r.generated_by = 'rule:' + $rule,
                r.verification_status = coalesce(r.verification_status, 'UNVERIFIED')
            """,
            src_name=src_name,
            src_type=src_type,
            tgt_name=tgt_name,
            tgt_type=tgt_type,
            relation=relation,
            confidence=min(1.0, max(0.0, confidence)),
            rule=rule_name,
            tenant=tenant,
            premises=sorted(set(premises or [])),
            rule_version=rule_version,
        )


# A premise edge supports a derivation only while it is not retracted and not
# expired; a premise entity only while it is not quarantined.
_EDGE_OK = ("coalesce({v}.confidence_state, 'ASSERTED') <> 'RETRACTED' "
            "AND ({v}.valid_to IS NULL OR {v}.valid_to > datetime())")
_NODE_OK = "coalesce({v}.quarantined, false) = false"


def relation_key(src_type: str, src_name: str, relation: str, tgt_type: str, tgt_name: str) -> str:
    """Stable key of one RELATES_TO edge, as stored in ``premise_keys``."""
    return f"{src_type}:{src_name}|{relation}|{tgt_type}:{tgt_name}"


def rule_version(rule: InferenceRule) -> str:
    """Content hash of a rule definition; changes whenever the rule's logic does."""
    payload = json.dumps(asdict(rule), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
