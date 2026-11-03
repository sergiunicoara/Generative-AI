"""Seeded synthetic knowledge packs and contexts for scale and property tests.

Nothing here is domain content: dimension values are opaque tokens (``M1``,
``C3`` ...), facts are numeric gauges, and every item is synthetic.  The same
seed always gives the same pack, so benchmarks and generated evaluations are
reproducible.  Used by ``scripts/benchmark_knowledge_eval.py`` and the
generated-context tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import random

from graphrag.context_graph.knowledge import (
    DIMENSIONS, Applicability, BusinessContext, DimensionScope, EvidenceRef, Fact, KnowledgeItem,
)

VOCABULARY: dict[str, tuple[str, ...]] = {
    "customer": tuple(f"C{i}" for i in range(1, 7)),
    "brand": tuple(f"B{i}" for i in range(1, 5)),
    "market": tuple(f"M{i}" for i in range(1, 9)),
    "product": tuple(f"P{i}" for i in range(1, 11)),
    "product_family": tuple(f"F{i}" for i in range(1, 4)),
    "workflow": ("W1", "W2"),
}
EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
FACTS = tuple(f"gauge.g{i}" for i in range(12))
_EVIDENCE = (EvidenceRef(source_id="SYN-EVIDENCE"),)


def synthetic_items(count: int, *, seed: int = 7, tenant: str = "syn", restrict: float = 0.35) -> list[KnowledgeItem]:
    """``count`` published items with random scopes, windows and constraints."""
    rng = random.Random(seed)
    items: list[KnowledgeItem] = []
    for n in range(1, count + 1):
        scopes = {}
        for dim in DIMENSIONS:
            roll = rng.random()
            if roll < restrict:
                values = rng.sample(VOCABULARY[dim], k=rng.choice((1, 1, 2)))
                scopes[dim] = DimensionScope(mode="in", values=values)
            elif roll < restrict + 0.05:
                scopes[dim] = DimensionScope(mode="not_in", values=rng.sample(VOCABULARY[dim], k=1))
            else:
                scopes[dim] = DimensionScope.unrestricted()
        valid_from = EPOCH + timedelta(days=rng.randint(0, 120)) if rng.random() < 0.3 else None
        valid_to = EPOCH + timedelta(days=rng.randint(200, 500)) if rng.random() < 0.2 else None
        modality = rng.choices(("mandatory", "recommended", "prohibited"), weights=(5, 3, 2))[0]
        fact = rng.choice(FACTS)
        operator = rng.choice(("gte", "lte"))
        items.append(KnowledgeItem.model_validate(dict(
            id=f"SYN-{n:05d}", version=1, tenant=tenant, title=f"synthetic rule {n}",
            statement=f"synthetic rule {n}", subject_key=f"syn.subject_{rng.randint(0, max(4, count // 25))}",
            modality=modality, authority="internal", status="published",
            applicability=Applicability(**scopes, valid_from=valid_from, valid_to=valid_to),
            recorded_at=EPOCH,
            constraint={"all_of": [{"fact": fact, "operator": operator, "value": rng.randint(1, 9)}]},
            constraint_provenance={"authored_by": "system:synthetic", "approval_ref": "SYN"},
            sources=_EVIDENCE, reviewed_by="system:synthetic", review_ref="SYN",
        )))
    return items


def synthetic_context(rng: random.Random, *, multi: float = 0.2, missing: float = 0.1) -> BusinessContext:
    values: dict[str, list[str] | None] = {}
    for dim in DIMENSIONS:
        roll = rng.random()
        if roll < missing:
            values[dim] = None
        elif roll < missing + multi:
            values[dim] = rng.sample(VOCABULARY[dim], k=2)
        else:
            values[dim] = [rng.choice(VOCABULARY[dim])]
    return BusinessContext(**values, as_of=EPOCH + timedelta(days=rng.randint(0, 540)))


def synthetic_facts(rng: random.Random) -> dict[str, Fact]:
    return {name: Fact(value=rng.randint(0, 10), evidence=_EVIDENCE) for name in FACTS if rng.random() < 0.9}


__all__ = ["EPOCH", "FACTS", "VOCABULARY", "synthetic_context", "synthetic_facts", "synthetic_items"]
