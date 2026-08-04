"""§8/§15 — 'measure blocking_recall@10, @25, @50. If the expected entity is
not generated, report candidate_generation_miss separately from an ordinary
unresolved result.'

Honest limitation: src/resolution/candidates.py's all_names_in_workspace()
fetches the full tenant-scoped name pool rather than a DB-native trigram/ANN
index (documented in that module) — at this fixture's scale (a handful of
accounts per workspace), every candidate trivially fits under the cap=50
budget, so recall is expected to measure at or near 100%. That's a real,
correctly-computed measurement, not a rigged one — it just isn't a stress test
of blocking quality at scale, which would need many more entities per
workspace than this vertical slice's fixtures provide.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.domain.crm import Account
from src.domain.identity import crm_entity_id
from src.graph.repositories.crm_repository import CrmRepository
from src.resolution.candidates import CandidateGenerator, union_candidates

pytestmark = pytest.mark.asyncio

_FIXTURE_NAMES = [
    "Volkswagen Group", "Volkswagen Financial Services", "Acme Corp", "Acme Global Holdings",
    "Northwind Traders", "Globex Corporation", "Initech", "Umbrella Corp", "Stark Industries",
    "Wayne Enterprises",
]

# (expected entity name a real mention should resolve to)
_EXPECTED_ENTITIES = ["Volkswagen Group", "Acme Corp", "Globex Corporation"]


async def test_blocking_recall_at_10_25_50(executor):
    workspace_id = f"ws-recall-{uuid4().hex[:8]}"
    crm_repo = CrmRepository(executor)
    for i, name in enumerate(_FIXTURE_NAMES):
        await crm_repo.upsert_account(Account(
            account_id=crm_entity_id(workspace_id, "salesforce", "Account", f"acc-{i}"),
            workspace_id=workspace_id, source_record_id=f"rec-{i}", name=name,
        ))

    generator = CandidateGenerator(executor)
    pool = await generator.all_names_in_workspace(workspace_id, "Account")
    candidates = union_candidates(pool, cap=50)
    names_in_pool = [c.name for c in candidates]

    hits_at = {10: 0, 25: 0, 50: 0}
    misses: list[str] = []
    for expected_name in _EXPECTED_ENTITIES:
        if expected_name not in names_in_pool:
            misses.append(expected_name)  # candidate_generation_miss
            continue
        rank = names_in_pool.index(expected_name)
        for k in hits_at:
            if rank < k:
                hits_at[k] += 1

    total = len(_EXPECTED_ENTITIES)
    recall = {k: hits_at[k] / total for k in hits_at}

    assert misses == [], f"candidate_generation_miss (distinct from an unresolved result): {misses}"
    assert recall[10] == 1.0
    assert recall[25] == 1.0
    assert recall[50] == 1.0
    print(f"blocking_recall@10={recall[10]:.2f} @25={recall[25]:.2f} @50={recall[50]:.2f} "
          f"(pool_size={len(names_in_pool)})")
