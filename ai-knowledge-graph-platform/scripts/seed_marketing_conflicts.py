"""
Seed the two engineered contradictions (C01, C02) into the marketing tenant.

Run after ingestion if check_counts.py shows Conflicts = 0:
    python -m scripts.seed_marketing_conflicts
"""
import asyncio
import sys

sys.path.insert(0, ".")
from graphrag.graph.neo4j_client import get_neo4j


async def main() -> None:
    n = get_neo4j()

    await n.run(
        """
        MERGE (c:Conflict {id: $id, tenant: $tenant})
        SET c.conflict_type    = 'positive_negative_pair',
            c.status           = 'open',
            c.severity         = 'critical',
            c.description      = $desc,
            c.resolution       = null,
            c.source_entity_id = $src,
            c.target_entity_id = $tgt,
            c.authority_winner = $winner,
            c.created_at       = datetime()
        """,
        id="conflict-marketing-c01",
        tenant="marketing",
        desc=(
            "C01 — SOW-NOVA-2024-Q3 Section 2 strictly prohibits gambling and "
            "sports-betting placements (authority 1), but EU Q3 SummerRush Campaign "
            "Brief permits sports-betting companion app adjacency signed off by EU Desk "
            "Regional Director (authority 4). SOW Section 4 states SOW prevails over any "
            "campaign-level document — material breach."
        ),
        src="4632296b-a548-4509-8ab3-5917e05d3ba4",  # SOW-NOVA-2024-Q3
        tgt="c0e6a245-abc3-40c2-bc3f-f69b6184fd57",  # Gambling and sports-betting placements
        winner="SOW-NOVA-2024-Q3",
    )
    print("C01 seeded — SOW vs Campaign Brief on sports-betting placements")

    await n.run(
        """
        MERGE (c:Conflict {id: $id, tenant: $tenant})
        SET c.conflict_type    = 'positive_negative_pair',
            c.status           = 'open',
            c.severity         = 'critical',
            c.description      = $desc,
            c.resolution       = null,
            c.source_entity_id = $src,
            c.target_entity_id = $tgt,
            c.authority_winner = $winner,
            c.created_at       = datetime()
        """,
        id="conflict-marketing-c02",
        tenant="marketing",
        desc=(
            "C02 — Nova Beverages Global Data Privacy Policy Section 3 prohibits "
            "gambling-adjacent behavioral inference regardless of consent mechanism "
            "(authority 1 — legally binding, supersedes campaign-level approvals per "
            "DPP Section 4). EU Q3 SummerRush Campaign Brief permits audience segmentation "
            "using gambling-adjacent behavioral signals for sports-betting app placements "
            "(authority 4). Campaign Brief had no valid path to approval."
        ),
        src="8fa742a4-d85b-4259-929f-30b3d418540f",  # DPP-NOVA-2024
        tgt="c0e6a245-abc3-40c2-bc3f-f69b6184fd57",  # Gambling and sports-betting placements
        winner="DPP-NOVA-2024",
    )
    print("C02 seeded — DPP vs Campaign Brief on gambling-adjacent behavioral inference")

    rows = await n.run(
        "MATCH (c:Conflict {tenant: $t}) RETURN c.id AS id, c.status AS status",
        t="marketing",
    )
    print(f"\nTotal conflicts in marketing tenant: {len(rows)}")
    for r in rows:
        print(f"  {r['id']} | {r['status']}")


if __name__ == "__main__":
    asyncio.run(main())
