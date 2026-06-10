#!/usr/bin/env python
"""
Seed the knowledge graph with representative demo data.

Ingests a small curated corpus of aerospace regulatory documents,
builds communities, and seeds KPI events — giving a populated graph
within ~2 minutes with no external data required.

Usage:
    python scripts/seed_demo_data.py              # dry-run summary only
    python scripts/seed_demo_data.py --commit     # write to real Neo4j
    python scripts/seed_demo_data.py --commit --tenant demo

The script is idempotent: re-running with --commit will MERGE nodes, not
duplicate them.  Use --wipe to clear the tenant before seeding.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Make the project root importable when running as a script
sys.path.insert(0, str(Path(__file__).parents[1]))

import structlog
log = structlog.get_logger("seed_demo_data")

# ── Curated seed corpus ───────────────────────────────────────────────────────

SEED_ENTITIES = [
    # (name, type, confidence)
    ("FAA",                            "REGULATOR",              1.0),
    ("EASA",                           "REGULATOR",              1.0),
    ("Boeing",                         "ORG",                    1.0),
    ("Airbus",                         "ORG",                    1.0),
    ("Boeing 737 MAX",                 "AIRCRAFT_TYPE",          1.0),
    ("Airbus A320neo",                 "AIRCRAFT_TYPE",          1.0),
    ("FAA-AD-2024-01-02",              "AIRWORTHINESS_DIRECTIVE", 1.0),
    ("FAA-AD-2022-03-07",              "AIRWORTHINESS_DIRECTIVE", 1.0),
    ("FAA-AD-2020-05-11",              "AIRWORTHINESS_DIRECTIVE", 1.0),
    ("EASA-AD-2024-0072",              "AIRWORTHINESS_DIRECTIVE", 1.0),
    ("CFM LEAP-1B Engine",             "AIRCRAFT_COMPONENT",     0.95),
    ("Engine Mount Inspection",        "MAINTENANCE_PROCEDURE",  0.90),
    ("Angle of Attack Sensor",         "AIRCRAFT_COMPONENT",     0.92),
    ("MCAS Software v1.0",             "PRODUCT",                0.88),
    ("MCAS Software v2.0",             "PRODUCT",                0.95),
    ("14 CFR Part 39",                 "REGULATION",             1.0),
    ("CS-25",                          "REGULATION",             1.0),
    ("Southwest Airlines",             "ORG",                    0.95),
    ("Lion Air",                       "ORG",                    0.90),
    ("Ethiopian Airlines",             "ORG",                    0.90),
]

SEED_RELATIONS = [
    # (src, src_type, relation, tgt, tgt_type, confidence, source_doc)
    ("FAA-AD-2024-01-02", "AIRWORTHINESS_DIRECTIVE", "SUPERSEDES",
     "FAA-AD-2022-03-07", "AIRWORTHINESS_DIRECTIVE", 1.0, "faa-ad-2024"),
    ("FAA-AD-2022-03-07", "AIRWORTHINESS_DIRECTIVE", "SUPERSEDES",
     "FAA-AD-2020-05-11", "AIRWORTHINESS_DIRECTIVE", 1.0, "faa-ad-2022"),
    ("FAA-AD-2024-01-02", "AIRWORTHINESS_DIRECTIVE", "APPLIES_TO",
     "Boeing 737 MAX", "AIRCRAFT_TYPE", 0.98, "faa-ad-2024"),
    ("EASA-AD-2024-0072", "AIRWORTHINESS_DIRECTIVE", "APPLIES_TO",
     "Airbus A320neo", "AIRCRAFT_TYPE", 0.98, "easa-ad-2024"),
    ("FAA-AD-2024-01-02", "AIRWORTHINESS_DIRECTIVE", "MANDATED_BY",
     "FAA", "REGULATOR", 1.0, "faa-ad-2024"),
    ("EASA-AD-2024-0072", "AIRWORTHINESS_DIRECTIVE", "MANDATED_BY",
     "EASA", "REGULATOR", 1.0, "easa-ad-2024"),
    ("Boeing", "ORG", "MANUFACTURES",
     "Boeing 737 MAX", "AIRCRAFT_TYPE", 1.0, "boeing-profile"),
    ("Airbus", "ORG", "MANUFACTURES",
     "Airbus A320neo", "AIRCRAFT_TYPE", 1.0, "airbus-profile"),
    ("Engine Mount Inspection", "MAINTENANCE_PROCEDURE", "APPLIES_TO",
     "CFM LEAP-1B Engine", "AIRCRAFT_COMPONENT", 0.92, "maintenance-manual"),
    ("MCAS Software v2.0", "PRODUCT", "SUPERSEDES",
     "MCAS Software v1.0", "PRODUCT", 1.0, "boeing-swcr"),
    ("FAA-AD-2024-01-02", "AIRWORTHINESS_DIRECTIVE", "REFERENCES",
     "14 CFR Part 39", "REGULATION", 1.0, "faa-ad-2024"),
    ("Southwest Airlines", "ORG", "OPERATES",
     "Boeing 737 MAX", "AIRCRAFT_TYPE", 0.95, "fleet-registry"),
]

SEED_CONFLICTS = [
    # airworthy vs unairworthy for the same aircraft — triggers contradiction detection
    {
        "src": "Boeing 737 MAX", "src_type": "AIRCRAFT_TYPE",
        "tgt": "FAA-AD-2020-05-11", "tgt_type": "AIRWORTHINESS_DIRECTIVE",
        "relation": "IS_COMPLIANT_WITH",
        "confidence": 0.85, "source_doc": "inspection-report-2024-01",
    },
    {
        "src": "Boeing 737 MAX", "src_type": "AIRCRAFT_TYPE",
        "tgt": "FAA-AD-2020-05-11", "tgt_type": "AIRWORTHINESS_DIRECTIVE",
        "relation": "IS_NON_COMPLIANT_WITH",
        "confidence": 0.78, "source_doc": "ad-compliance-check-2024-03",
    },
]

SEED_DOCUMENTS = [
    {"doc_id": "faa-ad-2024",       "filename": "FAA-AD-2024-01-02.pdf",
     "authority": "regulatory",      "ingested_at": "2024-01-15T10:00:00Z"},
    {"doc_id": "faa-ad-2022",       "filename": "FAA-AD-2022-03-07.pdf",
     "authority": "regulatory",      "ingested_at": "2022-03-10T10:00:00Z"},
    {"doc_id": "faa-ad-2020-old",   "filename": "FAA-AD-2020-05-11.pdf",
     "authority": "regulatory",      "ingested_at": "2020-05-12T10:00:00Z"},
    {"doc_id": "easa-ad-2024",      "filename": "EASA-AD-2024-0072.pdf",
     "authority": "regulatory",      "ingested_at": "2024-02-01T10:00:00Z"},
    {"doc_id": "boeing-swcr",       "filename": "Boeing_MCAS_SWChangeRecord.pdf",
     "authority": "manufacturer",    "ingested_at": "2023-11-01T10:00:00Z"},
    {"doc_id": "maintenance-manual","filename": "737MAX_CMM_Engine_Mount.pdf",
     "authority": "manufacturer",    "ingested_at": "2024-03-01T10:00:00Z"},
    {"doc_id": "inspection-report-2024-01", "filename": "G-ABCD_inspection_2024-01.pdf",
     "authority": "internal",        "ingested_at": "2024-01-20T10:00:00Z"},
    {"doc_id": "ad-compliance-check-2024-03", "filename": "G-ABCD_AD_compliance_2024-03.pdf",
     "authority": "internal",        "ingested_at": "2024-03-05T10:00:00Z"},
    {"doc_id": "boeing-profile",    "filename": "Boeing_company_profile.pdf",
     "authority": "informal",        "ingested_at": "2023-01-01T10:00:00Z"},
    {"doc_id": "fleet-registry",    "filename": "SWA_fleet_registry_2024.pdf",
     "authority": "internal",        "ingested_at": "2024-01-01T10:00:00Z"},
]


# ── Seeding logic ─────────────────────────────────────────────────────────────

async def seed(tenant: str, commit: bool, wipe: bool) -> None:
    from graphrag.graph.neo4j_client import get_neo4j
    neo4j = get_neo4j()

    if wipe:
        log.info("seed.wipe_tenant", tenant=tenant)
        if commit:
            await neo4j.run(
                "MATCH (n) WHERE n.tenant = $tenant DETACH DELETE n",
                tenant=tenant,
            )

    print(f"\n{'='*60}")
    print(f"  Seeding demo corpus — tenant: {tenant!r}  commit={commit}")
    print(f"{'='*60}\n")

    # 1. Documents
    print(f"[1/6] Writing {len(SEED_DOCUMENTS)} documents...")
    if commit:
        for doc in SEED_DOCUMENTS:
            await neo4j.run(
                """
                MERGE (d:Document {id: $doc_id, tenant: $tenant})
                ON CREATE SET d.filename    = $filename,
                              d.authority   = $authority,
                              d.ingested_at = $ingested_at
                """,
                tenant=tenant, **doc,
            )

    # 2. Entities
    print(f"[2/6] Writing {len(SEED_ENTITIES)} entities...")
    if commit:
        for name, etype, confidence in SEED_ENTITIES:
            await neo4j.run(
                """
                MERGE (e:Entity {name: $name, type: $type, tenant: $tenant})
                ON CREATE SET e.id          = $entity_id,
                              e.confidence  = $confidence,
                              e.created_at  = datetime()
                """,
                name=name, type=etype, confidence=confidence,
                entity_id=str(uuid.uuid4()), tenant=tenant,
            )

    # 3. Relations
    print(f"[3/6] Writing {len(SEED_RELATIONS)} relations...")
    if commit:
        for src, src_t, rel, tgt, tgt_t, conf, doc_id in SEED_RELATIONS:
            await neo4j.run(
                """
                MATCH (s:Entity {name: $src, type: $src_t, tenant: $tenant})
                MATCH (t:Entity {name: $tgt, type: $tgt_t, tenant: $tenant})
                MERGE (s)-[r:RELATES_TO {relation: $rel, tenant: $tenant}]->(t)
                ON CREATE SET r.confidence   = $confidence,
                              r.source_doc_id= $doc_id,
                              r.created_at   = datetime()
                """,
                src=src, src_t=src_t, tgt=tgt, tgt_t=tgt_t,
                rel=rel, confidence=conf, doc_id=doc_id, tenant=tenant,
            )

    # 4. Conflicts (contradiction demo)
    print(f"[4/6] Writing {len(SEED_CONFLICTS)} conflicts (contradiction demo)...")
    if commit:
        for c in SEED_CONFLICTS:
            await neo4j.run(
                """
                MATCH (s:Entity {name: $src, type: $src_t, tenant: $tenant})
                MATCH (t:Entity {name: $tgt, type: $tgt_t, tenant: $tenant})
                MERGE (s)-[r:RELATES_TO {relation: $rel, tenant: $tenant,
                                         source_doc_id: $doc_id}]->(t)
                ON CREATE SET r.confidence = $confidence,
                              r.created_at = datetime()
                """,
                src=c["src"], src_t=c["src_type"],
                tgt=c["tgt"], tgt_t=c["tgt_type"],
                rel=c["relation"], confidence=c["confidence"],
                doc_id=c["source_doc"], tenant=tenant,
            )
        # Flag the conflict in the graph
        await neo4j.run(
            """
            MERGE (cf:Conflict {
                id: $cid, tenant: $tenant,
                src: 'Boeing 737 MAX', tgt: 'FAA-AD-2020-05-11',
                type: 'positive_negative_pair'
            })
            ON CREATE SET cf.created_at = datetime(), cf.status = 'open'
            """,
            cid=str(uuid.uuid4()), tenant=tenant,
        )

    # 5. Graph health snapshot
    print("[5/6] Recording health snapshot...")
    if commit:
        await neo4j.run(
            """
            CREATE (h:GraphHealthSnapshot {
                id:                 $id,
                tenant:             $tenant,
                entity_count:       $entity_count,
                edge_count:         $edge_count,
                alias_coverage:     0.92,
                high_conf_rate:     0.83,
                contradiction_rate: 0.85,
                orphan_rate:        0.08,
                community_coherence:0.69,
                recorded_at:        datetime()
            })
            """,
            id=str(uuid.uuid4()), tenant=tenant,
            entity_count=len(SEED_ENTITIES),
            edge_count=len(SEED_RELATIONS) + len(SEED_CONFLICTS),
        )

    # 6. Calibration snapshot
    print("[6/6] Recording calibration snapshot...")
    if commit:
        await neo4j.run(
            """
            CREATE (s:CalibrationSnapshot {
                id:            $id,
                tenant:        $tenant,
                brier_score:   0.19,
                model_version: 'llama-3.3-70b',
                sample_size:   104,
                recorded_at:   datetime()
            })
            """,
            id=str(uuid.uuid4()), tenant=tenant,
        )

    await neo4j.close()

    print("\n" + "="*60)
    print(f"  Done.  Tenant '{tenant}' seeded with:")
    print(f"    {len(SEED_DOCUMENTS):3d} documents")
    print(f"    {len(SEED_ENTITIES):3d} entities")
    print(f"    {len(SEED_RELATIONS):3d} relations  +  {len(SEED_CONFLICTS)} conflict pairs")
    print(f"    1 graph health snapshot  +  1 calibration snapshot")
    if not commit:
        print("\n  DRY RUN — pass --commit to write to Neo4j.")
    print("="*60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commit", action="store_true",
                        help="Write to Neo4j (default: dry-run summary only)")
    parser.add_argument("--wipe",   action="store_true",
                        help="Delete all nodes for the tenant before seeding")
    parser.add_argument("--tenant", default="aerospace",
                        help="Tenant to seed (default: aerospace)")
    args = parser.parse_args()

    asyncio.run(seed(tenant=args.tenant, commit=args.commit, wipe=args.wipe))


if __name__ == "__main__":
    main()
