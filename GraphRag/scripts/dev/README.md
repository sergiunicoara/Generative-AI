# scripts/dev/

One-off development helper scripts — not part of the production pipeline.

| Script | Purpose |
|--------|---------|
| `check_db.py` | Print entity/chunk/relation counts from a live Neo4j instance |
| `check_embed.py` | Verify embedding dimensions for all entities |
| `check_key.py` | Print first/last chars of the loaded Google API key |
| `test_gnn.py` | Manual smoke-test for GNN scorer against a local graph |
| `test_hybrid.py` | Manual smoke-test for the hybrid retrieval pipeline |

Run from the repo root:
```bash
python scripts/dev/check_db.py
```
