# scripts/dev/

One-off development helper scripts — not part of the production pipeline.

| Script | Purpose |
|--------|---------|
| `check_db.py` | Print entity/chunk/relation counts from a live Neo4j instance |
| `check_embed.py` | Smoke-test the Gemini embed API on one hardcoded string, print the resulting vector's dimension (legacy — Gemini is no longer on the default embedding path; see ADR 0004) |
| `check_key.py` | Print first/last chars of the loaded Google API key (legacy — Gemini is no longer on the default path; see ADR 0004) |
| `diagnose_aerospace_retrieval.py` | Trace retrieval scoring for a single aerospace query |
| `diagnose_automotive_retrieval.py` | Trace retrieval scoring for a single automotive query |
| `make_linkedin_cover.py` | Render the LinkedIn cover image asset |
| `make_social_preview.py` | Render the social/OG preview image asset |
| `test_gnn.py` | Manual smoke-test for GNN scorer against a local graph |
| `test_hybrid.py` | Manual smoke-test for the hybrid retrieval pipeline |

Run from the repo root:
```bash
python scripts/dev/check_db.py
```
