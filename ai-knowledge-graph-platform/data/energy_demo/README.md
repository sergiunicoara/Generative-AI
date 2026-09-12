# Synthetic source exports

All files in this directory are fictional fixtures. `snowflake_observations.json`
models a Snowflake-style telemetry extract, and is genuinely the source this
demo runs from: `graphrag/ingestion/rml_rdf.py`'s `materialize_rml()` executes
`ontology/mappings/energy-observations.rml.ttl` against it for real (see
`docs/demos/energy_asset_intelligence.md`'s "Limitations" section). SAP-style
assets/work orders are likewise a real, executed R2RML source
(`ontology/mappings/energy-assets.r2rml.ttl` against an ephemeral or supplied
SQLite export -- `graphrag/domains/energy/fixtures.py`). Only the
SharePoint-style document revisions (manufacturer bulletins) and the
turbine/gearbox topology stay hand-written in-code, so the demo stays
deterministic without external credentials.
