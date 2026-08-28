# Security and privacy review

## Verified locally

- `Limina/.env` is loaded only from the harness directory and is ignored by Git.
- The API key is consumed in memory and is not printed by configuration checks, CLI output, JSON, CSV, or reports.
- Historical and captured datasets pass through the conservative sanitizer before persistence.
- The generated Limina HTML report was inspected for email-shaped strings; none were found. Numeric sequences remain because they can be IDs/metrics and require human review.
- Limina output is stored under ignored `results/` and copied to ignored `artifacts/`.
- Recruiter Agent source files were not modified.

## Vendor claims (not independently verified)

The SDK/repository describes asynchronous tracing, privacy/retention guarantees, and key isolation. This harness does not independently verify those vendor claims.

## Not independently verified

- Account pricing or partner-beta allocation.
- Long-term retention, model-training, or deletion behavior at the remote service.
- Whether every generated HTML field is free of indirect identifiers.
- Runtime tracing overhead in the Recruiter Agent process; no production tracing decorator was added.

## Operational decision

Keep external evaluation opt-in. Review sanitized payloads before any production export, and never place API keys in datasets, reports, README examples, or tests.
