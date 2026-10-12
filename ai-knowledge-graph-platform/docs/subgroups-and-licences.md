# Subgroup evaluation and the model licence register

Both are offline and cost nothing. Neither changes runtime behaviour.

## Subgroup evaluation

`python scripts/eval_subgroups.py [--write]` regroups the recorded results and writes `evals/subgroup_report.json`.
It reads `evals/last_run.json` (golden set by question type), re-scores the routing sets with the deterministic router
(by expected class and by set) and groups `evals/entity_resolution_results.json` by case category.

Each group gets a rate and a Wilson 95% interval. A group is flagged only if it has at least `min_n` (3) items and is at
least `min_gap` (0.15) under the overall rate; smaller groups are marked `insufficient_n`. A flag means "look here".

Current output (descriptive, from recorded runs):

* Golden set: overall 30/34; flagged `multi_hop` (4/6) and `authority_chain` (2/3). Both groups are tiny, so the
  intervals are wide and overlap the overall rate.
* Routing: no class or set is flagged (max gap 0.09 by class). Sets 1 and 2 were used to write rules, so only set 3
  is a clean estimate; see [query-routing.md](query-routing.md).
* Entity resolution: `acronym`, `near-identifier-distinct` and `shared-tokens-distinct` are flagged; see
  [entity-resolution.md](entity-resolution.md).

"Subgroups" here are question and case categories. The platform records no demographic attributes, so this is not a
fairness audit of people; it shows where quality is uneven across kinds of input.

## Model licence register

`config/licenses.yml` lists each model family the code can call, its licence or terms, and a `review_status`.
`graphrag/core/license_register.py` scans `config/` and `graphrag/` for model identifiers; the unit test
`tests/unit/test_license_register.py` (part of the normal CI test job) fails if a model has no register entry, uses a
licence outside `allowed_licences`, or if a register row no longer matches any model. `python scripts/check_licenses.py
[--strict]` prints the report; `--strict` also fails while any entry is `needs_review`.

All entries are currently `needs_review`: they were filled in from public descriptions and have not been confirmed
against the live model cards or terms by a person. The check enforces that a licence is *recorded and allowed*, not that
it is legally sufficient. Datasets are an empty list because none are bundled; add them with the same shape.
