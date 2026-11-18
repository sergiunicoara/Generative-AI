Baseline: full.  Valid runs per profile: full=3, vector_only=3, text_hybrid=3, full_context=3

| vs | scope | n | pass (base/other) | only base | only other | McNemar p | Holm p | faith pairs | mean diff | 95% CI | distinguishable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vector_only | all | 34 | 28/23 | 9 | 4 | 0.267 | 0.801 | 26 | 0.028 | [-0.023, +0.084] | no |
| vector_only | multi_hop | 6 | 5/6 | 0 | 1 | 1.000 | n/a | 4 | -0.001 | [-0.089, +0.087] | no |
| text_hybrid | all | 34 | 28/24 | 6 | 2 | 0.289 | 0.801 | 28 | 0.014 | [-0.040, +0.079] | no |
| text_hybrid | multi_hop | 6 | 5/4 | 1 | 0 | 1.000 | n/a | 4 | -0.029 | [-0.063, +0.000] | no |
| full_context | all | 34 | 28/26 | 5 | 3 | 0.727 | 0.801 | 30 | 0.164 | [+0.082, +0.254] | faithfulness |
| full_context | multi_hop | 6 | 5/5 | 0 | 0 | 1.000 | n/a | 5 | 0.121 | [-0.048, +0.355] | no |

Paired bootstrap of per-question citation recall (baseline minus other; answer set = recall_citable, context set = recall_raw; no multiplicity adjustment):

| vs | scope | answer pairs | answer mean diff | answer 95% CI | context pairs | context mean diff | context 95% CI | distinguishable |
|---|---|---|---|---|---|---|---|---|
| vector_only | all | 32 | -0.023 | [-0.163, +0.109] | 33 | 0.202 | [+0.086, +0.328] | context |
| vector_only | multi_hop | 6 | -0.231 | [-0.444, -0.000] | 6 | 0.000 | [-0.250, +0.250] | answer |
| text_hybrid | all | 32 | 0.071 | [-0.016, +0.172] | 33 | 0.222 | [+0.111, +0.348] | context |
| text_hybrid | multi_hop | 6 | -0.120 | [-0.306, +0.037] | 6 | 0.139 | [+0.000, +0.306] | no |
| full_context | all | 32 | -0.075 | [-0.167, +0.000] | 0 | n/a | n/a | no |
| full_context | multi_hop | 6 | -0.231 | [-0.444, -0.000] | 0 | n/a | n/a | answer |

Holm adjustment is applied to the all-questions McNemar tests across the compared profiles; multi_hop rows are descriptive (a handful of questions cannot reach significance).
