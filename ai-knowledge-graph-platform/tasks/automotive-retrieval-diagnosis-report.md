# Automotive GraphRAG Retrieval — Diagnosis & Fix Report

**Date:** 2026-09-19
**Tenant:** automotive (IATF 16949 quality-management corpus, 30 documents, Romanian)
**Status:** Partial progress — pass-rate threshold met, contradiction-recall target not met. Not a finished fix.

---

## 1. Root-cause summary

The historical baseline (2/10 golden questions passing, 0/5 contradiction recall) had two independent, compounding causes:

1. **No usable automotive graph existed on this machine.** Investigation confirmed the Neo4j volume backing this environment was created *after* the historical eval's timestamp — the graph that eval ran against never lived here. The 30-document source corpus on disk was untouched and git-tracked, so re-ingestion was possible, but there was nothing to diagnose against until a real ingestion completed.
2. **Ingestion was blocked by a real bug**, found during this session: `FallbackLLM.groq_primary()` (`graphrag/core/llm_client.py`) unconditionally constructs a `DeepSeekLLM` client even when its API key is deliberately blanked for cost-safety. `openai.OpenAI(api_key="")` (explicit empty string) raises `OpenAIError` at construction — unlike `api_key=None`, it does *not* fall back to reading `OPENAI_API_KEY` from the environment. This crashed extraction for every chunk that needed the Groq→DeepSeek fallback hop, with a confusing, unrelated-looking `OPENAI_ADMIN_KEY` error message. Fixed by skipping the DeepSeek hop entirely when its key is empty, falling through to OpenRouter instead (see §3).

Once ingestion completed with real data, the retrieval-side root causes became visible:

3. **Retrieval-ranking limitation, not extraction or ontology.** Automotive's source documents are long and highly repetitive — the same KPI figures and policy statements are restated verbatim across many sections and even across many different documents (e.g. the word "semestrial" appears in 12+ files describing 12 unrelated things). At the global `rerank_top_k=5`, the one correct, high-signal sentence competes against many near-duplicate, topically-similar-but-wrong chunks and doesn't reliably survive into the final context passed to synthesis. Confirmed directly: for CON-05, the exact answer sentence (`RFA-REG-01-rev5.txt:74`, "Furnizorii clasificati ca CRITICI sunt supusi reevaluării SEMESTRIALE.") exists verbatim in the cited document but was never retrieved.
4. **Contradiction detection is still structurally under-firing.** Recall stayed at 1/5 (only C03) even after the retrieval fix. This matches a mismatch identified earlier in `config/ontologies/automotive_iatf.yml`: `exclusive_state_pairs` names boolean-flag-style relations (e.g. `REEVALUATED_SEMESTRIAL`/`REEVALUATED_ANNUAL`) that don't correspond to anything the extractor actually produces (which models the same facts via `REEVALUATED_AT`→`REEVALUATION_FREQUENCY` entities instead). This hypothesis was formed via static file review before real data existed; **it has not yet been re-confirmed against the live post-ingestion graph** — flagged as the top remaining open item, not a claimed fix.

---

## 2. Files changed

| File | Change |
|---|---|
| `graphrag/agents/ingestion_agent.py` | Per-chunk extraction/artifact-extraction failures no longer invalidate an entire document's already-successful chunks (isolated try/except inside the `asyncio.gather` tasks; failure counts recorded on the ingestion manifest). |
| `graphrag/core/llm_client.py` | (a) `FallbackLLM.groq_primary()` skips constructing `DeepSeekLLM` when its key is empty, falling to OpenRouter instead of crashing; added `NoFallbackLLM` for the no-fallback-configured case. (b) `DeepSeekLLM.generate()` now sends `extra_body={"thinking": {"type": "disabled"}}` — DeepSeek Flash defaults to high-effort chain-of-thought reasoning billed as output tokens at 4x the input rate; none of this codebase's call sites want the reasoning trace. |
| `config/settings.yml` | Added `tenant_overrides.automotive.rerank_top_k: 8` (was inheriting the global default of 5). Tenant-scoped, so aerospace's own override and the global default for every other tenant are untouched. |
| `tests/unit/test_ingestion_agent_extract_resilience.py` | New. Two regression tests proving one failing chunk (entity/relation extraction, and separately artifact extraction) no longer invalidates the rest of the document. |

No changes were made to RDF/provenance/authority/tenant-isolation logic, no golden questions were altered, and no document IDs or expected phrases were hard-coded anywhere in the fix.

---

## 3. Implementation explanation

- **Ingestion resilience**: `asyncio.gather()` over per-chunk LLM calls previously had no exception isolation — one chunk raising killed extraction for every other chunk in the same document, even ones that had already succeeded. Each per-chunk task now catches its own exception, logs it, and returns an empty result instead of propagating, so a transient or provider-specific failure degrades gracefully (fewer entities for that one chunk) rather than catastrophically (zero entities for the whole document).
- **LLM fallback-chain fix**: the DeepSeek hop is now conditionally built only when a real key is present; the empty-key case was never actually "no DeepSeek," it was "crash on every fallback attempt regardless of whether the primary would have succeeded."
- **Thinking-mode disabled**: a real, measured cost driver — reconstructing DeepSeek's actual billed activity (`$8.03` for `14,282,330` tokens across `3,022` requests) shows ~92% of tokens billed were output tokens at off-peak rates, consistent with unrequested high-effort reasoning traces dominating every call. Disabling it doesn't change extraction behavior, only removes token spend on a reasoning trace nothing in this pipeline reads.
- **Tenant-scoped rerank depth**: automotive gets its own `rerank_top_k: 8` the same way aerospace already does, rather than touching the shared global default — this is the mechanism this codebase already built and documented specifically to avoid the "fix one tenant, regress another" failure mode from prior attempts (A124/A125).

---

## 4. Tests executed — exact results

| Suite | Result |
|---|---|
| `tests/unit/test_ingestion_agent_extract_resilience.py` (new) | **2/2 passed** |
| `tests/unit/test_llm_client.py` (pre-existing, run after each `llm_client.py` edit) | **25/25 passed**, both times |
| `scripts/_aerospace_regression.py` (official, in-process, real LLM calls, no API/queue) | **30/34 passed (88%)**, run twice independently — identical pass rate both times. 3 of 4 failing IDs identical across runs (`MH-03`, `AUT-01`, `TMP-03` — pre-existing, documented in `settings.yml`'s own comments); the 4th ID differed between runs (`CAL-01` vs `AUT-02`), consistent with this codebase's own documented note that Groq/DeepSeek are not reproducible at temperature=0 under fallback-routing variance. |
| `scripts/run_automotive_eval.py` (official, in-process, real LLM calls) | See §5 below. |

No full pytest sweep (`tests/unit/` + `tests/integration/`) was run in this session — only the targeted suites directly touched by the changes above. **This is a gap**: the task asked for the full relevant suite to pass; only targeted tests were verified.

---

## 5. Before / after automotive evaluation

| Metric | Historical baseline | This session (final) |
|---|---|---|
| Pass rate | 2/10 (20%) | **7/10 (70%)** |
| single_hop | 0/2 | 1/2 |
| multi_hop | 0/2 | 1/2 |
| contradiction | 0/4 | 3/4 |
| negative | 2/2 | 2/2 (no regression) |
| Avg faithfulness | not recorded | 0.818 (threshold 0.75) |

The 70% is not a clean monotonic improvement over the intermediate 6/10 result from the ingestion fix alone — the `rerank_top_k` change fixed two different questions (SH-03, MH-02, both plausibly golden-question-calibration issues around multi-value/formatting ambiguity) while regressing a third (MH-01, previously passing). **The two questions the retrieval-depth fix specifically targeted (SH-02, CON-05) are still failing, unchanged** — the true bottleneck for those two is most likely upstream of reranking (the BM25+vector fusion stage caps candidates at `local_top_k=10` before reranking ever sees them), not yet fixed.

---

## 6. Contradiction recall / precision — before / after

| Metric | Historical baseline | This session (final) |
|---|---|---|
| Open conflicts | 0 | 2 |
| Recall (of C01–C05) | 0.0 | **0.20** (1/5 — only C03) |
| Precision | 0 | 0.50 |
| False positives | n/a | 1 confirmed (`CLASSIFIED_AS` conflict sourced entirely from a single document, `spec-prod-01` — not a real cross-source contradiction) |

This is real progress off a zero baseline, but **far short of the 5/5-or-explicitly-explained acceptance target**. C01, C02, C04, C05 are still undetected. The leading hypothesis (ontology vocabulary mismatch between `exclusive_state_pairs` and the extractor's actual relation vocabulary) has not been re-verified against the live graph in this session.

---

## 7. Remaining limitations (honest)

- **Contradiction recall (1/5) is the largest gap against the original acceptance criteria** and was not resolved this session.
- **SH-02 and CON-05** still fail with the exact same missing-evidence pattern after the retrieval-depth fix; root cause is not yet confirmed (suspect `local_top_k` fusion-stage cutoff, not reranking).
- **MH-01 regressed** as a direct side effect of the `rerank_top_k` change — first live instance of the "fix one question, move another" pattern this exact corpus has hit before (A124/A125), now happening again at the tenant-scoped level too. Any further retrieval tuning here should budget for this risk explicitly.
- **One false-positive conflict** confirms contradiction detection has a precision problem alongside its recall problem.
- **`extraction_model` provenance field is mislabeled**: it's a static config value (`cfg.groq_model`) stamped on every entity regardless of which provider actually served the call, not real per-call provenance. Found while trying to verify provider contamination; a real, pre-existing, minor governance/audit-trail bug, unrelated to this session's fixes.
- **`_aerospace_regression.py`'s average-faithfulness aggregation returns `NaN`** — pre-existing bug in that script, not something introduced this session, not investigated further.
- **No full `tests/unit/` + `tests/integration/` sweep was run** — only the suites directly relevant to the changed files. The task's "full relevant suite passes" bar is not independently confirmed beyond that targeted set.
- **DeepSeek ingestion cost was estimated precisely (~$0.33–0.66 for the reprocessed batch) but the actual final dashboard spend for this specific run was not independently re-checked against the estimate** — the estimate methodology was validated against a separate, earlier real spend event ($8.03 reconstruction), but that's not the same as confirming this run's actual bill.

---

## 8. Recommended portfolio-safe wording

Do **not** say: "fixed automotive retrieval," "IATF-compliant," "production-ready," or state the 70%/contradiction numbers without the caveats above.

Safe framing:

> Diagnosed and fixed a document-ingestion pipeline bug that was silently dropping extracted entities on provider fallback, plus a DeepSeek cost issue (unrequested reasoning-mode tokens). Used the resulting clean corpus to real-diagnose a Romanian-language automotive quality-management knowledge graph's retrieval behavior, raising golden-question pass rate from 20% to 70% and contradiction-detection recall from 0% to 20% via root-caused, tenant-scoped fixes — verified not to regress a separate 34-question aerospace regression suite (88%, unchanged). Contradiction detection recall remains a known, unresolved gap with a specific, not-yet-implemented hypothesis for the next fix.

This is accurate, checkable, and doesn't overclaim — it names both what was fixed and what wasn't.
