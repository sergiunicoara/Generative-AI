# IATF 16949 audit support

Evidence for an auditor to review, per clause. It never states that an organisation is compliant or certified.
Everything here is deterministic and offline except the API's sign-off storage (Neo4j). No LLM is involved.

| piece | where | status |
| --- | --- | --- |
| Clause catalogue (numbers + own-words headings, proposed evidence mapping) | `config/iatf_clauses.yml` | implemented; mapping **proposed**, needs a quality engineer |
| Evidence view: current and superseded evidence, passages with line numbers, conflicts, age | `graphrag/domains/automotive/audit.py` | implemented, tested |
| Coverage, freshness alert (non-zero exit), as-of export JSON/CSV | `scripts/iatf_audit.py` | implemented, tested |
| Auditor sign-off, hash-chained per tenant, reviewer = token subject | `graphrag/domains/automotive/signoff.py` | implemented; concurrency proven on live Neo4j |
| API | `/audit/iatf/clauses`, `/clauses/{id}`, `/clauses/{id}/signoff`, `/signoffs`, `/export` | implemented, tested |
| Golden checker percent fix + offline rescore of saved answers | `scripts/run_golden_eval.py`, `scripts/rescore_saved_answers.py` | implemented, measured |
| 12 held-out cases, structurally validated | `data/eval_golden/queries_automotive_heldout.json` | **written, never run** (running costs credit) |

## Evidence view

A clause is **evidenced** when at least one *current* document of a listed family contains one of its keywords. The
other statuses are **stale** (only evidence older than `max_document_age_days`), **only_superseded** (evidence exists
only in superseded or archived revisions) and **missing**. Each evidence entry carries passages (file line and text),
the document's recorded time and age, and its contradiction tags; `conflicts` lists every document sharing a tag.

* Superseded = predecessor in `supersession_chains` (`config/ontologies/automotive_iatf.yml`) or stored in the corpus
  `archive` folder.
* `as_of`: documents recorded after the date are ignored, and a successor recorded later does not yet supersede its
  predecessor. The corpus has no effective dates, so "recorded at" stands in for "in force from". This is an
  approximation and is stated in every export.
* Keyword matching is by word start, accent- and case-insensitive, with no stemming, so the catalogue lists the
  inflected forms it needs. A keyword hit is a located passage, not proof that the passage satisfies the requirement;
  the auditor decides.
* `evidence_digest` fingerprints the documents, recorded times and line numbers the auditor saw.

### Findings on the synthetic corpus (2026-10-13, 30 documents)

* 11 of 14 catalogued clauses evidenced. **Missing: 4.4.1.2 (product safety), 8.5.1.1 (control plan), 8.5.4
  (preservation/packaging).** The corpus is mostly supplier-management procedures; it has no control plan, product
  safety or packaging procedure. FP-PAC-04 mentions packing once in passing. The keywords were not widened to make
  these pass.
* 8.4.2.4 (supplier monitoring) shows CSR-CLIENT-2021 as superseded evidence and carries conflict tags C01, C02, C03 and C05.
  This is the same supersession and contradiction picture the golden failures below come from.
* `python scripts/iatf_audit.py freshness` exits 1 because of the three missing clauses. That is the intended alert.

## Auditor sign-off and the audit trail

`POST /audit/iatf/clauses/{id}/signoff` with `{"decision": "accepted" | "rejected" | "needs_more_evidence", "note"}`.

* Needs `write` scope and an authenticated subject. The reviewer is the token's subject, never a request field.
* The evidence digest is computed server-side from the evidence the API shows, so a client cannot sign off on a
  digest of its choosing.
* Entries are append-only `IatfSignoff` nodes chained by SHA-256 (`prev_hash` → `hash`) per tenant, with a
  `(tenant, seq)` uniqueness constraint. The append takes the chain head's write lock before comparing, so
  concurrent sign-offs never fork the chain: 12 concurrent appends on a live Neo4j produced one chain of 12.
* `GET /audit/iatf/signoffs` and the export report `chain.valid`. Editing, removing or truncating an entry is detected
  and the first broken sequence number is named.
* The export flags `signoff_digest_matches: false` when the evidence changed after the sign-off, so a stale approval
  is visible rather than silently carried forward.
* **Limit:** tamper-*evident*, not tamper-*proof*. Someone with database write access can rewrite the whole chain.
  For an external guarantee, publish the head hash (`chain.head`) periodically to a store the database administrator
  cannot change, for example a WORM bucket or a signed ticket. That is not wired.

Only tenants listed under `iatf_audit.tenants` in `config/settings.yml` get these routes; any other tenant receives 404.

## Golden-set diagnosis (no re-run)

The saved 2026-09-21 automotive run scored 6/10. All four failures were diagnosed from the saved answers and the corpus:

| id | cause | action |
| --- | --- | --- |
| SH-02, MH-02 | Correct answers written "1 %" / "95 %" (Romanian typography); the checker required "1%" / "95%" | checker now ignores whitespace before `%` (both sides) |
| SH-03 | Checker now passes it, but **not a genuine pass**: the answer refuses a single value, citing a conflict between 95 % and a 98 % that comes from the superseded CSR-CLIENT-2021 or an unrelated KPI line | recorded in `evals/automotive_rescored.json`; root cause is superseded/unrelated facts in the context |
| CON-05 | Genuine failure: answer says the context has no information (in English) although context exists | open. The question premise ("RFA-REG-01 says annual") is not supported: RFA-REG-01-rev5 line 74 also says semestrial for critical suppliers. Label left unchanged and flagged |

Re-scoring the saved answers with the fixed checker gives **9/10 by the checker, 8/10 judged genuine**. This measures
the checker fix only. The answers are the September ones, and no new generation was run.

## Held-out cases

12 new cases (`HO-*`) cover single-hop, multi-hop, temporal (supersession), a cross-document contradiction and two
negatives. `scripts/validate_heldout_cases.py` proves each label is answerable from the cited text and that each
negative's forbidden term appears nowhere in the corpus. They have **never been run**: running costs LLM credit and
needs an explicit decision. When run, score them once and do not edit the labels afterwards.

## Not production-ready yet: what remains

These need a customer, paid runs or infrastructure, so code alone cannot close them:

* A real quality system (scanned PDFs, Excel control plans, PFMEAs, 8Ds) and the ingestion for those formats.
* Confirmation of the clause mapping by a quality engineer, against the customer's licensed copy of the standard.
* A measured evaluation: run the held-out set, agree acceptance thresholds, and add evidence-recall per clause.
* An identity-provider integration test, a penetration test, data-residency review of the LLM provider, a load test
  and a disaster-recovery drill.
* Publishing the sign-off chain head to external write-once storage.
* An auditor UI on top of these routes.
