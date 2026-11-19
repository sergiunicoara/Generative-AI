# Golden-set miss classes

Failed rows per profile and class (all valid runs, 34 questions each):

| class | vector_only | text_hybrid | full | full_context |
|---|---|---|---|---|
| citation_uncitable_artifact | 9 | 12 | 0 | 11 |
| retrieval_miss_citation | 0 | 3 | 0 | 0 |
| citation_only | 0 | 0 | 0 | 7 |
| refused_with_evidence | 1 | 3 | 3 | 0 |
| refused_no_evidence | 7 | 3 | 0 | 3 |
| retrieval_miss | 10 | 6 | 3 | 0 |
| answer_error | 7 | 3 | 12 | 5 |
| **failed rows** | **34** of 102 | **30** of 102 | **18** of 102 | **26** of 102 |

Most frequent class per question (failed runs only):

| profile | question: class |
|---|---|
| vector_only | AGT-01: refused_no_evidence, AUT-01: retrieval_miss, AUT-02: retrieval_miss, AUT-03: citation_uncitable_artifact, CAL-01: answer_error, CON-01: answer_error, CTX-01: retrieval_miss, INF-01: citation_uncitable_artifact, MH-02: refused_with_evidence, MH-03: answer_error, NEG-03: answer_error, PRE-01: citation_uncitable_artifact, TMP-01: refused_no_evidence, TMP-03: refused_no_evidence |
| text_hybrid | AUT-01: retrieval_miss, AUT-02: citation_uncitable_artifact, AUT-03: retrieval_miss_citation, CAL-01: answer_error, INF-01: citation_uncitable_artifact, MH-02: refused_with_evidence, MH-03: retrieval_miss, PRE-01: citation_uncitable_artifact, TMP-01: citation_uncitable_artifact, TMP-03: refused_no_evidence |
| full | AGT-02: answer_error, AUT-01: answer_error, CAL-01: answer_error, MH-03: retrieval_miss, SH-05: answer_error, TMP-02: answer_error, TMP-03: refused_with_evidence |
| full_context | AUT-01: answer_error, AUT-02: citation_uncitable_artifact, AUT-03: citation_uncitable_artifact, INF-01: citation_uncitable_artifact, MH-02: answer_error, MH-03: citation_only, NEG-03: answer_error, PRE-01: citation_uncitable_artifact, TMP-01: citation_only, TMP-03: refused_no_evidence |

Limitation: Classes are document-level proxies. The retrieved context text is not stored (only document ids), so 'right document, wrong chunk' is counted as answer_error. A graph or schema gap cannot be decided from stored data (the graph state records counts only) and is never assigned. For full_context the whole corpus is in the prompt, so a content failure is an answer error by construction.
