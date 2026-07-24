# ADR-0004 — Groq for text generation, OpenAI for embeddings

**Status:** Amended (embeddings provider updated 2026-06-03)  
**Date:** 2026-05-30  
**Author:** Sergiu Nicoara

---

## Context

The platform requires two distinct LLM capabilities:

1. **Text generation** — extraction, synthesis, reasoning steps, RAGAS evaluation judge
2. **Embeddings** — 3072d vector representations for ANN search and entity resolution

Initially both were served by Google Gemini (`gemini-2.0-flash` for generation, `gemini-embedding-001` for embeddings). Under load, Gemini's free-tier rate limits (15 RPM / 1M TPM on Flash) became the ingestion bottleneck — a moderate corpus would exhaust daily quota before completing.

*Note: A subsequent amendment (2026-06-03) also migrated embeddings from Gemini to OpenAI `text-embedding-3-large` (3072d). See the Amendment section at the end of this document.*

---

## Decision

Split the providers:

- **Embeddings:** Keep `gemini-embedding-001` (3072d, stable API, high quality on technical text)
- **Text generation:** Switch to `groq/llama-3.3-70b-versatile` for synthesis and `groq/llama-3.1-8b-instant` for routing decisions

Rationale for each choice:

### Why keep Gemini for embeddings

- The Neo4j vector index is created at 3072 dimensions. Changing embedding providers requires re-embedding every chunk and rebuilding the index — an expensive, risky migration.
- `gemini-embedding-001` produces high-quality embeddings for domain-specific technical text (aerospace regulatory, financial, medical).
- Embedding calls are low-frequency compared to generation calls (one per chunk at ingestion time, not per query).
- The embedding API has separate quota from the generation API, so they don't compete.

### Why Groq for text generation

- **Quota:** Groq free tier offers 1,500 RPD / 6,000 RPM on llama-3.3-70b — sufficient for development and light production use without depleting budget.
- **Speed:** Groq runs on custom LPU hardware. llama-3.3-70b at ~150 tok/s vs Gemini Flash at ~60 tok/s; llama-3.1-8b-instant at ~800 tok/s.
- **Two-model design:** Groq's fast inference makes the 8B routing + 70B synthesis split economically viable. Each 8B reasoning step costs ~0.2s; the same step on a 70B model costs ~1.5s. This is the primary driver of the agentic path p95 dropping from 6.8s to 3.4s.
- **Decoupled quota:** Swapping generation providers does not affect the embedding index. Rolling back generation to Gemini is a single function change in `llm_client.py`.

### Why not a single provider for both

- No single provider offers both a competitive 3072d embedding model AND a fast inference endpoint on the same free/low-cost tier.
- Provider decoupling is an architectural feature: `get_embedder()` and `get_llm()` / `get_fast_llm()` are independent. A client deployment can point each at different internal endpoints without touching retrieval or graph logic.

---

## Two-model design (extension of this ADR)

Within Groq, the agentic IRCoT path uses two models:

| Call site | Model | Rationale |
|---|---|---|
| Routing steps (SEARCH/ANSWER) | `llama-3.1-8b-instant` | ~15 output tokens; speed matters, quality doesn't |
| Final synthesis | `llama-3.3-70b-versatile` | Quality matters; this is the user-facing answer |

Configured via `groq_model` and `groq_fast_model` in `settings.yml`.

---

## Consequences

**Positive:**
- Ingestion throughput is no longer Gemini-quota-limited
- Agentic path p95 reduced by ~50%
- Provider swap is a single-function change

**Negative / watch:**
- The RAGAS evaluation judge (`ragas_evaluator.py`) uses Groq with DeepSeek-V3 as fallback — if Groq quota is exhausted, RAGAS samples fail silently (logged as WARNING)
- Model attribution in result provenance reflects the synthesis model (`groq_model`); routing model is not surfaced in citations

**Migration path for client deployments:**
Change `get_llm()`, `get_fast_llm()`, and `get_embedder()` in `graphrag/core/llm_client.py`. Nothing else requires modification.

---

## Amendment — 2026-06-03: Embeddings migrated from Gemini to OpenAI

### Context

After ADR-0004 was accepted, the Google Gemini API key was revoked, removing access to `gemini-embedding-001`. The embeddings provider was migrated to OpenAI `text-embedding-3-large`.

### Decision

- **Embeddings:** Switch from `gemini-embedding-001` to `openai/text-embedding-3-large` (3072d)
- **Generation fallback:** Switch from Gemini Flash to DeepSeek-V3 (`deepseek-chat`) via OpenAI-compatible API

### Why OpenAI for embeddings

- `text-embedding-3-large` also produces 3072d vectors — no schema migration required; the existing Neo4j vector index is fully compatible
- Switching was a single-line change in `get_embedder()` in `llm_client.py`
- Cost: ~$0.13/1M tokens (negligible for the 12-doc corpus)
- Quality on technical text is at least equivalent to `gemini-embedding-001`

### Why DeepSeek-V3 as generation fallback

- OpenAI-compatible REST API — the existing `openai` SDK is reused
- Generous rate limits; instant failover on Groq 429 with no sleep required
- Cost: ~$0.07/1M input tokens (cheaper than Gemini Flash)

### Impact

No behavioral change to the retrieval pipeline. Vector dimensions unchanged. All 367 entities and 380 edges re-embedded and re-indexed without schema changes.

---

## Update 2026-07-24 — DeepSeek became the default primary generation engine

The decision above (Groq for synthesis, `llama-3.1-8b-instant` for routing) described
the architecture as of 2026-05-30/2026-06-03. Since then, `get_llm()` in
`graphrag/core/llm_client.py` was changed to default to a bare `DeepSeekLLM`
(`deepseek-v4-pro`) rather than Groq: one provider's extraction/synthesis voice
for the whole corpus, with no Groq round-trip. Groq is now an **opt-in override
only**, selected via `LLM_INGEST_PROVIDER=groq`, intended for quick, low-volume
dev runs.

`get_fast_llm()` — the separate, smaller model used only for the agentic
retriever's intermediate SEARCH/ANSWER routing decisions — is unaffected by this
change and still defaults to Groq's `llama-3.1-8b-instant` (with DeepSeek
fallback), consistent with the "Two-model design" section above.

The original reasoning in this ADR (why Groq was chosen over Gemini at the time,
the quota/speed tradeoffs) remains an accurate record of that decision and is
left unchanged. This update only corrects which provider is primary for
generation *today*.
