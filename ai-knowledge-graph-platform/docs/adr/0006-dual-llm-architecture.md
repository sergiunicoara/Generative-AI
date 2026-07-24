# ADR-0006 — Dual-LLM architecture: fast routing + quality synthesis

**Status:** Accepted  
**Date:** 2026-06-02  
**Author:** Sergiu Nicoara

---

## Context

The agentic IRCoT retriever (Interleaved Retrieval and Chain-of-Thought) runs a multi-step loop:

```
for step in range(max_steps):
    reasoning = await llm("Can you answer? If not, what do you still need to search for?")
    if "ANSWER:" in reasoning:
        return synthesise(reasoning)          # final answer
    sub_query = parse_search_query(reasoning)
    new_chunks = retrieve(sub_query)          # another retrieval pass
    context += new_chunks

final_answer = await llm(FINAL_PROMPT.format(context=context, question=question))
```

Initially, every LLM call in this loop used the same model: `llama-3.3-70b-versatile`. This is correct for the final synthesis step but wasteful for the intermediate reasoning steps.

The intermediate step asks the model to emit one of two structured tokens:
- `SEARCH: <sub-query>` (~15 output tokens)
- `ANSWER: <short text>` (~20 output tokens)

This is a classification / structured extraction task, not a generation task. It does not require a 70B model.

---

## Decision

Split the agentic loop into two model tiers:

| Step | Model | Rationale |
|---|---|---|
| Reasoning steps (SEARCH/ANSWER routing) | `llama-3.1-8b-instant` | Trivial structured output; speed dominates |
| Final synthesis | `llama-3.3-70b-versatile` | User-facing answer; quality dominates |

Implementation in `graphrag/retrieval/agentic_retriever.py`:

```python
async def _reason(self, prompt: str) -> str:
    """Fast 8B model for intermediate SEARCH/ANSWER routing."""
    return await get_fast_llm().generate(prompt)

async def _synthesize(self, prompt: str) -> str:
    """Full 70B model for final user-facing synthesis."""
    return await get_llm().generate(prompt)
```

`get_fast_llm()` points at `llama-3.1-8b-instant` (config: `groq_fast_model`).  
`get_llm()` points at `llama-3.3-70b-versatile` (config: `groq_model`).

`max_steps` reduced from 4 to 2. Empirically, most hard queries resolve within 2 iterations; steps 3–4 rarely surface chunks not already in context.

---

## Latency impact

For a typical 2-step agentic query:

| Stage | Old (70B all) | New (split) |
|---|---|---|
| Initial retrieval | ~0.5s | ~0.5s |
| Reasoning step 1 (8B) | 1.5s | 0.2s |
| Sub-retrieval | ~0.5s | ~0.5s |
| Reasoning step 2 (8B) | 1.5s | 0.2s |
| Final synthesis (70B) | ~1.5s | ~1.5s |
| **Total** | **~5.5s** | **~2.9s** |

Measured p95 improvement: **6.8s → 3.4s** (−50%).  
Combined p95 across hybrid and agentic: **5.9s → 2.7s** (under the 3s SLO).

---

## Considered alternatives

### Option A — Use 8B for everything including synthesis

- Maximum speed, minimum cost
- Rejected: synthesis quality degrades measurably on multi-hop reasoning questions where the final answer requires integrating evidence from 3–4 chunks. Faithfulness dropped from 0.840 to ~0.71 in informal testing.

### Option B — Keep 70B for everything

- Maximum quality
- Rejected: p95 latency exceeds the 3s SLO and the quality improvement for routing decisions is negligible (they're trivially structured outputs)

### Option C — Model cascade (try 8B, escalate if uncertain)

- Use 8B for routing; if it produces malformed output, retry with 70B
- More complex; adds latency on escalation; routing output is so constrained (`SEARCH:` or `ANSWER:`) that malformation is rare
- Rejected: unnecessary complexity for the current traffic pattern

---

## Consequences

**Positive:**
- Agentic p95 within SLO for the first time
- Cost reduction: 8B inference on Groq is free-tier; two 8B calls per query instead of two 70B calls
- Configurable: `groq_fast_model` in `settings.yml` allows swapping the routing model without changing code

**Negative / watch:**
- Model provenance: only the synthesis model (`groq_model`) is surfaced in `QueryResult.model_version`. The routing model is invisible in the audit trail. If debugging is needed, check `agentic_retriever.reason` logs at DEBUG level.
- If the 8B model is replaced with one that produces different structured output format (`SEARCH:` / `ANSWER:`), the parser in `agentic_retriever.py` must be updated in parallel.

**Extension points:**
- The same pattern applies to any multi-step pipeline where intermediate steps are structured and the final step is generative: ingestion extraction (8B for entity detection, 70B for relation extraction), evaluation (8B for classification, 70B for explanation).

---

## Update 2026-07-24 — synthesis model changed from Groq 70B to DeepSeek

This ADR's split (fast 8B for routing, larger model for synthesis) is still the
current architecture and remains correct. What changed is which provider backs
the *synthesis* tier: `get_llm()` in `graphrag/core/llm_client.py` now defaults
to a bare `DeepSeekLLM` (`deepseek-v4-pro`), not Groq's `llama-3.3-70b-versatile`
as originally implemented and described above. Groq generation is now an
opt-in override only, via `LLM_INGEST_PROVIDER=groq`.

`get_fast_llm()` — the routing tier discussed throughout this ADR — is
unaffected: it still defaults to Groq's `llama-3.1-8b-instant` (DeepSeek
fallback), so the "8B routing / large-model synthesis" split and the latency
analysis above remain accurate. Only the specific model name behind "final
synthesis" in the table and code comments should now be read as DeepSeek,
not `llama-3.3-70b-versatile`.
