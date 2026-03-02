# LinkedIn Post

---

Most vector DB benchmarks are measuring a fantasy. Here's what happens on real data.

They test with random fake data. Your production RAG system runs on real text — which clusters, overlaps, and behaves completely differently.

I ran a full stress test on 4 engines (Qdrant, Elasticsearch, Redis, pgvector) using 20,000 real Wikipedia embeddings. Here's what actually matters when you ship to production:

---

**🚀 Speed under load? (single node, 20k vectors, independent clients)**
Redis peaks at 918 req/sec. pgvector reaches 164. Qdrant 120. Elasticsearch 61 — on a single node.
Yes, ES scales horizontally. But most small-to-mid RAG deployments start on one node — and that's where Redis wins without extra infrastructure.
One thing that surprised me: pgvector's single-thread latency dropped from ~58ms to 18ms once each client used a persistent connection instead of reconnecting per query. The old number was penalising connection overhead, not search quality.

**🎯 Will it find the right answer?**
On fake benchmark data, all engines hit 100% recall.
On real data? Redis and pgvector cap at ~99.8% — at standard HNSW build params (m=16, ef_construction=100). Real topics create "hard negatives" that are genuinely difficult to resolve without rebuilding the index with higher-quality params.
→ That's real users occasionally getting the wrong document — worth knowing before you go live.

**✍️ What happens when your index gets updated?**
Every production RAG system has continuous writes — new documents, deletions, updates.
At our test scale (10k → 15k vectors with deletes), Elasticsearch latency jumped +44%.
Redis and Qdrant actually got faster — their internal graph rebalancing improved traversal at this size.
→ This is the test nobody runs before going live.

**🔍 Using a reranker (cross-encoder) for better quality?**
With a 50ms reranker in the pipeline:
- Redis: 55ms end-to-end
- Elasticsearch: 158ms end-to-end
→ When your reranker is slow (≥30ms), it dominates the latency budget. Fast retrieval still matters — it compounds. But if you swap in a 5ms reranker, engine choice matters a lot more again.

---

**The practical takeaway:**

* High traffic, single node, corpus fits in RAM? → **Redis** — Blazing fast, minimal infra — just watch memory pressure as you scale.
* Need the best recall + reliability? → **Qdrant** — Strong quality under real-world data and stable behavior under index churn.
* Already running Postgres? → **pgvector** — Zero new infrastructure, fragmentation-neutral; a reranker closes most of the latency gap with purpose-built engines.
* Already on the Elastic stack (multi-node)? → **Elasticsearch** — Throughput scales out well — but watch write fragmentation on each shard.
* Adding a reranker ≥30ms? — Retrieval speed still compounds — don't ignore it.

Full benchmark open source on GitHub: [link]

---

What vector DB are you running in production? Curious what tradeoffs you've hit. 👇

#RAG #vectordatabases #LLM #AIengineering #machinelearning
