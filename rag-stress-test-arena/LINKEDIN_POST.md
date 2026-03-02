# LinkedIn Post — copy-paste ready (Unicode bold renders correctly on LinkedIn)

---

𝗠𝗼𝘀𝘁 𝘃𝗲𝗰𝘁𝗼𝗿 𝗗𝗕 𝗯𝗲𝗻𝗰𝗵𝗺𝗮𝗿𝗸𝘀 𝗮𝗿𝗲 𝗺𝗲𝗮𝘀𝘂𝗿𝗶𝗻𝗴 𝗮 𝗳𝗮𝗻𝘁𝗮𝘀𝘆.

They test on random vectors.
Production RAG systems run on real text — clustered, overlapping, and full of hard negatives.

I ran a full stress test on 4 engines:
Qdrant, Elasticsearch, Redis, pgvector

20,000 real Wikipedia embeddings. Here's what actually matters in production.

All tests: same hardware, warm caches, fixed embedding model, identical HNSW parameters.

――――――――――――――――――――――

🚀 Speed under load
(single node, 20k vectors, independent clients)

Redis peaks at 𝟵𝟭𝟴 req/sec.
pgvector reaches 𝟭𝟲𝟰.
Qdrant 𝟭𝟮𝟬.
Elasticsearch 𝟲𝟭 — on a single node.

Yes, Elasticsearch scales horizontally.
But most small-to-mid RAG deployments start on one node — and that's where Redis wins without extra infrastructure.

𝗢𝗻𝗲 𝘁𝗵𝗶𝗻𝗴 𝘁𝗵𝗮𝘁 𝘀𝘂𝗿𝗽𝗿𝗶𝘀𝗲𝗱 𝗺𝗲:
pgvector's single-thread latency dropped from ~𝟱𝟴𝗺𝘀 to 𝟭𝟴𝗺𝘀 once each client used a persistent connection instead of reconnecting per query.

The old number was penalising connection overhead, not search quality.

――――――――――――――――――――――

🎯 Will it find the right answer?

On fake benchmark data, all engines hit 𝟭𝟬𝟬% recall.

On real data?
Redis and pgvector cap at ~𝟵𝟵.𝟴% recall under standard HNSW build params (m=16, ef_construction=100).

Real topics create "hard negatives" that are genuinely difficult to resolve without rebuilding the index with higher-quality params.

→ That's real users occasionally getting the wrong document — worth knowing before you go live.

――――――――――――――――――――――

✍️ What happens when your index gets updated?

Every production RAG system has continuous writes — new documents, deletions, updates.

At our test scale (10k → 15k vectors with deletes), Elasticsearch latency jumped +𝟰𝟰%.

Redis and Qdrant actually got faster — their internal graph rebalancing improved traversal at this size.

→ This is the test nobody runs before going live.

――――――――――――――――――――――

🔍 Using a reranker (cross-encoder)?

With a 𝟱𝟬𝗺𝘀 reranker in the pipeline:
• Redis: 𝟱𝟱𝗺𝘀 end-to-end
• Elasticsearch: 𝟭𝟱𝟴𝗺𝘀 end-to-end

When your reranker is slow (≥30ms), it dominates the latency budget.
Fast retrieval still matters — it compounds.

If you swap in a 𝟱𝗺𝘀 reranker, engine choice suddenly matters a lot more again.

――――――――――――――――――――――

The practical takeaway

• 𝗛𝗶𝗴𝗵 𝘁𝗿𝗮𝗳𝗳𝗶𝗰, 𝘀𝗶𝗻𝗴𝗹𝗲 𝗻𝗼𝗱𝗲, 𝗰𝗼𝗿𝗽𝘂𝘀 𝗳𝗶𝘁𝘀 𝗶𝗻 𝗥𝗔𝗠?
→ Redis — blazing fast, minimal infra (watch memory pressure as you scale)

• 𝗡𝗲𝗲𝗱 𝘁𝗵𝗲 𝗯𝗲𝘀𝘁 𝗿𝗲𝗰𝗮𝗹𝗹 + 𝗿𝗲𝗹𝗶𝗮𝗯𝗶𝗹𝗶𝘁𝘆?
→ Qdrant — strong quality on real data, stable under index churn

• 𝗔𝗹𝗿𝗲𝗮𝗱𝘆 𝗿𝘂𝗻𝗻𝗶𝗻𝗴 𝗣𝗼𝘀𝘁𝗴𝗿𝗲𝘀?
→ pgvector — zero new infrastructure, fragmentation-neutral; a reranker closes most of the latency gap

• 𝗔𝗹𝗿𝗲𝗮𝗱𝘆 𝗼𝗻 𝘁𝗵𝗲 𝗘𝗹𝗮𝘀𝘁𝗶𝗰 𝘀𝘁𝗮𝗰𝗸 (𝗺𝘂𝗹𝘁𝗶-𝗻𝗼𝗱𝗲)?
→ Elasticsearch — throughput scales out, but watch write fragmentation per shard

• 𝗔𝗱𝗱𝗶𝗻𝗴 𝗮 𝗿𝗲𝗿𝗮𝗻𝗸𝗲𝗿 ≥𝟯𝟬𝗺𝘀?
→ Retrieval speed still compounds — don't ignore it

――――――――――――――――――――――

Full benchmark open source on GitHub: [link]

𝗪𝗵𝗮𝘁 𝘃𝗲𝗰𝘁𝗼𝗿 𝗗𝗕 𝗮𝗿𝗲 𝘆𝗼𝘂 𝗿𝘂𝗻𝗻𝗶𝗻𝗴 𝗶𝗻 𝗽𝗿𝗼𝗱𝘂𝗰𝘁𝗶𝗼𝗻?
Curious what tradeoffs you've hit 👇

#RAG #vectordatabases #LLM #AIengineering #machinelearning
