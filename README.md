# Generative AI — Project Portfolio

Production-grade LLM and retrieval systems, built and measured end-to-end.
Each project below is self-contained with its own README, tests, and run instructions.

---

## 🏗 Flagship: [AI Knowledge Graph Platform](./AI%20knowledge%20graph%20platform)

**Production GraphRAG + knowledge graph platform for regulatory compliance.**

- 6-stage hybrid retrieval: vector ANN + BM25 → RRF fusion → cross-encoder reranking → multi-hop graph traversal → GNN (GAT) scoring → LLM synthesis with citations
- Agentic IRCoT fallback (dual-model: 8B routing + 70B synthesis) for low-confidence answers
- Knowledge engineering: LLM entity/relation extraction, 4-stage alias deduplication, Leiden community detection, contradiction detection (5 types), document authority + supersession chains
- Measured, not claimed: RAGAS faithfulness **0.940** on a 39-question golden set, hybrid p95 2.2s, 364 passing unit tests
- Operations: multi-tenant Neo4j, RabbitMQ workers, Redis caching, live observability dashboards, GDPR erasure, confidence calibration (Brier + isotonic regression)
- 6 Architecture Decision Records documenting the trade-offs

**Stack:** Python · Neo4j · FastAPI · RabbitMQ · Redis · Groq/DeepSeek · OpenAI embeddings · RAGAS · Plotly Dash

---

## 🔬 [RAG Failure Modes Playbook](./rag-failure-modes-playbook)

10 ways RAG silently fails — each with a runnable failure script and the engineered fix.
Chunking mistakes, embedding mismatch, prompt injection, citation hallucination, context overflow, stale indexes, and more.

## ⚡ [RAG Stress Test Arena](./rag-stress-test-arena)

Reproducible vector database benchmark: Qdrant, Elasticsearch, Redis, pgvector under identical conditions (same hardware, fixed HNSW params, real embeddings). Recall@K, p95 latency, throughput under concurrent load, index-churn behavior.

---

## Smaller projects & notes

Earlier explorations kept for reference: document Q&A chatbots, LLaMA fine-tuning, domain-specific supervised fine-tuning, OpenAI Vision calorie tracker, Gradio chat interfaces, Pydantic patterns.

---

📫 [LinkedIn — Sergiu Nicoara](https://www.linkedin.com/in/sergiu-nicoara)
