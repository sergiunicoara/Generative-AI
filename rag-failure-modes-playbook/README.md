# RAG Failure Modes Playbook: The Complete 10-Case Edition

## 📊 Project Overview
This repository is a battle-tested guide to identifying and fixing the most common failure modes in Retrieval-Augmented Generation (RAG) systems. It contains **10 runnable Python examples**, a **web dashboard**, and **production-ready fixes**.

---

## 🎯 The 10 Failure Modes

### Failure Mode #1: Chunking Mistakes
- **The Problem**: Text splitters breaking semantic units (e.g., mid-code block).
- **The Fix**: Context-aware chunking, recursive character splitting.
- **Run it**: `python 01_chunking_mistakes.py`

### Failure Mode #2: Embedding Mismatch
- **The Problem**: Using different models for indexing vs. retrieval.
- **The Fix**: Strict model versioning and embedding validation.
- **Run it**: `python 02_embedding_mismatch.py`

### Failure Mode #3: Prompt Injection
- **The Problem**: Malicious user input overriding RAG instructions.
- **The Fix**: Input sanitization, system prompt hardening, and guardrails.
- **Run it**: `python 03_prompt_injection.py`

### Failure Mode #4: Citation Hallucinations
- **The Problem**: LLM providing fake sources or misattributing facts.
- **The Fix**: Strict citation enforcement and post-retrieval verification.
- **Run it**: `python 04_citation_hallucinations.py`

### Failure Mode #5: Context Window Overflow
- **The Problem**: Retrieving too much data, cutting off critical context.
- **The Fix**: Token counting, dynamic top-k, and summarization.
- **Run it**: `python 05_context_window_overflow.py`

### Failure Mode #6: Bad Filters
- **The Problem**: Metadata filter typos or logic errors breaking retrieval.
- **The Fix**: Schema validation and fallback search logic.
- **Run it**: `python 06_bad_filters.py`

### Failure Mode #7: Stale Indexes
- **The Problem**: Outdated embeddings missing recent critical updates.
- **The Fix**: Incremental indexing and freshness-based reranking.
- **Run it**: `python 07_stale_indexes.py`

### Failure Mode #8: Multilingual Queries
- **The Problem**: Cross-language retrieval failing to find relevant docs.
- **The Fix**: Multilingual embeddings and query translation.
- **Run it**: `python 08_multilingual_queries.py`

### Failure Mode #9: Long-Tail Latency
- **The Problem**: P99 spikes due to complex retrieval or large clusters.
- **The Fix**: Query caching, ANN optimization, and timeouts.
- **Run it**: `python 09_long_tail_latency.py`

### Failure Mode #10: Retrieval Without Reranking
- **The Problem**: Vector similarity failing to capture true relevance.
- **The Fix**: Two-stage retrieval with Cross-Encoder reranking.
- **Run it**: `python 10_retrieval_without_reranking.py`


---

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Launch the Dashboard
Explore all 10 cases visually:
```bash
python dashboard/app.py
```

### 3. Run All Examples
```bash
./run_all_examples.bat
```

## 🛠️ Documentation Files
- **PROJECT_SUMMARY.md**: High-level overview of all deliverables.
- **docs/QUICK_START.md**: Detailed setup instructions.

---
**Built for production reliability.** 🚀
