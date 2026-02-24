# Deployment Guide: RAG Failure Modes Playbook (10-Case Edition)

## Overview
This guide covers deploying the interactive dashboard and the 10 failure mode examples.

## The 10 Production Scenarios
The playbook is now expanded to cover 10 distinct scenarios:
- 1. Chunking Mistakes (01_chunking_mistakes.py)
- 2. Embedding Mismatch (02_embedding_mismatch.py)
- 3. Prompt Injection (03_prompt_injection.py)
- 4. Citation Hallucinations (04_citation_hallucinations.py)
- 5. Context Window Overflow (05_context_window_overflow.py)
- 6. Bad Filters (06_bad_filters.py)
- 7. Stale Indexes (07_stale_indexes.py)
- 8. Multilingual Queries (08_multilingual_queries.py)
- 9. Long-Tail Latency (09_long_tail_latency.py)
- 10. Retrieval Without Reranking (10_retrieval_without_reranking.py)

## Deployment Steps
1. **Environment Setup**: `pip install -r requirements.txt`
2. **Data Ingestion**: Run scripts 01-05 to understand ingestion failures.
3. **Production Scaling**: Run scripts 06-10 to understand retrieval and latency issues.
4. **Dashboard Launch**: `python dashboard/app.py`
