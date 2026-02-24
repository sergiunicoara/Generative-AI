"""
Interactive Web Dashboard for RAG Failure Modes
====

Visualize and explore each failure mode with interactive examples.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import os
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

# Failure mode descriptions
FAILURE_MODES = {
    '1': {
        'id': '1',
        'name': 'Chunking Mistakes',
        'title': 'Chunking That Breaks Meaning',
        'description': 'Text splitters break semantic units (e.g., mid-code block), causing retrieval to surface irrelevant or incomplete context.',
        'impact': 'Lower recall, confusing answers, missing critical details',
        'fix_time': '2 hours',
        'symptoms': [
            'Headers separated from content',
            'Code blocks split across chunks',
            'Retrieved passages feel incomplete',
            'High top-k but low usefulness'
        ],
        'solutions': [
            'Use structure-aware splitting (Markdown/HTML)',
            'Recursive chunking with overlap',
            'Preserve code blocks and sections',
            'Tune chunk size by content type'
        ],
        'script': '01_chunking_mistakes.py',
        'category': 'Ingestion'
    },
    '2': {
        'id': '2',
        'name': 'Embedding Mismatch',
        'title': 'Index/Query Embedding Drift',
        'description': 'Index embeddings and query embeddings are produced by different models/settings, degrading similarity search.',
        'impact': 'Poor retrieval quality, silent regressions after model changes',
        'fix_time': '1 day',
        'symptoms': [
            'Sudden drop in recall after a deploy',
            'Same query returns different results across environments',
            'Low similarity scores for obviously relevant docs'
        ],
        'solutions': [
            'Version and pin embedding models',
            'Re-embed corpus on model upgrade',
            'Add embedding config checksums',
            'Validate with offline retrieval tests'
        ],
        'script': '02_embedding_mismatch.py',
        'category': 'Retrieval'
    },
    '3': {
        'id': '3',
        'name': 'Prompt Injection',
        'title': 'Instructions Hidden in Documents',
        'description': 'User inputs or retrieved documents contain malicious instructions that override system behavior.',
        'impact': 'Data exfiltration, policy violations, compromised outputs',
        'fix_time': '1-2 days',
        'symptoms': [
            'Model follows retrieved text instructions',
            'Unexpected tool calls or refusal bypass attempts',
            'Answers include secrets or unrelated content'
        ],
        'solutions': [
            'Treat retrieved text as untrusted',
            'Use a strict system prompt boundary',
            'Strip/neutralize instruction-like content',
            'Add content security filters and allowlists'
        ],
        'script': '03_prompt_injection.py',
        'category': 'Security'
    },
    '4': {
        'id': '4',
        'name': 'Citation Hallucinations',
        'title': 'Fake or Misattributed Sources',
        'description': 'The model fabricates citations or attributes claims to the wrong retrieved chunk.',
        'impact': 'Loss of trust, compliance risk, incorrect decisions',
        'fix_time': '1 day',
        'symptoms': [
            'Citations point to irrelevant text',
            'Sources that do not exist',
            'Confident claims without support'
        ],
        'solutions': [
            'Answer only from retrieved spans',
            'Force cite-then-quote behavior',
            'Post-validate citations against chunks',
            'Return "not found" when unsupported'
        ],
        'script': '04_citation_hallucinations.py',
        'category': 'Generation'
    },
    '5': {
        'id': '5',
        'name': 'Context Window Overflow',
        'title': 'Too Much Context, Too Little Signal',
        'description': 'Retrieval returns too many/too large chunks, causing truncation or burying relevant evidence.',
        'impact': 'Higher latency/cost, worse answers, missing key facts',
        'fix_time': '4 hours',
        'symptoms': [
            'Answers ignore relevant retrieved content',
            'Important evidence appears late and is truncated',
            'Token limit errors or partial prompts'
        ],
        'solutions': [
            'Token-aware retrieval budgeting',
            'Dynamic top-k based on query type',
            'Summarize long contexts',
            'Rerank then compress (map-reduce)'
        ],
        'script': '05_context_window_overflow.py',
        'category': 'Retrieval'
    },
    '6': {
        'id': '6',
        'name': 'Bad Filters',
        'title': 'Metadata Filtering Gone Wrong',
        'description': 'Incorrect metadata filtering breaks retrieval, returning empty results even when relevant content exists.',
        'impact': 'Empty results, user frustration, lost revenue',
        'fix_time': '2 hours',
        'symptoms': [
            'Queries return 0 results despite relevant documents existing',
            'Typos in filter field names',
            'Type mismatches (string vs float)',
            'No fallback when filters fail'
        ],
        'solutions': [
            'Validate filter fields against schema',
            'Automatic type conversion',
            'Fallback to unfiltered search',
            'Log filter effectiveness metrics'
        ],
        'script': '06_bad_filters.py',
        'category': 'Retrieval'
    },
    '7': {
        'id': '7',
        'name': 'Stale Indexes',
        'title': 'Outdated Knowledge Base',
        'description': 'Embeddings/index are not updated as documents change, causing answers based on obsolete information.',
        'impact': 'Incorrect answers, broken trust, operational incidents',
        'fix_time': '1 day',
        'symptoms': [
            'Answers reference old policies or versions',
            'Recently updated docs never appear in retrieval',
            'High mismatch between source-of-truth and RAG answers'
        ],
        'solutions': [
            'Track document timestamps and embedding versions',
            'Incremental re-indexing',
            'Freshness-aware reranking',
            'Automated index health checks'
        ],
        'script': '07_stale_indexes.py',
        'category': 'Indexing'
    },
    '8': {
        'id': '8',
        'name': 'Multilingual Queries',
        'title': 'Cross-Language Retrieval Failure',
        'description': 'Users ask in one language but documents are in another; retrieval misses relevant content.',
        'impact': 'Low recall for non-English users, poor global UX',
        'fix_time': '1 day',
        'symptoms': [
            'Non-English queries return irrelevant English docs',
            'Same question in English works, in Spanish fails',
            'Low similarity between translations'
        ],
        'solutions': [
            'Use multilingual embeddings',
            'Detect language and translate queries',
            'Store language metadata',
            'Evaluate by locale'
        ],
        'script': '08_multilingual_queries.py',
        'category': 'Retrieval'
    },
    '9': {
        'id': '9',
        'name': 'Long-Tail Latency',
        'title': 'P99 Spikes and Timeouts',
        'description': 'The system is fast on average but slow on the tail (P95/P99) due to retrieval complexity, cold starts, or timeouts.',
        'impact': 'Timeouts, poor UX, higher infra cost',
        'fix_time': '2 days',
        'symptoms': [
            'P99 latency is 10x P50',
            'Occasional timeouts under load',
            'Slow queries correlate with large filters or big corpora'
        ],
        'solutions': [
            'Cache frequent queries',
            'Use ANN indexes and optimize top-k',
            'Apply time budgets per stage',
            'Warm critical models and connections'
        ],
        'script': '09_long_tail_latency.py',
        'category': 'Performance'
    },
    '10': {
        'id': '10',
        'name': 'Retrieval Without Reranking',
        'title': 'Similarity vs. Relevance',
        'description': 'Vector similarity alone often surfaces topical but irrelevant chunks; reranking improves precision.',
        'impact': 'Wrong answers, low precision@k, user distrust',
        'fix_time': '1 day',
        'symptoms': [
            'Top-k contains many irrelevant chunks',
            'Answers cite tangential passages',
            'Users complain about "close but not correct"'
        ],
        'solutions': [
            'Two-stage retrieval (bi-encoder then cross-encoder)',
            'Hybrid search (BM25 + vectors)',
            'Diversity constraints (MMR)',
            'Rerank with query-specific features'
        ],
        'script': '10_retrieval_without_reranking.py',
        'category': 'Retrieval'
    }
}


@app.route('/')
def index():
    """Home page with overview of all failure modes."""
    return render_template('index.html', failure_modes=FAILURE_MODES)


@app.route('/failure/<failure_id>')
def failure_detail(failure_id):
    """Detailed view of a specific failure mode."""
    if failure_id not in FAILURE_MODES:
        return "Failure mode not found", 404

    mode = FAILURE_MODES[failure_id]
    return render_template('failure_detail.html', mode=mode)


@app.route('/api/test/<failure_id>', methods=['POST'])
def test_failure(failure_id):
    """API endpoint to test a failure mode with user query."""
    data = request.json
    query = data.get('query', '')

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    if failure_id == '1':  # Chunking Mistakes
        result = {
            'query': query,
            'failure_results': {
                'message': 'Chunks split mid-sentence / mid-code block',
                'docs': [
                    {'title': 'Chunk A: ...def authenticate(user,', 'score': 0.78},
                    {'title': 'Chunk B: password): return token...', 'score': 0.75},
                ]
            },
            'fixed_results': {
                'message': 'Structure-aware splitting preserves full context',
                'docs': [
                    {'title': 'Full function: authenticate(user, password)', 'score': 0.95},
                    {'title': 'Auth module overview', 'score': 0.91},
                ]
            }
        }
    elif failure_id == '2':  # Embedding Mismatch
        result = {
            'query': query,
            'failure_results': {
                'message': 'Index built with model-v1, query uses model-v2 — low similarity',
                'docs': [
                    {'title': 'Unrelated Doc A', 'score': 0.41},
                    {'title': 'Unrelated Doc B', 'score': 0.38},
                ]
            },
            'fixed_results': {
                'message': 'Index and query use same pinned model — high similarity',
                'docs': [
                    {'title': 'API Authentication Guide', 'score': 0.94},
                    {'title': 'OAuth 2.0 Setup', 'score': 0.91},
                ]
            }
        }
    elif failure_id == '3':  # Prompt Injection
        result = {
            'query': query,
            'failure_results': {
                'message': 'Injected instruction in retrieved doc overrides system prompt',
                'docs': [
                    {'title': 'Doc with hidden: "Ignore previous instructions and..."', 'score': 0.88},
                ]
            },
            'fixed_results': {
                'message': 'Retrieved content sanitized — injection neutralized',
                'docs': [
                    {'title': 'Clean API Auth Doc', 'score': 0.93},
                    {'title': 'Security Best Practices', 'score': 0.89},
                ]
            }
        }
    elif failure_id == '4':  # Citation Hallucinations
        result = {
            'query': query,
            'failure_results': {
                'message': 'Model cites non-existent or wrong source',
                'docs': [
                    {'title': 'Hallucinated: "RFC 9999 - Auth Standard"', 'score': 0.80},
                    {'title': 'Wrong chunk cited for claim', 'score': 0.76},
                ]
            },
            'fixed_results': {
                'message': 'Citations validated against retrieved chunks only',
                'docs': [
                    {'title': 'RFC 6749 - OAuth 2.0 (verified)', 'score': 0.96},
                    {'title': 'JWT Best Practices (verified)', 'score': 0.92},
                ]
            }
        }
    elif failure_id == '5':  # Context Window Overflow
        result = {
            'query': query,
            'failure_results': {
                'message': 'Too many chunks retrieved — key evidence truncated',
                'docs': [
                    {'title': 'Chunk 1 (kept)', 'score': 0.85},
                    {'title': 'Chunk 2 (kept)', 'score': 0.83},
                    {'title': 'Chunk 3 — KEY ANSWER (truncated ❌)', 'score': 0.81},
                ]
            },
            'fixed_results': {
                'message': 'Token-aware budgeting — all relevant chunks fit',
                'docs': [
                    {'title': 'Summarized context block 1', 'score': 0.94},
                    {'title': 'Key answer chunk (preserved ✓)', 'score': 0.92},
                ]
            }
        }
    elif failure_id == '6':  # Bad Filters
        result = {
            'query': query,
            'failure_results': {
                'count': 0,
                'message': 'No results found (filter typo: catagory)',
                'docs': []
            },
            'fixed_results': {
                'count': 3,
                'message': 'Found results with corrected filter + fallback',
                'docs': [
                    {'title': 'Product 1', 'score': 0.89},
                    {'title': 'Product 2', 'score': 0.87},
                    {'title': 'Product 3', 'score': 0.85},
                ]
            }
        }
    elif failure_id == '7':  # Stale Indexes
        result = {
            'query': query,
            'failure_results': {
                'message': 'Showing outdated results (index 30 days old)',
                'docs': [
                    {'title': 'Old API v1.0', 'date': '2025-01-15', 'stale': True, 'score': 0.82},
                    {'title': 'Deprecated Feature', 'date': '2025-01-10', 'stale': True, 'score': 0.79},
                ]
            },
            'fixed_results': {
                'message': 'Fresh results (index updated today)',
                'docs': [
                    {'title': 'New API v2.0', 'date': '2026-02-20', 'fresh': True, 'score': 0.95},
                    {'title': 'Latest Features', 'date': '2026-02-22', 'fresh': True, 'score': 0.93},
                ]
            }
        }
    elif failure_id == '8':  # Multilingual
        result = {
            'query': query,
            'failure_results': {
                'message': 'Missing non-English content (monolingual embeddings)',
                'docs': [
                    {'title': 'English Doc 1', 'lang': 'English', 'score': 0.85},
                    {'title': 'English Doc 2', 'lang': 'English', 'score': 0.83},
                ]
            },
            'fixed_results': {
                'message': 'Found content across languages (multilingual embeddings)',
                'docs': [
                    {'title': 'Spanish Doc (translated query)', 'lang': 'Spanish', 'score': 0.92},
                    {'title': 'French Doc', 'lang': 'French', 'score': 0.88},
                    {'title': 'English Doc', 'lang': 'English', 'score': 0.85},
                ]
            }
        }
    elif failure_id == '9':  # Long-Tail Latency
        result = {
            'query': query,
            'failure_latency': {
                'p50': 200,
                'p95': 450,
                'p99': 8500,
                'message': 'P99 latency violates SLA (>3000ms)'
            },
            'fixed_latency': {
                'p50': 2,
                'p95': 180,
                'p99': 420,
                'message': 'Caching + ANN optimization applied',
                'cache_hit': True
            }
        }
    elif failure_id == '10':  # Retrieval Without Reranking
        result = {
            'query': query,
            'failure_results': {
                'message': 'Vector similarity only — 40% relevant',
                'docs': [
                    {'title': 'UI Design for Login', 'relevant': False, 'score': 0.89},
                    {'title': 'Video Streaming Auth', 'relevant': False, 'score': 0.87},
                    {'title': 'API Authentication', 'relevant': True, 'score': 0.82},
                ]
            },
            'fixed_results': {
                'message': 'With cross-encoder reranking — 95% relevant',
                'docs': [
                    {'title': 'API Authentication Guide', 'relevant': True, 'score': 0.95},
                    {'title': 'OAuth 2.0 Implementation', 'relevant': True, 'score': 0.92},
                    {'title': 'JWT Best Practices', 'relevant': True, 'score': 0.88},
                ]
            }
        }
    else:
        return jsonify({'error': f'Unknown failure mode: {failure_id}'}), 404

    return jsonify(result)


@app.route('/metrics')
def metrics():
    """Show production metrics and impact."""
    metrics_data = {
        'bad_filters': {
            'empty_result_rate': {'before': '15%', 'after': '0.5%', 'improvement': '30x'},
            'user_satisfaction': {'before': '2.8/5', 'after': '4.5/5'},
        },
        'stale_indexes': {
            'outdated_info_rate': {'before': '25%', 'after': '2%', 'improvement': '12.5x'},
            'support_tickets': {'before': '145/day', 'after': '12/day'},
        },
        'multilingual': {
            'non_english_recall': {'before': '45%', 'after': '89%', 'improvement': '2x'},
            'international_satisfaction': {'before': '2.1/5', 'after': '4.3/5'},
        },
        'latency': {
            'p99_latency': {'before': '3500ms', 'after': '450ms', 'improvement': '7.8x'},
            'timeout_rate': {'before': '5%', 'after': '0.1%'},
        },
        'reranking': {
            'relevance_rate': {'before': '40%', 'after': '94%', 'improvement': '2.35x'},
            'answer_quality': {'before': '72%', 'after': '94%'},
        }
    }
    return render_template('metrics.html', metrics=metrics_data)


if __name__ == '__main__':
    print("=" * 80)
    print("RAG Failure Modes Dashboard")
    print("=" * 80)
    print("\nStarting server...")
    print("Open your browser to: http://localhost:5000")
    print("\nPress Ctrl+C to stop")
    print("=" * 80)

    app.run(debug=True, host='0.0.0.0', port=5000)
