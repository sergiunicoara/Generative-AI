"""
RAG Failure Mode #5: Retrieval Without Reranking
=================================================

Problem: Relying solely on vector similarity leads to poor document selection,
irrelevant context, and degraded answer quality.

Common Issues:
- Vector similarity ≠ relevance for answering
- Embeddings capture semantic meaning but miss query intent
- No second-stage refinement
- All top-k documents used regardless of quality
- Missing keyword/BM25 signals
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Dict, Tuple
import os
import re
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


def print_section(title: str, color=Fore.CYAN):
    """Print a formatted section header."""
    print(f"\n{color}{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}{Style.RESET_ALL}\n")


def load_technical_docs() -> str:
    """Load technical documentation."""
    with open('sample_data/technical_documentation.txt', 'r') as f:
        return f.read()


def create_diverse_documents() -> List[Document]:
    """Create documents with varying relevance to test reranking."""
    
    documents = [
        # Highly relevant documents
        Document(
            page_content="""
            Authentication Best Practices
            
            When implementing authentication, always use HTTPS to encrypt credentials
            in transit. Store passwords using bcrypt or Argon2 hashing algorithms.
            Never store passwords in plain text. Implement rate limiting to prevent
            brute force attacks. Use JWT tokens with short expiration times.
            Rotate API keys regularly and never commit them to version control.
            """,
            metadata={'doc_id': 'auth_001', 'category': 'security', 'relevance': 'high'}
        ),
        Document(
            page_content="""
            API Authentication Methods
            
            Modern APIs support multiple authentication methods:
            1. API Keys: Simple but less secure, use for server-to-server
            2. OAuth 2.0: Industry standard, great for third-party access
            3. JWT: Stateless tokens, perfect for microservices
            4. Basic Auth: Simple but requires HTTPS
            Each method has trade-offs between security and convenience.
            """,
            metadata={'doc_id': 'auth_002', 'category': 'security', 'relevance': 'high'}
        ),
        
        # Medium relevance - related but not directly answering
        Document(
            page_content="""
            Rate Limiting Implementation
            
            Rate limiting protects your API from abuse. Implement sliding window
            algorithms to track request counts. Return HTTP 429 when limits exceeded.
            Use Redis for distributed rate limiting. Consider different limits
            for different user tiers. Document your rate limits clearly in API docs.
            """,
            metadata={'doc_id': 'ratelimit_001', 'category': 'performance', 'relevance': 'medium'}
        ),
        Document(
            page_content="""
            API Documentation Standards
            
            Good API documentation includes: endpoint descriptions, request/response
            examples, error codes, rate limits, and authentication requirements.
            Use OpenAPI/Swagger for interactive docs. Provide code examples in
            multiple languages. Keep documentation in sync with code changes.
            """,
            metadata={'doc_id': 'docs_001', 'category': 'documentation', 'relevance': 'medium'}
        ),
        
        # Low relevance - semantically similar but wrong topic
        Document(
            page_content="""
            User Interface Design for Login Screens
            
            A good login screen has clear labels, proper error messages, and
            password visibility toggles. Use material design principles. Ensure
            accessibility with proper ARIA labels. Support password managers.
            Include "forgot password" and "remember me" options. Mobile responsive.
            """,
            metadata={'doc_id': 'ui_001', 'category': 'frontend', 'relevance': 'low'}
        ),
        Document(
            page_content="""
            Database Schema for User Accounts
            
            User tables should include: id (primary key), email (unique), password_hash,
            created_at, updated_at, last_login, is_active. Use foreign keys for
            related tables. Index email and username fields. Consider soft deletes.
            Separate PII into encrypted tables for GDPR compliance.
            """,
            metadata={'doc_id': 'db_001', 'category': 'database', 'relevance': 'low'}
        ),
        
        # Very low relevance - contains keywords but wrong context
        Document(
            page_content="""
            API Testing with Authentication
            
            When testing authenticated endpoints, first obtain a valid token.
            Store the token in test fixtures. Use mocks for auth services in unit tests.
            Test both valid and invalid tokens. Verify proper 401/403 responses.
            Use tools like Postman or pytest for automated testing.
            """,
            metadata={'doc_id': 'test_001', 'category': 'testing', 'relevance': 'low'}
        ),
        
        # Edge case - keyword match but irrelevant
        Document(
            page_content="""
            Authentication in Video Streaming
            
            Streaming services use DRM and token-based authentication. MPEG-DASH
            and HLS protocols support encrypted streams. Use signed URLs that expire.
            Implement geo-blocking based on user location. Monitor for account sharing.
            Use CDN with authentication at the edge.
            """,
            metadata={'doc_id': 'stream_001', 'category': 'media', 'relevance': 'very_low'}
        ),
    ]
    
    return documents


def calculate_keyword_score(query: str, document: str) -> float:
    """Simple BM25-style keyword scoring."""
    query_terms = set(query.lower().split())
    doc_terms = document.lower().split()
    
    # Term frequency in document
    tf_scores = []
    for term in query_terms:
        count = doc_terms.count(term)
        if count > 0:
            # Simple TF calculation
            tf = count / len(doc_terms)
            tf_scores.append(tf)
    
    return sum(tf_scores) if tf_scores else 0.0


def calculate_cross_encoder_score(query: str, document: str) -> float:
    """
    Simulate cross-encoder scoring (query-document interaction).
    In production, use actual cross-encoder models like:
    - cross-encoder/ms-marco-MiniLM-L-6-v2
    - cross-encoder/ms-marco-TinyBERT-L-2
    """
    # Simulate: check for exact query intent match
    query_lower = query.lower()
    doc_lower = document.lower()
    
    score = 0.0
    
    # Exact phrase match (strong signal)
    if query_lower in doc_lower:
        score += 0.5
    
    # Key term matching with context
    query_terms = query_lower.split()
    for i, term in enumerate(query_terms):
        if term in doc_lower:
            # Check if nearby terms also match (context)
            context_match = 0
            if i > 0 and query_terms[i-1] in doc_lower:
                context_match += 1
            if i < len(query_terms)-1 and query_terms[i+1] in doc_lower:
                context_match += 1
            
            score += 0.1 + (context_match * 0.05)
    
    # Length penalty (very short or very long docs score lower)
    doc_length = len(doc_lower.split())
    if doc_length < 20 or doc_length > 500:
        score *= 0.8
    
    return min(score, 1.0)


def run_failure():
    """
    Demonstrates RETRIEVAL WITHOUT RERANKING failures.
    
    Problems:
    1. Top-k based solely on vector similarity
    2. No query-document interaction modeling
    3. Missing keyword signals
    4. All retrieved docs used (no filtering)
    """
    print_section("FAILURE MODE: Retrieval Without Reranking", Fore.RED)
    
    documents = create_diverse_documents()
    
    print(f"{Fore.CYAN}Document Collection:{Style.RESET_ALL}")
    print(f"Total documents: {len(documents)}\n")
    
    for doc in documents:
        relevance = doc.metadata.get('relevance', 'unknown')
        doc_id = doc.metadata.get('doc_id', 'unknown')
        category = doc.metadata.get('category', 'unknown')
        
        rel_color = {
            'high': Fore.GREEN,
            'medium': Fore.YELLOW,
            'low': Fore.RED,
            'very_low': Fore.RED
        }.get(relevance, Fore.WHITE)
        
        print(f"  {doc_id:15s} | {category:15s} | {rel_color}{relevance:10s}{Style.RESET_ALL}")
    
    query = "How do I implement API authentication?"
    print(f"\n{Fore.CYAN}User Query:{Style.RESET_ALL} {query}\n")
    
    if not os.getenv('OPENAI_API_KEY'):
        print(f"{Fore.YELLOW}⚠️  OPENAI_API_KEY not set. Showing conceptual demo.{Style.RESET_ALL}\n")
        
        print(f"{Fore.RED}❌ PROBLEM 1: Vector similarity alone is insufficient{Style.RESET_ALL}")
        print("Vector embeddings capture semantic similarity, not relevance:")
        print("  Query: 'How do I implement API authentication?'")
        print("\n  Vector similarity scores:")
        print(f"    {Fore.RED}0.87{Style.RESET_ALL} - 'User Interface Design for Login Screens' ❌")
        print(f"      (High similarity: 'login', 'password', 'authentication')")
        print(f"      (But WRONG: UI design, not API implementation)")
        print(f"    {Fore.RED}0.85{Style.RESET_ALL} - 'Authentication in Video Streaming' ❌")
        print(f"      (High similarity: 'authentication', 'token')")
        print(f"      (But WRONG: video streaming, not APIs)")
        print(f"    {Fore.GREEN}0.82{Style.RESET_ALL} - 'API Authentication Methods' ✓")
        print(f"      (Slightly lower similarity, but ACTUALLY RELEVANT)")
        print("\n  Problem: Most similar ≠ Most relevant for answering\n")
        
        print(f"{Fore.RED}❌ PROBLEM 2: No query-document interaction{Style.RESET_ALL}")
        print("Bi-encoder (vector similarity):")
        print("  - Encodes query and docs independently")
        print("  - No interaction between query and document")
        print("  - Misses subtle relevance signals")
        print("\nExample:")
        print("  Query: 'best practices for authentication'")
        print("  Doc A: 'Authentication Best Practices' (title match)")
        print("  Doc B: 'API Security Guidelines' (contains best practices)")
        print("  ")
        print("  Bi-encoder may score them similarly")
        print("  But Doc A directly answers, Doc B is tangential\n")
        
        print(f"{Fore.RED}❌ PROBLEM 3: Missing keyword signals{Style.RESET_ALL}")
        print("Pure vector search ignores exact matches:")
        print("  Query: 'JWT authentication'")
        print("  Doc A: 'Use JWT tokens for stateless auth' (exact match)")
        print("  Doc B: 'Token-based authentication methods' (semantic)")
        print("  ")
        print("  Vector similarity might favor Doc B")
        print("  But Doc A has exact term match → likely more relevant\n")
        
        print(f"{Fore.RED}❌ PROBLEM 4: Using all top-k docs{Style.RESET_ALL}")
        print("Naive approach: retrieve top-5, use all of them")
        print("  ")
        print("  Retrieved docs:")
        print("    1. API Authentication (relevance: 0.95) ✓")
        print("    2. OAuth 2.0 Guide (relevance: 0.92) ✓")
        print("    3. UI Login Design (relevance: 0.30) ❌")
        print("    4. Testing Auth (relevance: 0.25) ❌")
        print("    5. Video DRM (relevance: 0.15) ❌")
        print("  ")
        print("  Problem: LLM gets 3 irrelevant docs in context")
        print("  Result: Confused, hallucinated, or wrong answers\n")
        
        print(f"{Fore.RED}Impact on Answer Quality:{Style.RESET_ALL}")
        print("  - 30-50% of retrieved docs are irrelevant")
        print("  - LLM wastes tokens on noise")
        print("  - Answers less accurate")
        print("  - May cite wrong sources")
        print("  - User trust eroded\n")
        
        return
    
    # If API key available, demonstrate actual retrieval
    embeddings = OpenAIEmbeddings()
    
    print(f"{Fore.YELLOW}Creating vector store...{Style.RESET_ALL}")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    print(f"\n{Fore.RED}Retrieval WITHOUT reranking:{Style.RESET_ALL}\n")
    
    results = vectorstore.similarity_search_with_score(query, k=5)
    
    print(f"Top-5 results (by vector similarity only):")
    for i, (doc, score) in enumerate(results, 1):
        doc_id = doc.metadata.get('doc_id', 'unknown')
        category = doc.metadata.get('category', 'unknown')
        relevance = doc.metadata.get('relevance', 'unknown')
        
        rel_icon = "✓" if relevance == 'high' else ("~" if relevance == 'medium' else "✗")
        rel_color = Fore.GREEN if relevance == 'high' else (Fore.YELLOW if relevance == 'medium' else Fore.RED)
        
        print(f"  {i}. {rel_icon} {doc_id:15s} | Similarity: {score:.3f} | {rel_color}{relevance:10s}{Style.RESET_ALL}")
    
    print(f"\n{Fore.RED}Problem: Irrelevant docs ranked highly due to keyword overlap!{Style.RESET_ALL}")


def run_fixed():
    """
    Demonstrates FIXED retrieval with reranking.
    
    Solutions:
    1. Two-stage retrieval (retrieve more, rerank to fewer)
    2. Cross-encoder reranking
    3. Hybrid search (vector + keyword)
    4. Relevance threshold filtering
    5. Diversity-aware selection
    """
    print_section("FIXED VERSION: Retrieval with Reranking", Fore.GREEN)
    
    documents = create_diverse_documents()
    query = "How do I implement API authentication?"
    
    print(f"{Fore.CYAN}User Query:{Style.RESET_ALL} {query}\n")
    
    print(f"{Fore.GREEN}✅ SOLUTION 1: Two-stage retrieval{Style.RESET_ALL}")
    print("Stage 1: Fast retrieval (bi-encoder)")
    print("  - Retrieve top-20 candidates")
    print("  - Fast: O(log n) with ANN")
    print("  - Casts wide net")
    print("\nStage 2: Reranking (cross-encoder)")
    print("  - Rerank 20 → top-5")
    print("  - Slower but accurate")
    print("  - Models query-document interaction\n")
    
    print(f"{Fore.GREEN}✅ SOLUTION 2: Cross-encoder reranking{Style.RESET_ALL}")
    print("Cross-encoder architecture:")
    print("  Input: [CLS] query [SEP] document [SEP]")
    print("  - Joint encoding of query + document")
    print("  - Attention between query and document tokens")
    print("  - Output: single relevance score")
    print("\n  Popular models:")
    print("  - cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("  - cross-encoder/ms-marco-TinyBERT-L-2")
    print("  - Fine-tuned on MS MARCO passage ranking\n")
    
    print(f"{Fore.GREEN}✅ SOLUTION 3: Hybrid search (vector + keyword){Style.RESET_ALL}")
    print("Combine signals:")
    print("  final_score = 0.7 * vector_score + 0.3 * bm25_score")
    print("\n  Benefits:")
    print("  - Vector: semantic similarity")
    print("  - BM25: exact keyword matches")
    print("  - Hybrid: best of both worlds\n")
    
    print(f"{Fore.GREEN}✅ SOLUTION 4: Relevance threshold{Style.RESET_ALL}")
    print("Don't use all top-k:")
    print("  retrieved = get_top_20(query)")
    print("  reranked = rerank(query, retrieved)")
    print("  filtered = [d for d in reranked if d.score > 0.5]")
    print("  ")
    print("  Result: Only use high-quality docs")
    print("  Typical: 3-5 docs instead of 10\n")
    
    print(f"{Fore.GREEN}✅ SOLUTION 5: Diversity-aware selection{Style.RESET_ALL}")
    print("Avoid redundant documents:")
    print("  - Maximal Marginal Relevance (MMR)")
    print("  - Select diverse docs covering different aspects")
    print("  - Prevents over-representation of one topic\n")
    
    # Demonstrate reranking
    if not os.getenv('OPENAI_API_KEY'):
        print(f"{Fore.YELLOW}⚠️  OPENAI_API_KEY not set. Showing simulated reranking.{Style.RESET_ALL}\n")
        
        print(f"{Fore.CYAN}Simulated reranking process:{Style.RESET_ALL}\n")
        
        print("Stage 1: Vector similarity (top-8 candidates)")
        candidates = [
            ('auth_001', 0.85, 'high'),
            ('auth_002', 0.83, 'high'),
            ('ui_001', 0.81, 'low'),         # Deceptive: keyword overlap
            ('ratelimit_001', 0.79, 'medium'),
            ('stream_001', 0.77, 'very_low'),  # Deceptive: keyword overlap
            ('docs_001', 0.75, 'medium'),
            ('test_001', 0.73, 'low'),
            ('db_001', 0.70, 'low'),
        ]
        
        for doc_id, sim, rel in candidates:
            rel_color = Fore.GREEN if rel == 'high' else (Fore.YELLOW if rel == 'medium' else Fore.RED)
            print(f"  {doc_id:15s} | Sim: {sim:.2f} | {rel_color}{rel:10s}{Style.RESET_ALL}")
        
        print("\nStage 2: Cross-encoder reranking (simulated)")
        
        # Simulate cross-encoder giving better scores to truly relevant docs
        reranked = [
            ('auth_001', 0.95, 'high'),      # Boosted
            ('auth_002', 0.92, 'high'),      # Boosted
            ('ratelimit_001', 0.65, 'medium'), # Moderate
            ('docs_001', 0.58, 'medium'),    # Moderate
            ('ui_001', 0.35, 'low'),         # Demoted (was 0.81!)
            ('stream_001', 0.22, 'very_low'), # Demoted (was 0.77!)
            ('test_001', 0.28, 'low'),
            ('db_001', 0.25, 'low'),
        ]
        
        print("\nReranked (by relevance):")
        for doc_id, score, rel in reranked[:5]:
            rel_icon = "✓" if rel in ['high', 'medium'] else "✗"
            rel_color = Fore.GREEN if rel == 'high' else (Fore.YELLOW if rel == 'medium' else Fore.RED)
            print(f"  {rel_icon} {doc_id:15s} | Score: {score:.2f} | {rel_color}{rel:10s}{Style.RESET_ALL}")
        
        print("\nStage 3: Threshold filtering (score > 0.5)")
        filtered = [(d, s, r) for d, s, r in reranked if s > 0.5]
        
        print(f"\nFinal selection ({len(filtered)} docs):")
        for doc_id, score, rel in filtered:
            print(f"  ✓ {doc_id:15s} | Score: {score:.2f} | {Fore.GREEN}{rel:10s}{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}Impact:{Style.RESET_ALL}")
        print("  Before reranking: 2/5 relevant (40%)")
        print("  After reranking:  4/4 relevant (100%)")
        print("  Quality improvement: 2.5x")
        
        return
    
    # If API key available, do actual demonstration
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    print(f"{Fore.CYAN}Stage 1: Vector retrieval (top-8 candidates){Style.RESET_ALL}\n")
    
    candidates = vectorstore.similarity_search_with_score(query, k=8)
    
    print("Retrieved candidates:")
    for i, (doc, score) in enumerate(candidates, 1):
        doc_id = doc.metadata.get('doc_id')
        relevance = doc.metadata.get('relevance')
        print(f"  {i}. {doc_id:15s} | Sim: {score:.3f} | Relevance: {relevance}")
    
    print(f"\n{Fore.CYAN}Stage 2: Simulated cross-encoder reranking{Style.RESET_ALL}\n")
    
    # Simulate reranking (in production, use actual cross-encoder)
    reranked = []
    for doc, vec_score in candidates:
        # Simulate cross-encoder score
        cross_score = calculate_cross_encoder_score(query, doc.page_content)
        
        # Combine scores
        final_score = 0.5 * (1 - vec_score) + 0.5 * cross_score
        
        reranked.append((doc, final_score))
    
    # Sort by reranked score
    reranked.sort(key=lambda x: x[1], reverse=True)
    
    print("Reranked results:")
    for i, (doc, score) in enumerate(reranked[:5], 1):
        doc_id = doc.metadata.get('doc_id')
        relevance = doc.metadata.get('relevance')
        rel_icon = "✓" if relevance in ['high', 'medium'] else "✗"
        rel_color = Fore.GREEN if relevance == 'high' else (Fore.YELLOW if relevance == 'medium' else Fore.RED)
        print(f"  {i}. {rel_icon} {doc_id:15s} | Score: {score:.3f} | {rel_color}{relevance}{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}{'=' * 80}")
    print("KEY LESSONS:")
    print("=" * 80)
    print("1. ✅ Use two-stage retrieval: retrieve many, rerank to few")
    print("2. ✅ Cross-encoder reranking models query-document interaction")
    print("3. ✅ Hybrid search (vector + BM25) captures multiple signals")
    print("4. ✅ Apply relevance threshold - don't use low-scoring docs")
    print("5. ✅ Consider diversity to avoid redundant results")
    print("6. ✅ Reranking improves answer quality by 2-3x")
    print(f"{'=' * 80}{Style.RESET_ALL}\n")


def main():
    """Run both failure and fixed examples."""
    print(f"\n{Fore.MAGENTA}{'=' * 80}")
    print("RAG FAILURE MODE #5: Retrieval Without Reranking")
    print("=" * 80)
    print("\nThis example demonstrates how relying solely on vector similarity")
    print("leads to poor document selection and degraded answer quality.")
    print(f"{'=' * 80}{Style.RESET_ALL}\n")
    
    # Run failure case
    run_failure()
    
    input(f"\n{Fore.CYAN}Press Enter to see the fixed version...{Style.RESET_ALL}")
    
    # Run fixed case
    run_fixed()
    
    print(f"\n{Fore.MAGENTA}{'=' * 80}")
    print("Comparison: Failure vs Fixed")
    print("=" * 80)
    print("\nFAILURE:")
    print("  ❌ Single-stage retrieval")
    print("  ❌ Vector similarity only")
    print("  ❌ No query-document interaction")
    print("  ❌ Uses all top-k docs")
    print("  ❌ 40% relevance rate")
    print("\nFIXED:")
    print("  ✅ Two-stage retrieval")
    print("  ✅ Cross-encoder reranking")
    print("  ✅ Hybrid search (vector + keyword)")
    print("  ✅ Relevance threshold filtering")
    print("  ✅ 90%+ relevance rate")
    print(f"{'=' * 80}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
