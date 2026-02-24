"""
RAG Failure Mode #4: Long-Tail Latency Spikes
==============================================

Problem: Unpredictable latency spikes cause timeouts, poor user experience,
and system instability in production RAG systems.

Common Issues:
- P50 latency: 200ms (good) → P99 latency: 8000ms (disaster)
- Vector similarity search complexity: O(n*d) for large indexes
- Cold start problems with embedding models
- No caching strategy
- Synchronous processing blocks on slow operations
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Dict, Any
import os
import time
import random
from collections import deque
from colorama import Fore, Style, init
import statistics

# Initialize colorama
init(autoreset=True)


def print_section(title: str, color=Fore.CYAN):
    """Print a formatted section header."""
    print(f"\n{color}{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}{Style.RESET_ALL}\n")


def create_large_document_set(num_docs: int = 1000) -> List[Document]:
    """Create a large set of documents for latency testing."""
    documents = []
    
    categories = ['Technology', 'Health', 'Finance', 'Education', 'Travel']
    topics = ['AI', 'Blockchain', 'Cloud', 'Security', 'Analytics']
    
    for i in range(num_docs):
        category = random.choice(categories)
        topic = random.choice(topics)
        
        content = f"""
        Document {i+1}: {topic} in {category}
        
        This is a detailed article about {topic} in the {category} sector.
        We cover various aspects including implementation, best practices,
        challenges, and future trends. The content is comprehensive and
        aims to provide value to readers interested in {topic}.
        
        Key points:
        - Understanding {topic} fundamentals
        - Practical applications in {category}
        - Case studies and real-world examples
        - Future outlook and predictions
        
        This document contains approximately 200 tokens to simulate realistic
        content length that would be encountered in production systems.
        """
        
        doc = Document(
            page_content=content,
            metadata={
                'doc_id': f'doc_{i+1:04d}',
                'category': category,
                'topic': topic,
                'size_tokens': 200,
            }
        )
        documents.append(doc)
    
    return documents


def simulate_latency_spike(base_latency_ms: float = 100) -> float:
    """Simulate realistic latency with occasional spikes."""
    # 95% of requests: normal latency
    # 4% of requests: 2-3x slower
    # 1% of requests: 10-40x slower (long tail!)
    
    rand = random.random()
    
    if rand < 0.95:
        # Normal case: base latency with small variance
        return base_latency_ms * random.uniform(0.8, 1.2)
    elif rand < 0.99:
        # Moderate slowdown: 2-3x
        return base_latency_ms * random.uniform(2.0, 3.0)
    else:
        # Long-tail spike: 10-40x!
        return base_latency_ms * random.uniform(10.0, 40.0)


def measure_latency_distribution(operation_fn, num_trials: int = 100) -> Dict[str, float]:
    """Measure latency distribution including P50, P95, P99."""
    latencies = []
    
    for _ in range(num_trials):
        start = time.time()
        operation_fn()
        latency = (time.time() - start) * 1000  # Convert to ms
        latencies.append(latency)
    
    latencies.sort()
    
    return {
        'min': latencies[0],
        'p50': latencies[int(len(latencies) * 0.50)],
        'p90': latencies[int(len(latencies) * 0.90)],
        'p95': latencies[int(len(latencies) * 0.95)],
        'p99': latencies[int(len(latencies) * 0.99)],
        'max': latencies[-1],
        'mean': statistics.mean(latencies),
        'stddev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
    }


def print_latency_stats(stats: Dict[str, float], title: str):
    """Print latency statistics in a formatted way."""
    print(f"\n{Fore.CYAN}{title}{Style.RESET_ALL}")
    print(f"  Min:    {stats['min']:7.1f} ms")
    print(f"  P50:    {stats['p50']:7.1f} ms")
    print(f"  P90:    {stats['p90']:7.1f} ms")
    print(f"  P95:    {stats['p95']:7.1f} ms")
    
    # Color code P99 based on value
    p99 = stats['p99']
    if p99 < 500:
        p99_color = Fore.GREEN
    elif p99 < 2000:
        p99_color = Fore.YELLOW
    else:
        p99_color = Fore.RED
    
    print(f"  P99:    {p99_color}{p99:7.1f} ms{Style.RESET_ALL}")
    print(f"  Max:    {stats['max']:7.1f} ms")
    print(f"  Mean:   {stats['mean']:7.1f} ms ± {stats['stddev']:.1f}")


def run_failure():
    """
    Demonstrates LATENCY SPIKE problems.
    
    Problems:
    1. No query caching
    2. Large index without optimization
    3. Synchronous processing
    4. No timeout handling
    5. Cold start delays
    """
    print_section("FAILURE MODE: Long-Tail Latency Spikes", Fore.RED)
    
    print(f"{Fore.CYAN}Scenario: Production RAG system under load{Style.RESET_ALL}\n")
    
    # Simulate latency distribution
    print(f"{Fore.YELLOW}Simulating latency distribution...{Style.RESET_ALL}\n")
    
    print(f"{Fore.RED}❌ PROBLEM 1: No caching → repeated slow lookups{Style.RESET_ALL}")
    print("Same query executed multiple times:")
    
    latencies_no_cache = []
    for i in range(10):
        latency = simulate_latency_spike(base_latency_ms=150)
        latencies_no_cache.append(latency)
        status = "✓" if latency < 500 else "✗"
        color = Fore.GREEN if latency < 500 else Fore.RED
        print(f"  Request {i+1}: {color}{latency:6.1f} ms {status}{Style.RESET_ALL}")
    
    print(f"\n{Fore.RED}❌ PROBLEM 2: Large index → O(n) search complexity{Style.RESET_ALL}")
    print("Index size impact on latency:")
    
    for size in [100, 1000, 10000, 100000]:
        # Simulate O(n) complexity
        base = 50
        complexity_factor = size / 1000
        latency = base * complexity_factor
        
        color = Fore.GREEN if latency < 200 else (Fore.YELLOW if latency < 1000 else Fore.RED)
        print(f"  {size:6d} docs → {color}{latency:7.1f} ms{Style.RESET_ALL}")
    
    print(f"\n{Fore.RED}❌ PROBLEM 3: Synchronous processing → blocking{Style.RESET_ALL}")
    print("Request timeline (synchronous):")
    print("  [Embed query: 100ms] → [Search: 150ms] → [Rerank: 200ms] → [LLM: 800ms]")
    print(f"  Total: {Fore.RED}1,250 ms{Style.RESET_ALL} (blocking entire time)")
    
    print(f"\n{Fore.RED}❌ PROBLEM 4: No timeout handling{Style.RESET_ALL}")
    print("When slow requests occur:")
    print("  - Request 1: 150 ms ✓")
    print("  - Request 2: 180 ms ✓")
    print(f"  - Request 3: {Fore.RED}8,500 ms ✗{Style.RESET_ALL} (long-tail spike!)")
    print("  - No timeout → user waits forever")
    print("  - Resources locked up")
    print("  - Other requests delayed")
    
    print(f"\n{Fore.RED}❌ PROBLEM 5: Cold start delays{Style.RESET_ALL}")
    print("First request after deployment:")
    print("  - Load embedding model: 2,000 ms")
    print("  - Load vector index: 1,500 ms")
    print("  - Initialize LLM: 1,000 ms")
    print(f"  - First query: {Fore.RED}4,500 ms{Style.RESET_ALL} (terrible UX!)")
    
    # Latency distribution analysis
    print(f"\n{Fore.RED}❌ PROBLEM 6: Long-tail distribution{Style.RESET_ALL}")
    print("Without optimization:")
    
    # Simulate many requests
    all_latencies = [simulate_latency_spike(200) for _ in range(1000)]
    all_latencies.sort()
    
    p50 = all_latencies[int(len(all_latencies) * 0.50)]
    p99 = all_latencies[int(len(all_latencies) * 0.99)]
    
    print(f"  P50 (median): {p50:6.1f} ms {Fore.GREEN}✓ Looks good!{Style.RESET_ALL}")
    print(f"  P99:          {Fore.RED}{p99:6.1f} ms ✗ Terrible!{Style.RESET_ALL}")
    print(f"  Ratio:        {p99/p50:6.1f}x slower")
    print("\n  Impact: 1% of users experience 10-30x slower responses!")
    print("  SLA violation: Cannot guarantee < 2s latency")
    
    if not os.getenv('OPENAI_API_KEY'):
        print(f"\n{Fore.YELLOW}⚠️  OPENAI_API_KEY not set. Showing conceptual demo.{Style.RESET_ALL}")
        print("\nFor full latency measurement with actual embeddings, set OPENAI_API_KEY.\n")
        return
    
    # If API key available, do actual latency testing
    print(f"\n{Fore.YELLOW}Running actual latency test...{Style.RESET_ALL}")
    
    # Create moderate-sized document set
    documents = create_large_document_set(num_docs=100)
    embeddings = OpenAIEmbeddings()
    
    print(f"Creating vector store with {len(documents)} documents...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Measure without optimization
    def search_operation():
        query = f"Tell me about AI in Technology"
        vectorstore.similarity_search(query, k=5)
    
    print(f"\n{Fore.RED}Measuring latency WITHOUT optimizations:{Style.RESET_ALL}")
    stats = measure_latency_distribution(search_operation, num_trials=20)
    print_latency_stats(stats, "Unoptimized Search Latency")


def run_fixed():
    """
    Demonstrates FIXED latency management.
    
    Solutions:
    1. Query result caching
    2. Approximate nearest neighbor (ANN) search
    3. Async processing
    4. Timeout handling
    5. Warm-up strategies
    6. Request hedging
    """
    print_section("FIXED VERSION: Latency Optimization Strategies", Fore.GREEN)
    
    print(f"{Fore.GREEN}✅ SOLUTION 1: Query result caching{Style.RESET_ALL}")
    print("Cache identical queries:")
    
    class QueryCache:
        def __init__(self, max_size: int = 1000):
            self.cache = {}
            self.access_order = deque(maxlen=max_size)
        
        def get(self, query: str):
            if query in self.cache:
                return self.cache[query]
            return None
        
        def set(self, query: str, result):
            self.cache[query] = result
            self.access_order.append(query)
    
    cache = QueryCache()
    query = "What is machine learning?"
    
    # First request: Cache miss
    print(f"  Request 1: Cache MISS → {Fore.YELLOW}200 ms{Style.RESET_ALL} (actual search)")
    cache.set(query, ["doc1", "doc2", "doc3"])
    
    # Subsequent requests: Cache hit
    for i in range(2, 6):
        print(f"  Request {i}: Cache HIT  → {Fore.GREEN}2 ms{Style.RESET_ALL} (100x faster!)")
    
    print("\n  Impact: 100x faster for repeated queries")
    print("  Cache hit rate: 60-80% typical in production")
    
    print(f"\n{Fore.GREEN}✅ SOLUTION 2: Approximate Nearest Neighbor (ANN){Style.RESET_ALL}")
    print("Replace brute-force search with ANN algorithms:")
    print("  - FAISS (Facebook): HNSW, IVF indices")
    print("  - Annoy (Spotify): Tree-based")
    print("  - ScaNN (Google): Quantization-based")
    print("\nComplexity comparison:")
    print("  Brute-force: O(n*d) → 5,000 ms for 100k docs")
    print(f"  HNSW index:  O(log n) → {Fore.GREEN}50 ms{Style.RESET_ALL} for 100k docs (100x faster!)")
    
    print(f"\n{Fore.GREEN}✅ SOLUTION 3: Async processing pipeline{Style.RESET_ALL}")
    print("Parallelize independent operations:")
    print("\n  BEFORE (synchronous):")
    print("  [Embed: 100ms] → [Search: 150ms] → [Rerank: 200ms] → Total: 450ms")
    print("\n  AFTER (async):")
    print("  ┌─ [Embed: 100ms] ─┐")
    print("  ├─ [Search: 150ms] ┼─→ Total: 150ms (3x faster!)")
    print("  └─ [Rerank: 200ms] ┘")
    
    print(f"\n{Fore.GREEN}✅ SOLUTION 4: Timeout & circuit breaker{Style.RESET_ALL}")
    print("Prevent cascading failures:")
    print("""
    def search_with_timeout(query, timeout_ms=2000):
        try:
            result = vectorstore.search(query, timeout=timeout_ms)
            return result
        except TimeoutError:
            # Fallback: return cached results or error
            logger.warning(f'Search timeout: {query}')
            return cached_fallback(query)
    """)
    print("  Impact: Bounded latency, no infinite waits")
    
    print(f"\n{Fore.GREEN}✅ SOLUTION 5: Model warming{Style.RESET_ALL}")
    print("Pre-load models at startup:")
    print("  1. Load embedding model during deployment")
    print("  2. Run dummy queries to warm caches")
    print("  3. Pre-load vector index into memory")
    print("  4. Initialize LLM connections")
    print("\n  First query latency:")
    print(f"    Without warming: {Fore.RED}4,500 ms{Style.RESET_ALL}")
    print(f"    With warming:    {Fore.GREEN}180 ms{Style.RESET_ALL} (25x improvement!)")
    
    print(f"\n{Fore.GREEN}✅ SOLUTION 6: Request hedging{Style.RESET_ALL}")
    print("Send duplicate requests to combat tail latency:")
    print("  1. Send primary request to Server A")
    print("  2. After 500ms, send hedge request to Server B")
    print("  3. Use whichever responds first")
    print("  4. Cancel the other")
    print("\n  Impact on P99 latency:")
    print(f"    Without hedging: {Fore.RED}3,500 ms{Style.RESET_ALL}")
    print(f"    With hedging:    {Fore.GREEN}650 ms{Style.RESET_ALL} (5x better!)")
    
    print(f"\n{Fore.GREEN}✅ SOLUTION 7: Batch processing{Style.RESET_ALL}")
    print("Group queries for efficiency:")
    print("  10 sequential queries: 10 × 150ms = 1,500ms")
    print("  10 batched queries:   1 × 200ms = 200ms (7.5x faster!)")
    
    print(f"\n{Fore.GREEN}✅ SOLUTION 8: Index sharding{Style.RESET_ALL}")
    print("Distribute index across machines:")
    print("  Single machine:  100k docs → 500ms latency")
    print("  10 shards:       10k docs each → 80ms latency (6x faster!)")
    
    # Demonstrate caching benefit
    print(f"\n{Fore.CYAN}Demonstration: Cache impact{Style.RESET_ALL}")
    
    queries = [
        "machine learning",
        "deep learning",
        "machine learning",  # Repeat
        "neural networks",
        "machine learning",  # Repeat
        "deep learning",     # Repeat
    ]
    
    cached_results = QueryCache()
    
    print("\nQuery execution with caching:")
    total_no_cache = 0
    total_with_cache = 0
    
    for i, query in enumerate(queries, 1):
        # Simulate latency
        base_latency = 150
        
        if cached_results.get(query):
            cache_latency = 2
            status = f"{Fore.GREEN}HIT{Style.RESET_ALL} "
        else:
            cache_latency = base_latency
            cached_results.set(query, f"results_{i}")
            status = f"{Fore.YELLOW}MISS{Style.RESET_ALL}"
        
        total_no_cache += base_latency
        total_with_cache += cache_latency
        
        print(f"  {i}. '{query[:20]}' → {status} ({cache_latency:3.0f} ms)")
    
    print(f"\n  Total without cache: {Fore.RED}{total_no_cache} ms{Style.RESET_ALL}")
    print(f"  Total with cache:    {Fore.GREEN}{total_with_cache} ms{Style.RESET_ALL}")
    print(f"  Speedup:             {Fore.GREEN}{total_no_cache/total_with_cache:.1f}x{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}{'=' * 80}")
    print("KEY LESSONS:")
    print("=" * 80)
    print("1. ✅ Cache query results (60-80% hit rate typical)")
    print("2. ✅ Use ANN algorithms (HNSW, IVF) instead of brute-force")
    print("3. ✅ Implement timeouts and circuit breakers")
    print("4. ✅ Warm up models at startup")
    print("5. ✅ Use async processing for independent operations")
    print("6. ✅ Consider request hedging for critical P99 latency")
    print("7. ✅ Monitor P99, not just P50 - tail latency matters!")
    print(f"{'=' * 80}{Style.RESET_ALL}\n")


def main():
    """Run both failure and fixed examples."""
    print(f"\n{Fore.MAGENTA}{'=' * 80}")
    print("RAG FAILURE MODE #4: Long-Tail Latency Spikes")
    print("=" * 80)
    print("\nThis example demonstrates how unpredictable latency spikes cause")
    print("poor user experience and SLA violations in production systems.")
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
    print("  ❌ No caching")
    print("  ❌ Brute-force search O(n)")
    print("  ❌ Synchronous processing")
    print("  ❌ No timeouts")
    print("  ❌ P99 latency: 3,500+ ms")
    print("\nFIXED:")
    print("  ✅ Query caching (100x faster)")
    print("  ✅ ANN search O(log n)")
    print("  ✅ Async pipeline")
    print("  ✅ Timeout handling")
    print("  ✅ P99 latency: < 500 ms")
    print(f"{'=' * 80}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
