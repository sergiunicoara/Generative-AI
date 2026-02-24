"""
RAG Failure Mode #5: Context Window Overflow
=============================================

Problem: Retrieving too much context that exceeds the LLM's token limit, causing
truncation, failures, or degraded performance. Even with large context windows,
stuffing too many documents reduces answer quality.

Common Issues:
- Exceeding model's maximum token limit (crashes or truncation)
- Retrieved documents get silently truncated
- Important information lost due to position bias
- High cost from using maximum context unnecessarily
- Slow inference from processing huge contexts
"""

from typing import List, Dict, Tuple
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_all_documents() -> Dict[str, str]:
    """Load all sample documents for context window testing."""
    docs = {}
    
    files = [
        'technical_docs.txt',
        'legal_document.txt',
        'medical_records.txt',
        'product_reviews.txt'
    ]
    
    for filename in files:
        try:
            with open(f'sample_data/{filename}', 'r') as f:
                docs[filename] = f.read()
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
    
    return docs


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4


def create_mock_retrieval(query: str, documents: Dict[str, str], 
                          top_k: int = 10) -> List[Document]:
    """
    Simulate retrieval that returns many documents.
    In real scenario, this would use vector similarity search.
    """
    
    # Split all documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    all_chunks = []
    for filename, content in documents.items():
        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            all_chunks.append(Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "chunk_id": i,
                    "tokens": count_tokens(chunk)
                }
            ))
    
    # Return top_k chunks (simulating retrieval)
    return all_chunks[:top_k]


def run_failure():
    """
    Demonstrates CONTEXT WINDOW OVERFLOW problem.
    
    Problem: Naive RAG retrieves too many documents, exceeding token limits
    and degrading performance.
    """
    print("=" * 80)
    print("FAILURE MODE: Context Window Overflow")
    print("=" * 80)
    
    documents = load_all_documents()
    
    print(f"\n📚 Knowledge Base: {len(documents)} documents")
    
    total_tokens = sum(count_tokens(doc) for doc in documents.values())
    print(f"   Total tokens in knowledge base: {total_tokens:,}")
    
    # Model limits
    MODEL_LIMITS = {
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
    }
    
    print("\n📊 Common Model Context Limits:")
    for model, limit in MODEL_LIMITS.items():
        print(f"   {model}: {limit:,} tokens")
    
    # Simulate naive retrieval
    print("\n" + "=" * 80)
    print("❌ NAIVE APPROACH: Retrieve Everything")
    print("=" * 80)
    
    query = "What are the authentication methods and patient medications?"
    
    # Retrieve way too many documents
    top_k_excessive = 50
    retrieved_docs = create_mock_retrieval(query, documents, top_k=top_k_excessive)
    
    print(f"\n🔍 Query: '{query}'")
    print(f"   Retrieved: {len(retrieved_docs)} documents")
    
    # Calculate total tokens
    context_tokens = sum(doc.metadata['tokens'] for doc in retrieved_docs)
    query_tokens = count_tokens(query)
    
    # Typical prompt structure tokens
    system_prompt_tokens = 150
    output_tokens_reserved = 1000  # Reserve space for response
    
    total_tokens_needed = (
        system_prompt_tokens + 
        query_tokens + 
        context_tokens + 
        output_tokens_reserved
    )
    
    print(f"\n📊 Token Breakdown:")
    print(f"   System prompt: ~{system_prompt_tokens} tokens")
    print(f"   User query: {query_tokens} tokens")
    print(f"   Retrieved context: {context_tokens:,} tokens")
    print(f"   Reserved for output: {output_tokens_reserved} tokens")
    print(f"   " + "=" * 40)
    print(f"   TOTAL NEEDED: {total_tokens_needed:,} tokens")
    
    # Check against model limits
    model = "gpt-3.5-turbo"
    model_limit = MODEL_LIMITS[model]
    
    print(f"\n🎯 Target Model: {model}")
    print(f"   Context limit: {model_limit:,} tokens")
    print(f"   Tokens needed: {total_tokens_needed:,} tokens")
    
    if total_tokens_needed > model_limit:
        overflow = total_tokens_needed - model_limit
        print(f"   ❌ OVERFLOW: {overflow:,} tokens over limit!")
        print(f"   💥 This will cause:")
        print(f"      - Context truncation (losing information)")
        print(f"      - API errors (request rejected)")
        print(f"      - Incorrect answers (missing critical data)")
    
    # Show what gets truncated
    print("\n⚠️  What Happens with Truncation:")
    print("   1. Oldest/last documents get dropped silently")
    print("   2. Important information may be lost")
    print("   3. LLM doesn't know what was truncated")
    print("   4. Answers become unreliable")
    print("   5. No warning to user about missing context")
    
    # Demonstrate position bias
    print("\n" + "=" * 80)
    print("❌ PROBLEM: Lost in the Middle")
    print("=" * 80)
    
    print("""
📉 Position Bias Problem:

Even if context fits, LLMs struggle with information in the middle of long contexts.
Research shows:
- Information at the START: Retrieved well ✓
- Information in the MIDDLE: Often missed ✗
- Information at the END: Retrieved well ✓

With 50 retrieved documents, critical information in documents 20-35 
is likely to be ignored, even if the LLM can "see" it.
""")
    
    print("\n⚠️  Additional Problems:")
    print("   1. High API costs (paying for unnecessary tokens)")
    print("   2. Slow inference (processing huge contexts)")
    print("   3. Degraded quality (noise drowns out signal)")
    print("   4. Poor user experience (slow responses)")
    print("   5. Wasted context window (irrelevant docs)")


def run_fixed():
    """
    Demonstrates SMART CONTEXT MANAGEMENT.
    
    Solution: Selective retrieval, reranking, token budgeting, and 
    context compression techniques.
    """
    print("\n" + "=" * 80)
    print("FIXED: Smart Context Management")
    print("=" * 80)
    
    documents = load_all_documents()
    
    # Model configuration
    model = "gpt-3.5-turbo"
    model_limit = 4096
    
    # Token budget allocation
    TOKEN_BUDGET = {
        "system_prompt": 200,
        "query": 100,
        "context": 2500,  # Most of the budget
        "output": 1000,
        "safety_margin": 296  # Buffer
    }
    
    total_budget = sum(TOKEN_BUDGET.values())
    
    print(f"\n✅ SMART APPROACH: Token Budget Management")
    print(f"   Model: {model}")
    print(f"   Context limit: {model_limit:,} tokens")
    print(f"   Allocated budget: {total_budget:,} tokens")
    
    print(f"\n📊 Token Budget Allocation:")
    for component, tokens in TOKEN_BUDGET.items():
        percentage = (tokens / total_budget) * 100
        print(f"   {component:20s}: {tokens:4d} tokens ({percentage:5.1f}%)")
    
    # Strategy 1: Limit retrieved documents
    print("\n" + "=" * 80)
    print("STRATEGY 1: Selective Retrieval")
    print("=" * 80)
    
    query = "What are the authentication methods and patient medications?"
    
    # Smart retrieval with token budget
    context_budget = TOKEN_BUDGET["context"]
    
    print(f"\n🎯 Context Budget: {context_budget} tokens")
    
    # Retrieve fewer, more relevant documents
    top_k_smart = 8
    retrieved_docs = create_mock_retrieval(query, documents, top_k=top_k_smart)
    
    print(f"   Initial retrieval: {top_k_smart} documents")
    
    # Select documents that fit budget
    selected_docs = []
    current_tokens = 0
    
    for doc in retrieved_docs:
        doc_tokens = doc.metadata['tokens']
        if current_tokens + doc_tokens <= context_budget:
            selected_docs.append(doc)
            current_tokens += doc_tokens
        else:
            break
    
    print(f"   Selected: {len(selected_docs)} documents")
    print(f"   Total context tokens: {current_tokens}")
    print(f"   Remaining budget: {context_budget - current_tokens}")
    print(f"   ✓ Fits within budget!")
    
    # Strategy 2: Reranking
    print("\n" + "=" * 80)
    print("STRATEGY 2: Reranking for Relevance")
    print("=" * 80)
    
    print("""
✅ Reranking Strategy:

1. Initial retrieval: Get top 20 candidates
2. Rerank: Score documents by relevance to query
3. Select: Take top 5-8 most relevant
4. Verify: Ensure token budget not exceeded

Benefits:
- Higher quality context (most relevant docs)
- Fewer tokens needed
- Better answer quality
- Reduced cost
""")
    
    # Simulate reranking scores
    print("\n📊 Reranking Example:")
    print("   Doc ID  | Source              | Tokens | Relevance")
    print("   " + "-" * 60)
    
    for i, doc in enumerate(selected_docs[:5]):
        # Simulate relevance score
        relevance = 0.95 - (i * 0.05)
        source = doc.metadata['source'][:18]
        tokens = doc.metadata['tokens']
        print(f"   {i+1:2d}      | {source:20s} | {tokens:4d}   | {relevance:.2f}")
    
    # Strategy 3: Context Compression
    print("\n" + "=" * 80)
    print("STRATEGY 3: Context Compression")
    print("=" * 80)
    
    print("""
✅ Compression Techniques:

1. Extractive Summarization:
   - Extract only relevant sentences from each document
   - Remove boilerplate and redundant information
   - Keep key facts and entities
   
2. Query-Focused Extraction:
   - For query "authentication methods"
   - Extract: Only authentication-related paragraphs
   - Skip: Unrelated sections about rate limiting, errors, etc.
   
3. Deduplication:
   - Remove duplicate information across documents
   - Keep most complete version
   
4. Sentence Filtering:
   - Score sentences by relevance to query
   - Keep only top-scoring sentences
   - Maintain coherence
""")
    
    # Example compression
    original_doc = """# API Authentication Guide

The Authentication API uses OAuth 2.0 for secure access. To authenticate:

1. Register your application at https://developer.example.com
2. Obtain your client_id and client_secret
3. Request an access token using the /oauth/token endpoint
4. Include the token in the Authorization header: Bearer <token>

Token Expiration:
Access tokens expire after 3600 seconds (1 hour). Refresh tokens are valid for 30 days."""
    
    compressed_doc = """Authentication API uses OAuth 2.0. Steps: (1) Register application, (2) Obtain client_id and client_secret, (3) Request access token from /oauth/token, (4) Include in Authorization header as Bearer token. Tokens expire after 1 hour."""
    
    original_tokens = count_tokens(original_doc)
    compressed_tokens = count_tokens(compressed_doc)
    reduction = ((original_tokens - compressed_tokens) / original_tokens) * 100
    
    print(f"\n📉 Compression Example:")
    print(f"   Original: {original_tokens} tokens")
    print(f"   Compressed: {compressed_tokens} tokens")
    print(f"   Reduction: {reduction:.1f}%")
    print(f"   ✓ Information preserved, tokens reduced!")
    
    # Strategy 4: Hierarchical Retrieval
    print("\n" + "=" * 80)
    print("STRATEGY 4: Hierarchical Retrieval")
    print("=" * 80)
    
    print("""
✅ Two-Stage Retrieval:

Stage 1 - Coarse Retrieval:
- Retrieve document summaries or titles
- Ask LLM which documents are relevant
- Low token cost (summaries only)

Stage 2 - Fine Retrieval:
- Retrieve full content only for selected documents
- Use full context from relevant docs
- Higher quality, controlled tokens

Example:
Query: "What are the patient's medications?"

Stage 1: Retrieve 20 document summaries (200 tokens)
LLM identifies: "Medical records" and "Prescription history" are relevant

Stage 2: Retrieve only those 2 full documents (800 tokens)
Total: 1000 tokens vs 5000 tokens for all documents
""")
    
    print("\n✅ Improvements:")
    print("   1. Context fits within token limits")
    print("   2. Most relevant information prioritized")
    print("   3. Lower API costs (fewer tokens)")
    print("   4. Faster inference (smaller context)")
    print("   5. Better answer quality (less noise)")
    print("   6. Scalable (works with large knowledge bases)")


def demonstrate_best_practices():
    """Show best practices for context window management."""
    print("\n" + "=" * 80)
    print("BEST PRACTICES: Context Window Management")
    print("=" * 80)
    
    print("""
🎯 PRODUCTION-READY STRATEGIES:

1. TOKEN BUDGET MANAGEMENT:
   
   class TokenBudgetManager:
       def __init__(self, model_limit: int):
           self.model_limit = model_limit
           self.allocation = {
               "system": 0.05,    # 5% for system prompt
               "query": 0.05,     # 5% for user query
               "context": 0.65,   # 65% for context
               "output": 0.20,    # 20% for response
               "margin": 0.05     # 5% safety margin
           }
       
       def get_context_budget(self) -> int:
           return int(self.model_limit * self.allocation["context"])
   
   Usage:
   budget_mgr = TokenBudgetManager(model_limit=4096)
   context_budget = budget_mgr.get_context_budget()  # 2662 tokens

2. SMART RETRIEVAL PIPELINE:

   Step 1: Initial Retrieval
   - Use vector similarity to get top 20-30 candidates
   
   Step 2: Reranking
   - Use cross-encoder or LLM to rerank by relevance
   - Select top 5-10 most relevant
   
   Step 3: Token Fitting
   - Sort by relevance score
   - Add documents until token budget reached
   - Always include top 3 documents
   
   Step 4: Context Compression (optional)
   - Summarize or extract key sentences
   - Remove redundant information

3. DYNAMIC TOP-K SELECTION:

   def dynamic_top_k(query_complexity: str) -> int:
       if query_complexity == "simple":
           return 3  # "What is X?" needs few docs
       elif query_complexity == "medium":
           return 5  # "Compare X and Y" needs more
       elif query_complexity == "complex":
           return 8  # "Analyze trends" needs many
       return 5  # Default

4. CONTEXT WINDOW SIZING:

   Model Selection Based on Query:
   - Simple factual queries → gpt-3.5-turbo (4K)
   - Complex multi-document queries → gpt-4-turbo (128K)
   - Cost optimization: Use smallest sufficient model

5. MONITORING & ALERTS:

   Metrics to Track:
   - Average tokens per query
   - Token budget utilization
   - Truncation events
   - Answer quality vs context size
   - Cost per query
   
   Alerts:
   - Token budget exceeded (truncation risk)
   - Consistently high token usage (optimization opportunity)
   - Quality drops with context size increase

6. FALLBACK STRATEGIES:

   if context_tokens > budget:
       # Option 1: Compress context
       compressed = compress_documents(docs)
       
       # Option 2: Reduce retrieval
       docs = docs[:top_k // 2]
       
       # Option 3: Use larger model
       model = upgrade_to_larger_model()
       
       # Option 4: Multi-turn dialogue
       answer_part_1 = query_with_context(docs[:5])
       answer_part_2 = query_with_context(docs[5:10])
       final = combine_answers([answer_part_1, answer_part_2])

7. COST OPTIMIZATION:

   Cost Calculation:
   - Input tokens: $0.0015 per 1K tokens (GPT-3.5)
   - Output tokens: $0.002 per 1K tokens
   
   Example:
   - Query with 3K context: $0.0045 input + $0.002 output = $0.0065
   - Query with 10K context: $0.015 input + $0.002 output = $0.017
   - 2.6x cost increase for 3.3x more context!
   
   Optimization:
   - Compress context: Save 40-60% on tokens
   - Smart retrieval: Only relevant docs
   - Caching: Reuse context for similar queries
""")

    # Show example code
    print("\n📋 Example: Production Context Manager")
    print("=" * 40)
    
    context_manager_code = '''
class SmartContextManager:
    """Manages context window with token budgeting."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.token_limit = self._get_model_limit(model)
        self.context_budget = int(self.token_limit * 0.65)
    
    def prepare_context(self, 
                       query: str,
                       retrieved_docs: List[Document]) -> Tuple[str, Dict]:
        """Prepare context within token budget."""
        
        # 1. Rerank documents
        reranked = self.rerank_documents(query, retrieved_docs)
        
        # 2. Select documents within budget
        selected = self.fit_to_budget(reranked, self.context_budget)
        
        # 3. Build context string
        context = self.build_context(selected)
        
        # 4. Verify token count
        actual_tokens = count_tokens(context, self.model)
        
        metadata = {
            "retrieved": len(retrieved_docs),
            "selected": len(selected),
            "tokens": actual_tokens,
            "budget": self.context_budget,
            "utilization": actual_tokens / self.context_budget
        }
        
        return context, metadata
    
    def fit_to_budget(self, docs: List[Document], 
                     budget: int) -> List[Document]:
        """Select documents that fit within token budget."""
        
        selected = []
        current_tokens = 0
        
        for doc in docs:
            doc_tokens = count_tokens(doc.page_content)
            
            if current_tokens + doc_tokens <= budget:
                selected.append(doc)
                current_tokens += doc_tokens
            else:
                # Try compression
                compressed = self.compress_document(doc)
                compressed_tokens = count_tokens(compressed)
                
                if current_tokens + compressed_tokens <= budget:
                    doc.page_content = compressed
                    selected.append(doc)
                    current_tokens += compressed_tokens
                else:
                    break  # Budget exhausted
        
        return selected

# Usage:
manager = SmartContextManager(model="gpt-3.5-turbo")
context, metadata = manager.prepare_context(query, retrieved_docs)

print(f"Context prepared: {metadata['tokens']} tokens")
print(f"Budget utilization: {metadata['utilization']:.1%}")
'''
    
    print(context_manager_code)


if __name__ == "__main__":
    print("\n🔥 RAG Failure Mode #5: Context Window Overflow\n")
    
    # Run the failure example
    run_failure()
    
    # Run the fixed example
    run_fixed()
    
    # Show best practices
    demonstrate_best_practices()
    
    print("\n" + "=" * 80)
    print("💡 Key Lessons:")
    print("=" * 80)
    print("""
1. Context window limits are real - plan for them
2. More context ≠ better answers (quality over quantity)
3. Token budgeting prevents overflow and reduces costs
4. Reranking ensures most relevant docs are included
5. Compression can preserve information while saving tokens
6. Position bias means middle context gets ignored
7. Monitor token usage in production

🎯 Best Practices:
☑ Implement token budget management
☑ Rerank retrieved documents
☑ Compress context when needed
☑ Select appropriate model for query complexity
☑ Monitor token usage and costs
☑ Test with realistic query volumes
☑ Have fallback strategies for overflow

⚠️  Production Checklist:
- Set context token budgets per model
- Implement reranking pipeline
- Add token counting to all queries
- Monitor utilization metrics
- Alert on budget overruns
- Optimize cost vs. quality tradeoff
- Test edge cases (very long queries, many results)

💰 Cost Optimization:
- 3-5 well-chosen documents often better than 20+ 
- Compression saves 40-60% on token costs
- Smart retrieval reduces API calls
- Cache common queries and contexts
- Use smaller models when appropriate
""")
