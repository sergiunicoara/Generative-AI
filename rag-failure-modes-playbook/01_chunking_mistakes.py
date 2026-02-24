"""
RAG Failure Mode #1: Chunking Mistakes
======================================

Problem: Poor text splitting that destroys semantic context by breaking related 
information across chunks or creating chunks that are too small/large.

Common Issues:
- Breaking in the middle of sentences or code blocks
- Separating titles from their content
- Creating chunks too small (lacking context) or too large (diluting relevance)
- Not accounting for document structure (headers, lists, code)
"""

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os


def load_sample_document():
    """Load the technical documentation for chunking examples."""
    with open('sample_data/technical_docs.txt', 'r') as f:
        return f.read()


def run_failure():
    """
    Demonstrates BAD chunking that breaks context.
    
    Problem: Using arbitrary character splits without considering document structure.
    This breaks API documentation in the middle of code examples and separates
    headers from their content.
    """
    print("=" * 80)
    print("FAILURE MODE: Poor Chunking Strategy")
    print("=" * 80)
    
    doc_text = load_sample_document()
    
    # BAD: Using simple character splitter with no overlap and arbitrary size
    # This ignores markdown structure, code blocks, and semantic boundaries
    bad_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,  # Too small - breaks context
        chunk_overlap=0,  # No overlap - loses continuity
        length_function=len,
    )
    
    chunks = bad_splitter.split_text(doc_text)
    
    print(f"\n❌ Bad Chunking Strategy:")
    print(f"   - Total chunks: {len(chunks)}")
    print(f"   - Chunk size: 200 characters")
    print(f"   - Overlap: 0")
    print("\n📄 Example of broken chunks:\n")
    
    # Show how bad chunking breaks context
    for i, chunk in enumerate(chunks[:5]):
        print(f"--- Chunk {i+1} (length: {len(chunk)}) ---")
        print(chunk[:150] + "..." if len(chunk) > 150 else chunk)
        print()
    
    print("\n⚠️  Problems Observed:")
    print("   1. Headers separated from their content")
    print("   2. Code examples broken mid-block")
    print("   3. Related information split across chunks")
    print("   4. Chunks too small - lacking sufficient context")
    print("   5. No overlap means lost context at boundaries")
    
    # Demonstrate retrieval failure
    print("\n🔍 Retrieval Test: 'How do I authenticate with OAuth?'")
    
    # Create embeddings and vector store
    try:
        embeddings = OpenAIEmbeddings()
        docs = [Document(page_content=chunk) for chunk in chunks]
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # Try to retrieve relevant chunks
        query = "How do I authenticate with OAuth?"
        retrieved_docs = vectorstore.similarity_search(query, k=2)
        
        print("\n   Retrieved chunks:")
        for i, doc in enumerate(retrieved_docs):
            print(f"\n   Chunk {i+1}:")
            print(f"   {doc.page_content[:200]}...")
        
        print("\n   ❌ Result: Incomplete information - authentication steps split across chunks!")
        
    except Exception as e:
        print(f"\n   ⚠️  Note: OpenAI API key required for embeddings: {e}")
        print("   Set OPENAI_API_KEY environment variable to test retrieval")
    
    return chunks


def run_fixed():
    """
    Demonstrates GOOD chunking that preserves context.
    
    Solution: Use RecursiveCharacterTextSplitter with appropriate size, overlap,
    and separators that respect document structure.
    """
    print("\n" + "=" * 80)
    print("FIXED: Semantic-Aware Chunking Strategy")
    print("=" * 80)
    
    doc_text = load_sample_document()
    
    # GOOD: Using recursive splitter with semantic separators
    # This respects markdown structure and preserves context
    good_splitter = RecursiveCharacterTextSplitter(
        # Separators in order of preference - preserves structure
        separators=[
            "\n\n\n",  # Multiple blank lines (section breaks)
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentence breaks
            " ",       # Word breaks
            ""         # Character breaks (last resort)
        ],
        chunk_size=500,      # Larger chunks for better context
        chunk_overlap=100,   # Overlap to maintain continuity
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = good_splitter.split_text(doc_text)
    
    print(f"\n✅ Good Chunking Strategy:")
    print(f"   - Total chunks: {len(chunks)}")
    print(f"   - Chunk size: 500 characters")
    print(f"   - Overlap: 100 characters")
    print(f"   - Semantic separators: respects headers, paragraphs, code blocks")
    print("\n📄 Example of well-formed chunks:\n")
    
    # Show how good chunking preserves context
    for i, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {i+1} (length: {len(chunk)}) ---")
        print(chunk)
        print()
    
    print("\n✅ Improvements:")
    print("   1. Headers kept with their content")
    print("   2. Code blocks remain intact")
    print("   3. Related information stays together")
    print("   4. Overlap ensures context continuity")
    print("   5. Chunk size balances context vs. relevance")
    
    # Demonstrate successful retrieval
    print("\n🔍 Retrieval Test: 'How do I authenticate with OAuth?'")
    
    try:
        embeddings = OpenAIEmbeddings()
        docs = [Document(page_content=chunk, metadata={"chunk_id": i}) 
                for i, chunk in enumerate(chunks)]
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # Try to retrieve relevant chunks
        query = "How do I authenticate with OAuth?"
        retrieved_docs = vectorstore.similarity_search(query, k=2)
        
        print("\n   Retrieved chunks:")
        for i, doc in enumerate(retrieved_docs):
            print(f"\n   Chunk {i+1} (ID: {doc.metadata.get('chunk_id')}):")
            print(f"   {doc.page_content[:300]}...")
        
        print("\n   ✅ Result: Complete authentication flow retrieved!")
        print("   All necessary steps are in the retrieved chunks.")
        
    except Exception as e:
        print(f"\n   ⚠️  Note: OpenAI API key required for embeddings: {e}")
        print("   Set OPENAI_API_KEY environment variable to test retrieval")
    
    return chunks


def compare_strategies():
    """Compare chunking strategies side-by-side."""
    print("\n" + "=" * 80)
    print("COMPARISON: Bad vs Good Chunking")
    print("=" * 80)
    
    doc_text = load_sample_document()
    
    # Bad strategy
    bad_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=0,
    )
    bad_chunks = bad_splitter.split_text(doc_text)
    
    # Good strategy
    good_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
        chunk_size=500,
        chunk_overlap=100,
    )
    good_chunks = good_splitter.split_text(doc_text)
    
    print(f"\n📊 Metrics:")
    print(f"   Bad Strategy:  {len(bad_chunks)} chunks, avg {sum(len(c) for c in bad_chunks) / len(bad_chunks):.0f} chars")
    print(f"   Good Strategy: {len(good_chunks)} chunks, avg {sum(len(c) for c in good_chunks) / len(good_chunks):.0f} chars")
    
    print(f"\n💡 Key Takeaway:")
    print(f"   Good chunking reduces total chunks while INCREASING context quality.")
    print(f"   Fewer, better chunks = better retrieval + lower costs")


if __name__ == "__main__":
    print("\n🔥 RAG Failure Mode #1: Chunking Mistakes\n")
    
    # Run the failure example
    run_failure()
    
    # Run the fixed example
    run_fixed()
    
    # Compare strategies
    compare_strategies()
    
    print("\n" + "=" * 80)
    print("💡 Key Lessons:")
    print("=" * 80)
    print("""
1. Chunk size matters: Too small = no context, too large = diluted relevance
2. Overlap is crucial: Prevents losing context at chunk boundaries
3. Respect structure: Use semantic separators (paragraphs, headers, code blocks)
4. Test retrieval: Always validate that related info stays together
5. Domain-specific: Adjust strategy based on content (code, legal, medical, etc.)

🎯 Best Practices:
- Start with RecursiveCharacterTextSplitter
- Chunk size: 500-1000 characters for most use cases
- Overlap: 10-20% of chunk size
- Use semantic separators appropriate for your content
- Test retrieval quality with real queries
""")
