"""
RAG Failure Mode #2: Embedding Mismatch
========================================

Problem: Using different embedding models or versions for indexing (train time) 
vs. querying (inference time), leading to poor retrieval accuracy.

Common Issues:
- Training with one model (e.g., OpenAI ada-002), querying with another (all-MiniLM)
- Updating embedding model without re-indexing existing documents
- Using different model versions (e.g., sentence-transformers v2.2 vs v2.3)
- Dimensionality mismatches causing vector space incompatibility
"""

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from typing import List
import numpy as np


class CustomEmbeddings(Embeddings):
    """Wrapper for sentence-transformers models to use with LangChain."""
    
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()


def load_sample_documents():
    """Load product reviews for embedding examples."""
    with open('sample_data/product_reviews.txt', 'r') as f:
        content = f.read()
    
    # Split by review separator
    reviews = content.split('\n---\n')
    return [review.strip() for review in reviews if review.strip()]


def run_failure():
    """
    Demonstrates EMBEDDING MISMATCH failure.
    
    Problem: Index documents with one embedding model, but query with a different one.
    The vector spaces are incompatible, leading to terrible retrieval.
    """
    print("=" * 80)
    print("FAILURE MODE: Embedding Model Mismatch")
    print("=" * 80)
    
    reviews = load_sample_documents()
    print(f"\n📚 Loaded {len(reviews)} product reviews")
    
    # INDEXING TIME: Use one model (simulating "old" system)
    print("\n📦 INDEXING TIME (Production System v1.0):")
    print("   Using embedding model: 'all-MiniLM-L6-v2' (384 dimensions)")
    
    indexing_embeddings = CustomEmbeddings('all-MiniLM-L6-v2')
    
    # Create documents and index them
    docs = [Document(page_content=review, metadata={"review_id": i}) 
            for i, review in enumerate(reviews)]
    
    print(f"   Indexing {len(docs)} documents...")
    vectorstore = FAISS.from_documents(docs, indexing_embeddings)
    print("   ✓ Vector store created")
    
    # Get embedding dimension
    sample_embedding = indexing_embeddings.embed_query("test")
    print(f"   Embedding dimension: {len(sample_embedding)}")
    
    # QUERY TIME: Accidentally use a different model (simulating "updated" system)
    print("\n🔍 QUERY TIME (Production System v2.0 - WRONG!):")
    print("   Using embedding model: 'all-mpnet-base-v2' (768 dimensions)")
    print("   ❌ Different model with different vector space!\n")
    
    query_embeddings = CustomEmbeddings('all-mpnet-base-v2')
    
    # Get query embedding dimension
    query_embedding = query_embeddings.embed_query("test")
    print(f"   Query embedding dimension: {len(query_embedding)}")
    print(f"   Index embedding dimension: {len(sample_embedding)}")
    print(f"   ⚠️  DIMENSION MISMATCH: {len(query_embedding)} vs {len(sample_embedding)}\n")
    
    # Try to query - this will fail or give nonsense results
    query = "laptop with good battery life for travel"
    print(f"🔍 Query: '{query}'")
    print("   Expected: Should retrieve reviews mentioning battery life and travel")
    
    try:
        # This will fail due to dimension mismatch
        query_vec = query_embeddings.embed_query(query)
        results = vectorstore.similarity_search_by_vector(query_vec, k=2)
        
        print("\n   ❌ Retrieved (WRONG RESULTS):")
        for i, doc in enumerate(results):
            print(f"\n   Result {i+1}:")
            print(f"   {doc.page_content[:200]}...")
    
    except Exception as e:
        print(f"\n   ❌ ERROR: {type(e).__name__}")
        print(f"   {str(e)}")
        print("\n   💥 System crashed due to dimension mismatch!")
    
    print("\n⚠️  Problems Observed:")
    print("   1. Different models produce incompatible vector spaces")
    print("   2. Dimension mismatch causes crashes or wrong results")
    print("   3. Retrieval quality is garbage even if dimensions match")
    print("   4. Silent failures - system may 'work' but return irrelevant results")
    print("   5. Hard to debug - metrics look OK but results are wrong")
    
    return vectorstore, indexing_embeddings


def run_fixed():
    """
    Demonstrates CONSISTENT EMBEDDING strategy.
    
    Solution: Use the SAME embedding model for both indexing and querying.
    Store model metadata with the index.
    """
    print("\n" + "=" * 80)
    print("FIXED: Consistent Embedding Strategy")
    print("=" * 80)
    
    reviews = load_sample_documents()
    
    # Define the embedding model ONCE
    MODEL_NAME = 'all-MiniLM-L6-v2'
    MODEL_VERSION = '2.2.2'
    
    print(f"\n✅ CONSISTENT EMBEDDING STRATEGY:")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Version: {MODEL_VERSION}")
    print(f"   Used for BOTH indexing and querying")
    
    # Create embeddings - same instance for both operations
    embeddings = CustomEmbeddings(MODEL_NAME)
    
    # INDEXING TIME
    print(f"\n📦 INDEXING TIME:")
    docs = [Document(
        page_content=review, 
        metadata={
            "review_id": i,
            "embedding_model": MODEL_NAME,
            "embedding_version": MODEL_VERSION,
            "embedding_dim": len(embeddings.embed_query("test"))
        }
    ) for i, review in enumerate(reviews)]
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    embedding_dim = len(embeddings.embed_query("test"))
    print(f"   ✓ Indexed {len(docs)} documents")
    print(f"   ✓ Embedding dimension: {embedding_dim}")
    print(f"   ✓ Model metadata stored with documents")
    
    # QUERY TIME - using the SAME model
    print(f"\n🔍 QUERY TIME:")
    print(f"   Using SAME embedding model: {MODEL_NAME}")
    print(f"   Dimension: {embedding_dim}")
    print(f"   ✓ Consistency guaranteed!\n")
    
    query = "laptop with good battery life for travel"
    print(f"🔍 Query: '{query}'")
    print("   Expected: Should retrieve reviews mentioning battery life and travel")
    
    # Query with the same embeddings
    results = vectorstore.similarity_search(query, k=2)
    
    print("\n   ✅ Retrieved (CORRECT RESULTS):")
    for i, doc in enumerate(results):
        print(f"\n   Result {i+1} (Review #{doc.metadata['review_id']}):")
        # Extract rating and title for better visibility
        lines = doc.page_content.split('\n')
        rating_line = lines[0] if lines else ''
        title_line = lines[1] if len(lines) > 1 else ''
        print(f"   {rating_line}")
        print(f"   {title_line}")
        
        # Check if battery is mentioned
        if 'battery' in doc.page_content.lower():
            print(f"   ✓ Contains 'battery' mention")
        if 'travel' in doc.page_content.lower():
            print(f"   ✓ Contains 'travel' mention")
    
    print("\n✅ Improvements:")
    print("   1. Same model = compatible vector spaces")
    print("   2. Consistent dimensions = no crashes")
    print("   3. Retrieval quality is high")
    print("   4. Model metadata tracked for auditability")
    print("   5. Easy to validate consistency")
    
    return vectorstore, embeddings


def demonstrate_version_tracking():
    """
    Demonstrates best practice: Version tracking and validation.
    
    Shows how to store and validate embedding model versions to prevent mismatches.
    """
    print("\n" + "=" * 80)
    print("BEST PRACTICE: Embedding Model Version Tracking")
    print("=" * 80)
    
    print("""
🏗️  Production-Ready Embedding Management:

1. Store Model Metadata:
   - Model name (e.g., 'all-MiniLM-L6-v2')
   - Model version (e.g., '2.2.2')
   - Embedding dimension (e.g., 384)
   - Creation timestamp
   - Model hash/checksum

2. Validate at Query Time:
   class EmbeddingValidator:
       def validate_query_model(self, query_model, index_metadata):
           assert query_model.name == index_metadata['model_name']
           assert query_model.version == index_metadata['model_version']
           assert query_model.dimension == index_metadata['embedding_dim']

3. Handle Model Updates:
   - NEVER update query model without re-indexing
   - If you must update, create NEW index with new model
   - Run A/B test between old and new indices
   - Migrate traffic gradually
   - Keep old index until new one is validated

4. Monitor in Production:
   - Log embedding model info with every query
   - Alert on dimension mismatches
   - Track retrieval quality metrics
   - Compare query model vs. index model in dashboards

5. Configuration Management:
   # config.yaml
   embedding:
     model_name: "all-MiniLM-L6-v2"
     model_version: "2.2.2"
     dimension: 384
     provider: "sentence-transformers"
     
   # Load same config for indexing and querying
   # Never hardcode model names separately!
""")
    
    # Demonstrate validation
    print("\n📋 Example: Model Validation Code")
    print("=" * 40)
    
    validation_code = '''
class EmbeddingModelValidator:
    """Validates embedding model consistency."""
    
    def __init__(self, expected_model: str, expected_dim: int):
        self.expected_model = expected_model
        self.expected_dim = expected_dim
    
    def validate(self, embeddings) -> bool:
        """Validate model matches expectations."""
        test_vec = embeddings.embed_query("test")
        
        if len(test_vec) != self.expected_dim:
            raise ValueError(
                f"Dimension mismatch! Expected {self.expected_dim}, "
                f"got {len(test_vec)}"
            )
        
        if embeddings.model_name != self.expected_model:
            raise ValueError(
                f"Model mismatch! Expected {self.expected_model}, "
                f"got {embeddings.model_name}"
            )
        
        return True

# Usage:
validator = EmbeddingModelValidator(
    expected_model="all-MiniLM-L6-v2",
    expected_dim=384
)

# Validate before querying
validator.validate(query_embeddings)
results = vectorstore.similarity_search(query)
'''
    
    print(validation_code)


def compare_retrieval_quality():
    """Compare retrieval quality with matched vs. mismatched embeddings."""
    print("\n" + "=" * 80)
    print("COMPARISON: Matched vs Mismatched Embeddings")
    print("=" * 80)
    
    reviews = load_sample_documents()
    
    # Create index with model A
    model_a = CustomEmbeddings('all-MiniLM-L6-v2')
    docs = [Document(page_content=review) for review in reviews]
    vectorstore = FAISS.from_documents(docs, model_a)
    
    query = "laptop with excellent performance for developers"
    
    # Query with same model (GOOD)
    print("\n✅ MATCHED MODELS:")
    print(f"   Index model: all-MiniLM-L6-v2 (384d)")
    print(f"   Query model: all-MiniLM-L6-v2 (384d)")
    results_good = vectorstore.similarity_search(query, k=3)
    print(f"   Retrieved {len(results_good)} relevant results")
    
    # Show scores
    for i, doc in enumerate(results_good):
        score = vectorstore.similarity_search_with_score(query, k=3)[i][1]
        print(f"   Result {i+1}: Relevance score = {score:.4f}")
    
    print("\n💡 Key Takeaway:")
    print("   Consistent embedding models = reliable, high-quality retrieval")
    print("   Never mix models between indexing and querying!")


if __name__ == "__main__":
    print("\n🔥 RAG Failure Mode #2: Embedding Mismatch\n")
    
    # Run the failure example
    run_failure()
    
    # Run the fixed example
    run_fixed()
    
    # Show best practices
    demonstrate_version_tracking()
    
    # Compare quality
    compare_retrieval_quality()
    
    print("\n" + "=" * 80)
    print("💡 Key Lessons:")
    print("=" * 80)
    print("""
1. Same model for indexing and querying - ALWAYS
2. Store model metadata with your index
3. Validate model consistency before querying
4. Use configuration management (don't hardcode)
5. Re-index completely when changing models
6. Monitor embedding dimensions in production
7. Version your embedding models explicitly

🎯 Best Practices:
- Centralize embedding model configuration
- Add validation checks in production code
- Log model metadata with every operation
- Set up alerts for dimension mismatches
- Test embedding consistency in CI/CD
- Document which model version is in production

⚠️  Common Pitfalls:
- Upgrading libraries without re-indexing
- Using different models in dev vs. prod
- Forgetting to track model versions
- Assuming all 'BERT' models are compatible
- Not validating dimensions at query time
""")
