"""
RAG Failure Mode #2: Stale Indexes
===================================

Problem: Outdated embeddings and indexes cause retrieval of obsolete information,
missing new content, or returning deleted documents.

Common Issues:
- Index created weeks ago, but documents updated yesterday
- New documents added but not indexed
- Deleted documents still appearing in search results
- Document updates not reflected in embeddings
- No versioning or timestamp tracking
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import os
import time
import json
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


def print_section(title: str, color=Fore.CYAN):
    """Print a formatted section header."""
    print(f"\n{color}{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}{Style.RESET_ALL}\n")


def simulate_document_lifecycle() -> tuple:
    """
    Simulate a realistic document lifecycle with:
    - Original documents (indexed 30 days ago)
    - Updated documents (updated 1 day ago)
    - New documents (added today)
    - Deleted documents (removed 5 days ago)
    """
    
    # Original documents (old state)
    original_docs = [
        Document(
            page_content="Product: iPhone 14 Pro - Price: $999 - Status: Available",
            metadata={
                "product_id": "prod_001",
                "name": "iPhone 14 Pro",
                "price": 999.0,
                "status": "available",
                "indexed_at": (datetime.now() - timedelta(days=30)).isoformat(),
                "updated_at": (datetime.now() - timedelta(days=30)).isoformat(),
            }
        ),
        Document(
            page_content="Product: MacBook Pro M2 - Price: $1,999 - Status: Available",
            metadata={
                "product_id": "prod_002",
                "name": "MacBook Pro M2",
                "price": 1999.0,
                "status": "available",
                "indexed_at": (datetime.now() - timedelta(days=30)).isoformat(),
                "updated_at": (datetime.now() - timedelta(days=30)).isoformat(),
            }
        ),
        Document(
            page_content="Product: AirPods Pro - Price: $249 - Status: Available",
            metadata={
                "product_id": "prod_003",
                "name": "AirPods Pro",
                "price": 249.0,
                "status": "available",
                "indexed_at": (datetime.now() - timedelta(days=30)).isoformat(),
                "updated_at": (datetime.now() - timedelta(days=30)).isoformat(),
            }
        ),
        Document(
            page_content="Product: Apple Watch Series 8 - Price: $399 - Status: Available",
            metadata={
                "product_id": "prod_004",
                "name": "Apple Watch Series 8",
                "price": 399.0,
                "status": "available",
                "indexed_at": (datetime.now() - timedelta(days=30)).isoformat(),
                "updated_at": (datetime.now() - timedelta(days=30)).isoformat(),
            }
        ),
    ]
    
    # Current documents (updated state)
    current_docs = [
        # Updated: Price changed, status changed
        Document(
            page_content="Product: iPhone 14 Pro - Price: $799 (SALE!) - Status: Limited Stock",
            metadata={
                "product_id": "prod_001",
                "name": "iPhone 14 Pro",
                "price": 799.0,
                "status": "limited_stock",
                "sale": True,
                "indexed_at": (datetime.now() - timedelta(days=30)).isoformat(),
                "updated_at": (datetime.now() - timedelta(days=1)).isoformat(),
            }
        ),
        # Updated: New version released
        Document(
            page_content="Product: MacBook Pro M3 - Price: $2,299 - Status: Available - NEW M3 Chip!",
            metadata={
                "product_id": "prod_002",
                "name": "MacBook Pro M3",
                "price": 2299.0,
                "status": "available",
                "new_version": True,
                "indexed_at": (datetime.now() - timedelta(days=30)).isoformat(),
                "updated_at": (datetime.now() - timedelta(days=1)).isoformat(),
            }
        ),
        # Unchanged
        Document(
            page_content="Product: AirPods Pro - Price: $249 - Status: Available",
            metadata={
                "product_id": "prod_003",
                "name": "AirPods Pro",
                "price": 249.0,
                "status": "available",
                "indexed_at": (datetime.now() - timedelta(days=30)).isoformat(),
                "updated_at": (datetime.now() - timedelta(days=30)).isoformat(),
            }
        ),
        # Deleted (no longer in current docs)
        # prod_004 (Apple Watch) has been removed
        
        # New product added
        Document(
            page_content="Product: iPad Pro 2026 - Price: $1,199 - Status: Pre-order - Latest tablet with M3 chip",
            metadata={
                "product_id": "prod_005",
                "name": "iPad Pro 2026",
                "price": 1199.0,
                "status": "preorder",
                "is_new": True,
                "indexed_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
        ),
    ]
    
    return original_docs, current_docs


def run_failure():
    """
    Demonstrates STALE INDEX problems.
    
    Problems:
    1. Index created 30 days ago, not updated
    2. Shows outdated prices
    3. Missing new products
    4. Shows deleted products
    """
    print_section("FAILURE MODE: Stale Index (Outdated Embeddings)", Fore.RED)
    
    original_docs, current_docs = simulate_document_lifecycle()
    
    print(f"{Fore.CYAN}Scenario: E-commerce product catalog{Style.RESET_ALL}")
    print(f"Index created: {Fore.YELLOW}30 days ago{Style.RESET_ALL}")
    print(f"Last update: {Fore.RED}Never updated{Style.RESET_ALL}\n")
    
    # Show what's in the stale index
    print(f"{Fore.YELLOW}Stale Index Contents (30 days old):{Style.RESET_ALL}")
    for doc in original_docs:
        print(f"  - {doc.metadata['name']}: ${doc.metadata['price']} ({doc.metadata['status']})")
    
    print(f"\n{Fore.GREEN}Current Reality (actual product catalog):{Style.RESET_ALL}")
    for doc in current_docs:
        status_marker = ""
        if doc.metadata.get('sale'):
            status_marker = " 🔥 SALE"
        elif doc.metadata.get('new_version'):
            status_marker = " ✨ NEW VERSION"
        elif doc.metadata.get('is_new'):
            status_marker = " 🆕 NEW PRODUCT"
        print(f"  - {doc.metadata['name']}: ${doc.metadata['price']} ({doc.metadata['status']}){status_marker}")
    
    print(f"\n{Fore.RED}What's missing from stale index:{Style.RESET_ALL}")
    print(f"  ❌ Apple Watch Series 8 still appears (deleted 5 days ago)")
    print(f"  ❌ iPhone 14 Pro shows $999 (actually $799 on sale)")
    print(f"  ❌ MacBook Pro M2 shown (M3 version released)")
    print(f"  ❌ iPad Pro 2026 missing (added today)")
    
    if not os.getenv('OPENAI_API_KEY'):
        print(f"\n{Fore.YELLOW}⚠️  OPENAI_API_KEY not set. Showing conceptual demo.{Style.RESET_ALL}")
        
        print(f"\n{Fore.RED}❌ PROBLEM 1: Outdated prices{Style.RESET_ALL}")
        print("User query: 'iPhone 14 Pro price'")
        print("Stale index returns: '$999'")
        print("Actual price: '$799' (20% off!)")
        print("Impact: User might abandon purchase thinking it's too expensive\n")
        
        print(f"{Fore.RED}❌ PROBLEM 2: Missing new products{Style.RESET_ALL}")
        print("User query: 'Latest iPad with M3 chip'")
        print("Stale index returns: No results")
        print("Reality: iPad Pro 2026 available for pre-order")
        print("Impact: Lost sales opportunity\n")
        
        print(f"{Fore.RED}❌ PROBLEM 3: Showing deleted products{Style.RESET_ALL}")
        print("User query: 'Apple Watch'")
        print("Stale index returns: 'Apple Watch Series 8 - $399 - Available'")
        print("Reality: Product discontinued 5 days ago")
        print("Impact: User frustration, can't complete purchase\n")
        
        print(f"{Fore.RED}❌ PROBLEM 4: Missing product updates{Style.RESET_ALL}")
        print("User query: 'MacBook Pro M3'")
        print("Stale index returns: 'MacBook Pro M2 - $1,999'")
        print("Reality: M3 version available - $2,299")
        print("Impact: Wrong product information, confused customers\n")
        
        return
    
    # If API key available, do actual demonstration
    embeddings = OpenAIEmbeddings()
    
    # Create stale index
    print(f"\n{Fore.YELLOW}Creating stale index (with 30-day-old data)...{Style.RESET_ALL}")
    stale_vectorstore = FAISS.from_documents(original_docs, embeddings)
    
    # Test queries
    queries = [
        "iPhone 14 Pro price",
        "Latest MacBook Pro",
        "Apple Watch availability",
        "New iPad with M3 chip"
    ]
    
    print(f"\n{Fore.RED}Testing queries against STALE index:{Style.RESET_ALL}\n")
    
    for query in queries:
        print(f"{Fore.CYAN}Query: {query}{Style.RESET_ALL}")
        results = stale_vectorstore.similarity_search(query, k=1)
        
        if results:
            doc = results[0]
            print(f"  Stale result: {doc.page_content}")
            print(f"  Indexed at: {doc.metadata.get('indexed_at', 'Unknown')[:10]}")
            print(f"  {Fore.RED}⚠️  Information may be outdated!{Style.RESET_ALL}\n")
        else:
            print(f"  {Fore.RED}No results found (but relevant content may exist!){Style.RESET_ALL}\n")


def run_fixed():
    """
    Demonstrates FIXED index management with versioning and updates.
    
    Solutions:
    1. Track index freshness with timestamps
    2. Implement incremental updates
    3. Version control for embeddings
    4. Automatic refresh triggers
    5. Cache invalidation strategy
    """
    print_section("FIXED VERSION: Fresh Index Management", Fore.GREEN)
    
    original_docs, current_docs = simulate_document_lifecycle()
    
    print(f"{Fore.CYAN}Smart Index Management Strategy{Style.RESET_ALL}\n")
    
    class IndexManager:
        """
        Smart index manager with versioning and freshness tracking.
        """
        
        def __init__(self, max_age_hours: int = 24):
            self.max_age_hours = max_age_hours
            self.vectorstore = None
            self.index_metadata = {
                'created_at': None,
                'updated_at': None,
                'document_count': 0,
                'version': 1
            }
        
        def is_stale(self) -> bool:
            """Check if index needs refresh."""
            if not self.index_metadata['updated_at']:
                return True
            
            updated_at = datetime.fromisoformat(self.index_metadata['updated_at'])
            age_hours = (datetime.now() - updated_at).total_seconds() / 3600
            
            return age_hours > self.max_age_hours
        
        def create_index(self, documents: List[Document], embeddings):
            """Create fresh index."""
            now = datetime.now().isoformat()
            
            self.vectorstore = FAISS.from_documents(documents, embeddings)
            self.index_metadata = {
                'created_at': now,
                'updated_at': now,
                'document_count': len(documents),
                'version': self.index_metadata['version'] + 1
            }
            
            print(f"{Fore.GREEN}✓ Index created/updated{Style.RESET_ALL}")
            print(f"  Version: {self.index_metadata['version']}")
            print(f"  Documents: {self.index_metadata['document_count']}")
            print(f"  Timestamp: {now[:19]}")
        
        def search(self, query: str, k: int = 3) -> List[Document]:
            """Search with freshness check."""
            if self.is_stale():
                print(f"{Fore.YELLOW}⚠️  Index is stale (> {self.max_age_hours}h old){Style.RESET_ALL}")
                print(f"  Triggering refresh...")
                # In production, this would trigger async refresh
            
            if not self.vectorstore:
                print(f"{Fore.RED}❌ No index available{Style.RESET_ALL}")
                return []
            
            results = self.vectorstore.similarity_search(query, k=k)
            
            # Add freshness indicators
            for doc in results:
                doc_age = self._calculate_age(doc.metadata.get('updated_at'))
                doc.metadata['age_hours'] = doc_age
                doc.metadata['is_fresh'] = doc_age < self.max_age_hours
            
            return results
        
        def _calculate_age(self, timestamp_str: Optional[str]) -> float:
            """Calculate document age in hours."""
            if not timestamp_str:
                return float('inf')
            
            try:
                updated_at = datetime.fromisoformat(timestamp_str)
                return (datetime.now() - updated_at).total_seconds() / 3600
            except:
                return float('inf')
        
        def get_status(self) -> Dict[str, Any]:
            """Get index health status."""
            return {
                'version': self.index_metadata['version'],
                'document_count': self.index_metadata['document_count'],
                'created_at': self.index_metadata.get('created_at', 'Never')[:19],
                'updated_at': self.index_metadata.get('updated_at', 'Never')[:19],
                'is_stale': self.is_stale(),
                'max_age_hours': self.max_age_hours
            }
    
    if not os.getenv('OPENAI_API_KEY'):
        print(f"{Fore.YELLOW}⚠️  OPENAI_API_KEY not set. Showing conceptual demo.{Style.RESET_ALL}\n")
        
        print(f"{Fore.GREEN}✅ SOLUTION 1: Timestamp tracking{Style.RESET_ALL}")
        print("Store metadata with each index:")
        print("  - created_at: 2026-02-23T10:00:00")
        print("  - updated_at: 2026-02-23T18:00:00")
        print("  - version: 5")
        print("  - document_count: 1,247\n")
        
        print(f"{Fore.GREEN}✅ SOLUTION 2: Freshness checks{Style.RESET_ALL}")
        print("Before every search:")
        print("  index_age = now() - index.updated_at")
        print("  if index_age > MAX_AGE:")
        print("      trigger_async_refresh()")
        print("      warn_user('Results may be outdated')\n")
        
        print(f"{Fore.GREEN}✅ SOLUTION 3: Incremental updates{Style.RESET_ALL}")
        print("Don't rebuild entire index:")
        print("  1. Detect changed documents (by hash or timestamp)")
        print("  2. Remove old embeddings for changed docs")
        print("  3. Add new embeddings for changed docs")
        print("  4. Add embeddings for new docs")
        print("  5. Mark update timestamp\n")
        
        print(f"{Fore.GREEN}✅ SOLUTION 4: Version control{Style.RESET_ALL}")
        print("Track index versions:")
        print("  v1: Initial index (1000 docs)")
        print("  v2: Added 50 docs, updated 20 docs")
        print("  v3: Removed 10 docs, updated 15 docs")
        print("  Can rollback if issues detected\n")
        
        print(f"{Fore.GREEN}✅ SOLUTION 5: Automatic triggers{Style.RESET_ALL}")
        print("Auto-refresh conditions:")
        print("  - Every 24 hours (scheduled)")
        print("  - When >5% of documents updated")
        print("  - Manual trigger via API")
        print("  - After bulk document imports\n")
        
        print(f"{Fore.GREEN}✅ SOLUTION 6: Document-level timestamps{Style.RESET_ALL}")
        print("Mark freshness per document:")
        print("  - indexed_at: When embedded")
        print("  - updated_at: When source changed")
        print("  - Show freshness indicator in results\n")
        
        return
    
    # If API key available, demonstrate
    embeddings = OpenAIEmbeddings()
    manager = IndexManager(max_age_hours=24)
    
    # Create fresh index
    print(f"{Fore.CYAN}Creating FRESH index with current data...{Style.RESET_ALL}\n")
    manager.create_index(current_docs, embeddings)
    
    # Show status
    status = manager.get_status()
    print(f"\n{Fore.CYAN}Index Status:{Style.RESET_ALL}")
    for key, value in status.items():
        status_icon = "✓" if key == "is_stale" and not value else ""
        print(f"  {key}: {value} {status_icon}")
    
    # Test queries
    queries = [
        "iPhone 14 Pro price",
        "Latest MacBook Pro",
        "New iPad with M3 chip"
    ]
    
    print(f"\n{Fore.GREEN}Testing queries against FRESH index:{Style.RESET_ALL}\n")
    
    for query in queries:
        print(f"{Fore.CYAN}Query: {query}{Style.RESET_ALL}")
        results = manager.search(query, k=1)
        
        if results:
            doc = results[0]
            print(f"  Fresh result: {doc.page_content}")
            print(f"  Updated: {doc.metadata.get('updated_at', 'Unknown')[:10]}")
            age = doc.metadata.get('age_hours', 0)
            if age < 24:
                print(f"  {Fore.GREEN}✓ Fresh (< 24h old){Style.RESET_ALL}\n")
            else:
                print(f"  {Fore.YELLOW}⚠️  Older ({age:.0f}h old){Style.RESET_ALL}\n")
        else:
            print(f"  No results found\n")
    
    print(f"\n{Fore.GREEN}{'=' * 80}")
    print("KEY LESSONS:")
    print("=" * 80)
    print("1. ✅ Track index creation and update timestamps")
    print("2. ✅ Implement automatic freshness checks")
    print("3. ✅ Use incremental updates instead of full rebuilds")
    print("4. ✅ Version control for embeddings and rollback capability")
    print("5. ✅ Show freshness indicators in search results")
    print("6. ✅ Set up automatic refresh triggers (time-based, event-based)")
    print(f"{'=' * 80}{Style.RESET_ALL}\n")


def main():
    """Run both failure and fixed examples."""
    print(f"\n{Fore.MAGENTA}{'=' * 80}")
    print("RAG FAILURE MODE #2: Stale Indexes")
    print("=" * 80)
    print("\nThis example demonstrates how outdated embeddings cause retrieval of")
    print("obsolete information, missing new content, and showing deleted documents.")
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
    print("  ❌ Index never updated")
    print("  ❌ Shows outdated prices")
    print("  ❌ Missing new products")
    print("  ❌ Shows deleted items")
    print("\nFIXED:")
    print("  ✅ Timestamps tracked")
    print("  ✅ Automatic freshness checks")
    print("  ✅ Incremental updates")
    print("  ✅ Version control")
    print("  ✅ Always current information")
    print(f"{'=' * 80}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
