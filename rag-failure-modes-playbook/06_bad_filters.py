"""
RAG Failure Mode #1: Bad Filters (Metadata Filtering Gone Wrong)
=================================================================

Problem: Incorrect or overly restrictive metadata filtering breaks retrieval,
returning empty results or irrelevant documents even when relevant content exists.

Common Issues:
- Hardcoded filters that don't match actual metadata structure
- Typos in filter field names
- Wrong data types in filter conditions
- Overly restrictive filters that exclude relevant results
- Missing fallback when filtered search returns nothing
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Dict, Any
import os
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)


def load_sample_products() -> str:
    """Load e-commerce product data for demonstration."""
    with open('sample_data/ecommerce_products.txt', 'r') as f:
        return f.read()


def create_product_documents() -> List[Document]:
    """Create documents with rich metadata for filtering."""
    products_text = load_sample_products()
    
    # Split by product separator
    product_sections = products_text.split('\n---\n')
    
    documents = []
    for section in product_sections:
        if not section.strip():
            continue
            
        # Parse product information
        lines = section.strip().split('\n')
        metadata = {}
        content_lines = []
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                # Store metadata
                if key == 'product':
                    metadata['name'] = value
                elif key == 'category':
                    metadata['category'] = value
                    # Extract main category
                    metadata['main_category'] = value.split('>')[0].strip()
                elif key == 'brand':
                    metadata['brand'] = value
                elif key == 'price':
                    # Store as both string and float
                    metadata['price_str'] = value
                    try:
                        metadata['price'] = float(value.replace('$', ''))
                    except:
                        metadata['price'] = 0.0
                elif key == 'stock':
                    metadata['stock'] = value
                    metadata['in_stock'] = value.lower() == 'in stock'
                elif key == 'rating':
                    metadata['rating'] = value
                    try:
                        metadata['rating_float'] = float(value.split('/')[0])
                    except:
                        metadata['rating_float'] = 0.0
                elif key == 'release_date':
                    metadata['release_date'] = value
                
                content_lines.append(line)
            else:
                content_lines.append(line)
        
        content = '\n'.join(content_lines)
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    
    return documents


def print_section(title: str, color=Fore.CYAN):
    """Print a formatted section header."""
    print(f"\n{color}{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}{Style.RESET_ALL}\n")


def print_results(results: List[Document], title: str):
    """Print search results in a formatted way."""
    print(f"\n{Fore.YELLOW}{title}{Style.RESET_ALL}")
    print(f"Found {len(results)} results:\n")
    
    for i, doc in enumerate(results, 1):
        print(f"{Fore.GREEN}Result {i}:{Style.RESET_ALL}")
        print(f"  Name: {doc.metadata.get('name', 'N/A')}")
        print(f"  Category: {doc.metadata.get('category', 'N/A')}")
        print(f"  Brand: {doc.metadata.get('brand', 'N/A')}")
        print(f"  Price: {doc.metadata.get('price_str', 'N/A')}")
        print(f"  Stock: {doc.metadata.get('stock', 'N/A')}")
        print(f"  Rating: {doc.metadata.get('rating', 'N/A')}")
        print()


def run_failure():
    """
    Demonstrates BAD filtering that breaks retrieval.
    
    Problems:
    1. Typo in metadata field name
    2. Wrong data type for comparison
    3. Overly restrictive filters
    4. No fallback when filter returns empty results
    """
    print_section("FAILURE MODE: Bad Metadata Filters", Fore.RED)
    
    # Create documents and vectorstore
    documents = create_product_documents()
    print(f"📚 Created {len(documents)} product documents\n")
    
    # Show sample metadata structure
    print(f"{Fore.CYAN}Sample metadata structure:{Style.RESET_ALL}")
    if documents:
        sample_meta = documents[0].metadata
        for key, value in sample_meta.items():
            print(f"  {key}: {value} ({type(value).__name__})")
    
    # Use a simple embeddings for demo (no API key needed for metadata filtering)
    print(f"\n{Fore.YELLOW}Creating vector store...{Style.RESET_ALL}")
    
    # Check if OpenAI API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print(f"{Fore.RED}⚠️  OPENAI_API_KEY not set. Using dummy embeddings for demo.{Style.RESET_ALL}")
        print("For full functionality, set OPENAI_API_KEY environment variable.\n")
        # Create a simple demo without actual embeddings
        print(f"{Fore.CYAN}Simulating filter failures...{Style.RESET_ALL}\n")
        
        # Demonstrate the filter problems conceptually
        print(f"{Fore.RED}❌ FAILURE CASE 1: Typo in field name{Style.RESET_ALL}")
        print("Filter: {'catagory': 'Electronics'}  # Typo: 'catagory' instead of 'category'")
        print("Result: Returns 0 documents even though Electronics products exist\n")
        
        print(f"{Fore.RED}❌ FAILURE CASE 2: Wrong data type{Style.RESET_ALL}")
        print("Filter: {'price': '149.99'}  # String comparison")
        print("Actual metadata: {'price': 149.99}  # Float value")
        print("Result: Type mismatch - no documents returned\n")
        
        print(f"{Fore.RED}❌ FAILURE CASE 3: Overly restrictive filters{Style.RESET_ALL}")
        print("Filter: {'brand': 'TechSound', 'in_stock': True, 'rating_float': {'$gte': 4.8}}")
        print("Result: Too specific - excludes many relevant products\n")
        
        print(f"{Fore.RED}❌ FAILURE CASE 4: No fallback{Style.RESET_ALL}")
        print("User query: 'Best wireless headphones'")
        print("Filter: {'brand': 'NonExistentBrand'}")
        print("Result: Returns empty list, user sees 'No results found' even though relevant products exist\n")
        
        # Show what documents we have
        print(f"\n{Fore.GREEN}Available products that should match 'wireless headphones':{Style.RESET_ALL}")
        for doc in documents:
            if 'headphones' in doc.metadata.get('name', '').lower():
                print(f"  - {doc.metadata.get('name')} ({doc.metadata.get('brand')})")
        
        return
    
    # If API key is available, do actual retrieval
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    query = "Best wireless headphones under $200"
    print(f"\n{Fore.CYAN}User Query:{Style.RESET_ALL} {query}\n")
    
    # BAD CASE 1: Typo in metadata field name
    print(f"{Fore.RED}❌ FAILURE CASE 1: Typo in field name{Style.RESET_ALL}")
    try:
        # Wrong: 'catagory' instead of 'category'
        results = vectorstore.similarity_search(
            query,
            k=5,
            filter={'catagory': 'Electronics'}  # TYPO!
        )
        print_results(results, "Results with typo in filter:")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # BAD CASE 2: Wrong data type
    print(f"{Fore.RED}❌ FAILURE CASE 2: Wrong data type in filter{Style.RESET_ALL}")
    try:
        # Price stored as float, but filtering with string
        results = vectorstore.similarity_search(
            query,
            k=5,
            filter={'price': '149.99'}  # Should be float!
        )
        print_results(results, "Results with wrong data type:")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # BAD CASE 3: Overly restrictive
    print(f"{Fore.RED}❌ FAILURE CASE 3: Overly restrictive filters{Style.RESET_ALL}")
    try:
        results = vectorstore.similarity_search(
            query,
            k=5,
            filter={
                'brand': 'NonExistentBrand',  # Doesn't exist
                'in_stock': True,
                'rating_float': 5.0  # Too specific
            }
        )
        print_results(results, "Results with overly restrictive filter:")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # BAD CASE 4: No fallback
    print(f"{Fore.RED}❌ FAILURE CASE 4: No fallback when filter fails{Style.RESET_ALL}")
    results = vectorstore.similarity_search(
        query,
        k=5,
        filter={'brand': 'FakeBrand'}
    )
    if not results:
        print("Result: Empty list returned!")
        print("User sees: 'No results found' ❌")
        print("Reality: Relevant products exist, but filter excluded them all\n")


def run_fixed():
    """
    Demonstrates FIXED filtering with proper error handling.
    
    Solutions:
    1. Validate metadata field names before filtering
    2. Ensure correct data types
    3. Use smart filter logic (gradual relaxation)
    4. Implement fallback to unfiltered search
    5. Log filter effectiveness
    """
    print_section("FIXED VERSION: Smart Metadata Filtering", Fore.GREEN)
    
    # Create documents and vectorstore
    documents = create_product_documents()
    print(f"📚 Created {len(documents)} product documents\n")
    
    if not os.getenv('OPENAI_API_KEY'):
        print(f"{Fore.RED}⚠️  OPENAI_API_KEY not set. Using dummy embeddings for demo.{Style.RESET_ALL}")
        print("For full functionality, set OPENAI_API_KEY environment variable.\n")
        
        # Demonstrate the solutions conceptually
        print(f"{Fore.GREEN}✅ SOLUTION 1: Validate field names{Style.RESET_ALL}")
        print("Before filtering, check if field exists in metadata schema")
        print("Example:")
        print("  valid_fields = {'category', 'brand', 'price', 'in_stock', 'rating_float'}")
        print("  if filter_field not in valid_fields:")
        print("      log_warning(f'Invalid field: {filter_field}')")
        print("      # Use unfiltered search instead\n")
        
        print(f"{Fore.GREEN}✅ SOLUTION 2: Type checking and conversion{Style.RESET_ALL}")
        print("Example:")
        print("  if 'price' in filters:")
        print("      filters['price'] = float(filters['price'])  # Convert to correct type\n")
        
        print(f"{Fore.GREEN}✅ SOLUTION 3: Gradual filter relaxation{Style.RESET_ALL}")
        print("Strategy:")
        print("  1. Try with all filters")
        print("  2. If empty, remove least important filter")
        print("  3. Repeat until results found or no filters left")
        print("  4. This ensures users always get relevant results\n")
        
        print(f"{Fore.GREEN}✅ SOLUTION 4: Fallback to unfiltered search{Style.RESET_ALL}")
        print("Always have a safety net:")
        print("  results = search_with_filters(query, filters)")
        print("  if not results:")
        print("      log_warning('Filters too restrictive, using unfiltered search')")
        print("      results = search_without_filters(query)")
        print("  return results\n")
        
        print(f"{Fore.GREEN}✅ SOLUTION 5: Monitor filter effectiveness{Style.RESET_ALL}")
        print("Track metrics:")
        print("  - Filter success rate: 85% ✓")
        print("  - Average results with filters: 5.2")
        print("  - Average results without filters: 12.7")
        print("  - Fallback trigger rate: 15%\n")
        
        return
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    query = "Best wireless headphones under $200"
    print(f"\n{Fore.CYAN}User Query:{Style.RESET_ALL} {query}\n")
    
    # FIXED: Smart filtering with validation and fallback
    def smart_search(query: str, filters: Dict[str, Any] = None, k: int = 5) -> List[Document]:
        """
        Smart search with filter validation and fallback.
        """
        # Valid metadata fields
        valid_fields = {'category', 'main_category', 'brand', 'price', 'in_stock', 'rating_float', 'stock', 'name'}
        
        # Validate and clean filters
        if filters:
            cleaned_filters = {}
            for key, value in filters.items():
                if key not in valid_fields:
                    print(f"{Fore.YELLOW}⚠️  Skipping invalid filter field: {key}{Style.RESET_ALL}")
                    continue
                
                # Type conversion for price
                if key == 'price' and isinstance(value, str):
                    try:
                        value = float(value.replace('$', ''))
                        print(f"{Fore.GREEN}✓ Converted price filter to float: {value}{Style.RESET_ALL}")
                    except:
                        print(f"{Fore.YELLOW}⚠️  Could not convert price: {value}{Style.RESET_ALL}")
                        continue
                
                cleaned_filters[key] = value
            
            # Try with filters
            if cleaned_filters:
                print(f"{Fore.CYAN}Trying search with filters: {cleaned_filters}{Style.RESET_ALL}")
                results = vectorstore.similarity_search(query, k=k, filter=cleaned_filters)
                
                if results:
                    print(f"{Fore.GREEN}✓ Found {len(results)} results with filters{Style.RESET_ALL}")
                    return results
                else:
                    print(f"{Fore.YELLOW}⚠️  No results with filters, falling back to unfiltered search{Style.RESET_ALL}")
        
        # Fallback: unfiltered search
        print(f"{Fore.CYAN}Performing unfiltered search...{Style.RESET_ALL}")
        results = vectorstore.similarity_search(query, k=k)
        print(f"{Fore.GREEN}✓ Found {len(results)} results without filters{Style.RESET_ALL}")
        return results
    
    # CASE 1: Filter with validation
    print(f"\n{Fore.GREEN}✅ CASE 1: Smart handling of filter typos{Style.RESET_ALL}")
    results = smart_search(
        query,
        filters={'catagory': 'Electronics'}  # Typo, but handled gracefully
    )
    print_results(results[:2], "Results (with typo handling):")
    
    # CASE 2: Type conversion
    print(f"\n{Fore.GREEN}✅ CASE 2: Automatic type conversion{Style.RESET_ALL}")
    results = smart_search(
        query,
        filters={'main_category': 'Electronics'}  # Correct field and type
    )
    print_results(results[:2], "Results (with type conversion):")
    
    # CASE 3: Graceful degradation
    print(f"\n{Fore.GREEN}✅ CASE 3: Fallback when filters too restrictive{Style.RESET_ALL}")
    results = smart_search(
        query,
        filters={
            'brand': 'NonExistentBrand',
            'rating_float': 5.0
        }
    )
    print_results(results[:2], "Results (with fallback):")
    
    print(f"\n{Fore.GREEN}{'=' * 80}")
    print("KEY LESSONS:")
    print("=" * 80)
    print("1. ✅ Always validate filter field names against metadata schema")
    print("2. ✅ Implement type checking and automatic conversion")
    print("3. ✅ Use fallback to unfiltered search when filters return empty")
    print("4. ✅ Log filter effectiveness to identify issues")
    print("5. ✅ Consider gradual filter relaxation for better UX")
    print(f"{'=' * 80}{Style.RESET_ALL}\n")


def main():
    """Run both failure and fixed examples."""
    print(f"\n{Fore.MAGENTA}{'=' * 80}")
    print("RAG FAILURE MODE #1: Bad Metadata Filters")
    print("=" * 80)
    print("\nThis example demonstrates how incorrect metadata filtering breaks")
    print("retrieval and returns empty results even when relevant content exists.")
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
    print("  ❌ Hardcoded filters with typos")
    print("  ❌ No type validation")
    print("  ❌ Returns empty results")
    print("  ❌ Poor user experience")
    print("\nFIXED:")
    print("  ✅ Validates filter fields")
    print("  ✅ Automatic type conversion")
    print("  ✅ Fallback to unfiltered search")
    print("  ✅ Always returns relevant results")
    print(f"{'=' * 80}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
