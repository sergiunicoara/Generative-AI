"""
RAG Failure Mode #3: Multilingual Queries
==========================================

Problem: Cross-language retrieval fails when query language doesn't match 
document language, even when semantically identical content exists.

Common Issues:
- Query in English, documents in Spanish → no results
- Embedding models trained primarily on English
- No language detection or translation
- Mixed-language documents poorly handled
- Language-specific embeddings not used
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Dict, Optional
import os
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print(f"{Fore.YELLOW}⚠️  langdetect not installed. Using fallback.{Style.RESET_ALL}")

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print(f"{Fore.YELLOW}⚠️  deep-translator not installed. Using fallback.{Style.RESET_ALL}")


def print_section(title: str, color=Fore.CYAN):
    """Print a formatted section header."""
    print(f"\n{color}{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}{Style.RESET_ALL}\n")


def load_multilingual_content() -> List[Document]:
    """Load multilingual sample documents."""
    with open('sample_data/multilingual_content.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse by language sections
    sections = content.split('\n---\n')
    
    documents = []
    for section in sections:
        if not section.strip():
            continue
        
        lines = section.strip().split('\n')
        
        # Extract language, title, content
        language = None
        title = None
        content_text = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.endswith(':') and not title:
                # Language identifier
                language = line.rstrip(':')
            elif line.startswith('Title:') or line.startswith('Titre:') or \
                 line.startswith('Título:') or line.startswith('Titel:') or \
                 line.startswith('标题:') or line.startswith('タイトル:') or \
                 line.startswith('Заголовок:') or line.startswith('العنوان:') or \
                 line.startswith('Titolo:') or line.startswith('Тiтул:'):
                title = line.split(':', 1)[1].strip()
            elif line.startswith('Content:') or line.startswith('Contenu:') or \
                 line.startswith('Contenido:') or line.startswith('Inhalt:') or \
                 line.startswith('内容:') or line.startswith('内容:') or \
                 line.startswith('Содержание:') or line.startswith('المحتوى:') or \
                 line.startswith('Contenuto:') or line.startswith('Conteúdo:'):
                content_text.append(line.split(':', 1)[1].strip())
            else:
                content_text.append(line)
        
        if title and content_text:
            full_content = f"{title}\n\n{' '.join(content_text)}"
            doc = Document(
                page_content=full_content,
                metadata={
                    'language': language,
                    'title': title,
                    'topic': 'machine_learning'
                }
            )
            documents.append(doc)
    
    return documents


def detect_language(text: str) -> str:
    """Detect language of text."""
    if not LANGDETECT_AVAILABLE:
        return 'unknown'
    
    try:
        lang_code = detect(text)
        
        # Map to full names
        lang_map = {
            'en': 'English',
            'fr': 'French',
            'es': 'Spanish',
            'de': 'German',
            'zh-cn': 'Chinese',
            'ja': 'Japanese',
            'ru': 'Russian',
            'ar': 'Arabic',
            'pt': 'Portuguese',
            'it': 'Italian',
            'nl': 'Dutch'
        }
        
        return lang_map.get(lang_code, lang_code)
    except LangDetectException:
        return 'unknown'


def translate_text(text: str, target_lang: str = 'en') -> Optional[str]:
    """Translate text to target language."""
    if not TRANSLATOR_AVAILABLE:
        return None
    
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        return translator.translate(text)
    except Exception as e:
        print(f"{Fore.RED}Translation error: {e}{Style.RESET_ALL}")
        return None


def run_failure():
    """
    Demonstrates MULTILINGUAL RETRIEVAL failures.
    
    Problems:
    1. English query on Spanish documents → no results
    2. No language detection
    3. No translation layer
    4. Embedding space mismatch across languages
    """
    print_section("FAILURE MODE: Multilingual Query Failures", Fore.RED)
    
    documents = load_multilingual_content()
    
    print(f"{Fore.CYAN}Document Collection:{Style.RESET_ALL}")
    print(f"Total documents: {len(documents)}\n")
    
    # Show available languages
    languages = set()
    for doc in documents:
        lang = doc.metadata.get('language', 'Unknown')
        languages.add(lang)
        print(f"  {lang}: {doc.metadata.get('title', 'Untitled')[:50]}")
    
    print(f"\n{Fore.YELLOW}Available languages: {', '.join(sorted(languages))}{Style.RESET_ALL}")
    
    if not os.getenv('OPENAI_API_KEY'):
        print(f"\n{Fore.YELLOW}⚠️  OPENAI_API_KEY not set. Showing conceptual demo.{Style.RESET_ALL}\n")
        
        print(f"{Fore.RED}❌ PROBLEM 1: Language mismatch{Style.RESET_ALL}")
        print("Documents: French, Spanish, German, Chinese, Arabic")
        print("User query: 'What is machine learning?' (English)")
        print("Embedding space: Trained primarily on English")
        print("Result: Poor retrieval of non-English content")
        print("Why: Semantic similarity breaks across languages\n")
        
        print(f"{Fore.RED}❌ PROBLEM 2: No language detection{Style.RESET_ALL}")
        print("System doesn't detect that:")
        print("  - Query is in English")
        print("  - Documents are in multiple languages")
        print("  - Translation might help")
        print("Result: Naive retrieval, poor results\n")
        
        print(f"{Fore.RED}❌ PROBLEM 3: Missing Spanish content{Style.RESET_ALL}")
        print("Spanish user searches: '¿Qué es el aprendizaje automático?'")
        print("System behavior:")
        print("  1. Embeds Spanish query using English-biased model")
        print("  2. Searches against Spanish document embeddings")
        print("  3. Similarity scores are lower than they should be")
        print("  4. Relevant Spanish content ranked poorly")
        print("Result: User gets English content instead of Spanish\n")
        
        print(f"{Fore.RED}❌ PROBLEM 4: Mixed-language confusion{Style.RESET_ALL}")
        print("Document: English title + French content")
        print("Query: 'Introduction to AI' (English)")
        print("Result: Partial match on title, but content mismatch")
        print("Why: Embedding captures mixed signals\n")
        
        print(f"{Fore.RED}❌ PROBLEM 5: Character encoding issues{Style.RESET_ALL}")
        print("Languages: Arabic (RTL), Chinese (characters), Cyrillic")
        print("Tokenizer: Designed for Latin alphabet")
        print("Result: Poor tokenization → poor embeddings → poor retrieval\n")
        
        # Demonstrate with language detection if available
        if LANGDETECT_AVAILABLE:
            test_queries = [
                ("What is machine learning?", "English"),
                ("¿Qué es el aprendizaje automático?", "Spanish"),
                ("Qu'est-ce que l'apprentissage automatique?", "French"),
                ("Was ist maschinelles Lernen?", "German"),
            ]
            
            print(f"{Fore.CYAN}Language Detection (if we had it):{Style.RESET_ALL}")
            for query, expected_lang in test_queries:
                detected = detect_language(query)
                match_icon = "✓" if expected_lang.lower() in detected.lower() else "✗"
                print(f"  {match_icon} '{query[:40]}...' → {detected}")
        
        return
    
    # If API key available, demonstrate
    embeddings = OpenAIEmbeddings()
    
    print(f"\n{Fore.YELLOW}Creating vector store (English-biased embeddings)...{Style.RESET_ALL}")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Test cross-language queries
    test_cases = [
        ("What is machine learning?", "English", ["English", "French", "Spanish"]),
        ("apprentissage automatique", "French", ["French"]),
        ("aprendizaje automático", "Spanish", ["Spanish"]),
    ]
    
    print(f"\n{Fore.RED}Testing cross-language retrieval (FAILURE MODE):{Style.RESET_ALL}\n")
    
    for query, query_lang, expected_langs in test_cases:
        print(f"{Fore.CYAN}Query: '{query}' ({query_lang}){Style.RESET_ALL}")
        print(f"Expected to find: {', '.join(expected_langs)}")
        
        results = vectorstore.similarity_search(query, k=3)
        
        if results:
            print(f"  Retrieved {len(results)} results:")
            for i, doc in enumerate(results, 1):
                doc_lang = doc.metadata.get('language', 'Unknown')
                title = doc.metadata.get('title', 'No title')[:40]
                match_icon = "✓" if doc_lang in expected_langs else "✗"
                print(f"    {i}. {match_icon} {doc_lang}: {title}")
        else:
            print(f"  {Fore.RED}No results found!{Style.RESET_ALL}")
        
        print()


def run_fixed():
    """
    Demonstrates FIXED multilingual retrieval.
    
    Solutions:
    1. Language detection
    2. Query translation
    3. Multi-embedding strategy
    4. Language-aware ranking
    5. Fallback mechanisms
    """
    print_section("FIXED VERSION: Multilingual-Aware Retrieval", Fore.GREEN)
    
    documents = load_multilingual_content()
    
    class MultilingualRetriever:
        """
        Smart multilingual retriever with translation and language detection.
        """
        
        def __init__(self, documents: List[Document], embeddings):
            self.documents = documents
            self.embeddings = embeddings
            self.vectorstore = FAISS.from_documents(documents, embeddings)
            
            # Build language index
            self.language_index = {}
            for i, doc in enumerate(documents):
                lang = doc.metadata.get('language', 'Unknown')
                if lang not in self.language_index:
                    self.language_index[lang] = []
                self.language_index[lang].append(i)
        
        def detect_query_language(self, query: str) -> str:
            """Detect query language."""
            if LANGDETECT_AVAILABLE:
                return detect_language(query)
            return 'English'  # Default fallback
        
        def search_multilingual(self, query: str, k: int = 3) -> List[Document]:
            """
            Search with multilingual support.
            
            Strategy:
            1. Detect query language
            2. Search with original query
            3. If needed, translate and search again
            4. Merge and re-rank results
            """
            query_lang = self.detect_query_language(query)
            print(f"  Detected query language: {Fore.CYAN}{query_lang}{Style.RESET_ALL}")
            
            # Search with original query
            results = self.vectorstore.similarity_search(query, k=k*2)
            
            # If query is not in English and translator available, try translation
            if TRANSLATOR_AVAILABLE and 'english' not in query_lang.lower():
                print(f"  Translating query to English for broader search...")
                
                translated = translate_text(query, 'en')
                if translated:
                    print(f"  Translated: '{translated}'")
                    
                    # Search with translated query
                    translated_results = self.vectorstore.similarity_search(translated, k=k*2)
                    
                    # Merge results (remove duplicates)
                    seen_ids = set()
                    merged = []
                    for doc in results + translated_results:
                        doc_id = doc.page_content[:100]  # Use content as ID
                        if doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            merged.append(doc)
                    
                    results = merged
            
            # Rank by language match (prefer matching language)
            scored_results = []
            for doc in results:
                doc_lang = doc.metadata.get('language', 'Unknown')
                
                # Language match bonus
                lang_score = 1.0
                if query_lang.lower() in doc_lang.lower():
                    lang_score = 1.5  # Boost matching language
                
                scored_results.append((doc, lang_score))
            
            # Sort by score (higher is better)
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k
            return [doc for doc, score in scored_results[:k]]
        
        def get_language_stats(self) -> Dict[str, int]:
            """Get document count by language."""
            stats = {}
            for lang, doc_indices in self.language_index.items():
                stats[lang] = len(doc_indices)
            return stats
    
    if not os.getenv('OPENAI_API_KEY'):
        print(f"{Fore.YELLOW}⚠️  OPENAI_API_KEY not set. Showing conceptual demo.{Style.RESET_ALL}\n")
        
        print(f"{Fore.GREEN}✅ SOLUTION 1: Language detection{Style.RESET_ALL}")
        print("Detect language of every query:")
        print("  - User query: '¿Qué es AI?'")
        print("  - Detected: Spanish")
        print("  - Action: Prefer Spanish documents in ranking\n")
        
        print(f"{Fore.GREEN}✅ SOLUTION 2: Query translation{Style.RESET_ALL}")
        print("Multi-query approach:")
        print("  1. Search with original query: '¿Qué es AI?'")
        print("  2. Translate to English: 'What is AI?'")
        print("  3. Search with translated query")
        print("  4. Merge results, remove duplicates")
        print("  5. Boost documents matching original language\n")
        
        print(f"{Fore.GREEN}✅ SOLUTION 3: Language-specific embeddings{Style.RESET_ALL}")
        print("Use multilingual embedding models:")
        print("  - multilingual-e5-base")
        print("  - Sentence Transformers multilingual models")
        print("  - OpenAI's text-embedding-3 (better multilingual support)")
        print("  These models have aligned embeddings across languages\n")
        
        print(f"{Fore.GREEN}✅ SOLUTION 4: Document translation{Style.RESET_ALL}")
        print("Pre-translate high-value documents:")
        print("  - Original: Spanish documentation")
        print("  - Generate: English translation")
        print("  - Index both versions")
        print("  - Link translations via metadata")
        print("  - Serve appropriate version to user\n")
        
        print(f"{Fore.GREEN}✅ SOLUTION 5: Language-aware ranking{Style.RESET_ALL}")
        print("Ranking factors:")
        print("  - Semantic similarity: 60%")
        print("  - Language match: 30%")
        print("  - Freshness: 10%")
        print("  Example: Spanish query + Spanish doc → 1.3x boost\n")
        
        print(f"{Fore.GREEN}✅ SOLUTION 6: Fallback chain{Style.RESET_ALL}")
        print("If no results:")
        print("  1. Try relaxed filters")
        print("  2. Try translated query")
        print("  3. Try language-agnostic search")
        print("  4. Show results with 'Translated from X' label\n")
        
        # Show language detection examples
        if LANGDETECT_AVAILABLE:
            test_queries = [
                "What is machine learning?",
                "¿Qué es el aprendizaje automático?",
                "Qu'est-ce que l'apprentissage automatique?",
                "Was ist maschinelles Lernen?",
                "机器学习是什么？",
            ]
            
            print(f"{Fore.GREEN}Language Detection in Action:{Style.RESET_ALL}")
            for query in test_queries:
                detected = detect_language(query)
                print(f"  '{query[:40]}' → {detected}")
        
        return
    
    # If API key available, demonstrate
    embeddings = OpenAIEmbeddings()
    
    print(f"{Fore.CYAN}Creating multilingual-aware retriever...{Style.RESET_ALL}\n")
    retriever = MultilingualRetriever(documents, embeddings)
    
    # Show language distribution
    lang_stats = retriever.get_language_stats()
    print(f"{Fore.CYAN}Document language distribution:{Style.RESET_ALL}")
    for lang, count in sorted(lang_stats.items()):
        print(f"  {lang}: {count} document(s)")
    
    # Test multilingual queries
    test_queries = [
        "What is machine learning?",
        "aprendizaje automático",
        "apprentissage automatique",
    ]
    
    print(f"\n{Fore.GREEN}Testing multilingual-aware retrieval:{Style.RESET_ALL}\n")
    
    for query in test_queries:
        print(f"{Fore.CYAN}Query: '{query}'{Style.RESET_ALL}")
        results = retriever.search_multilingual(query, k=3)
        
        if results:
            print(f"  {Fore.GREEN}Found {len(results)} results:{Style.RESET_ALL}")
            for i, doc in enumerate(results, 1):
                doc_lang = doc.metadata.get('language', 'Unknown')
                title = doc.metadata.get('title', 'No title')[:50]
                print(f"    {i}. {doc_lang}: {title}")
        else:
            print(f"  {Fore.RED}No results{Style.RESET_ALL}")
        
        print()
    
    print(f"\n{Fore.GREEN}{'=' * 80}")
    print("KEY LESSONS:")
    print("=" * 80)
    print("1. ✅ Always detect query language")
    print("2. ✅ Use multilingual embedding models when possible")
    print("3. ✅ Implement query translation as fallback")
    print("4. ✅ Boost documents matching query language")
    print("5. ✅ Consider pre-translating important documents")
    print("6. ✅ Provide 'Translated from X' indicators to users")
    print(f"{'=' * 80}{Style.RESET_ALL}\n")


def main():
    """Run both failure and fixed examples."""
    print(f"\n{Fore.MAGENTA}{'=' * 80}")
    print("RAG FAILURE MODE #3: Multilingual Queries")
    print("=" * 80)
    print("\nThis example demonstrates cross-language retrieval failures when")
    print("query language doesn't match document language.")
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
    print("  ❌ No language detection")
    print("  ❌ No translation")
    print("  ❌ Poor cross-language retrieval")
    print("  ❌ Misses relevant content")
    print("\nFIXED:")
    print("  ✅ Automatic language detection")
    print("  ✅ Query translation")
    print("  ✅ Language-aware ranking")
    print("  ✅ Finds relevant content across languages")
    print(f"{'=' * 80}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
