"""
RAG Failure Mode #4: Citation Hallucinations
=============================================

Problem: LLM makes up sources, fabricates quotes, or misattributes information 
even when using RAG. The model cites non-existent documents or invents content
that sounds plausible but isn't in the retrieved context.

Common Issues:
- Fabricating document titles, authors, or dates
- Creating fake quotes that sound authoritative
- Mixing information from multiple sources without proper attribution
- Citing documents that don't exist in the knowledge base
- Misinterpreting or exaggerating claims from sources
"""

from langchain_core.documents import Document
from typing import List, Dict, Tuple
import re
import json


def load_medical_records():
    """Load medical records for citation examples."""
    with open('sample_data/medical_records.txt', 'r') as f:
        return f.read()


def create_mock_knowledge_base() -> List[Document]:
    """Create a mock knowledge base with medical documents."""
    doc_text = load_medical_records()
    
    # Split into sections
    sections = doc_text.split('\n---')
    
    docs = []
    for i, section in enumerate(sections):
        section = section.strip()
        if section:
            # Extract document type from section
            doc_type = "medical_record"
            if "X-RAY" in section:
                doc_type = "chest_xray"
            elif "LABORATORY" in section:
                doc_type = "lab_results"
            
            docs.append(Document(
                page_content=section,
                metadata={
                    "doc_id": f"DOC-{i+1:03d}",
                    "source": "medical_records.txt",
                    "type": doc_type,
                    "date": "2024-02-20"
                }
            ))
    
    return docs


def simulate_retrieval(query: str, knowledge_base: List[Document]) -> List[Document]:
    """Simulate document retrieval based on query keywords."""
    
    query_lower = query.lower()
    relevant_docs = []
    
    for doc in knowledge_base:
        # Simple keyword matching
        if any(keyword in doc.page_content.lower() for keyword in query_lower.split()):
            relevant_docs.append(doc)
    
    return relevant_docs[:3]  # Return top 3


def run_failure():
    """
    Demonstrates CITATION HALLUCINATION vulnerability.
    
    Problem: LLM generates responses with fake citations, invented quotes,
    or misattributed information.
    """
    print("=" * 80)
    print("FAILURE MODE: Citation Hallucinations")
    print("=" * 80)
    
    knowledge_base = create_mock_knowledge_base()
    
    print(f"\n📚 Knowledge Base: {len(knowledge_base)} documents")
    print("   - Medical records")
    print("   - X-ray reports")
    print("   - Lab results")
    
    # Naive prompt without citation enforcement
    naive_prompt = """Based on the following documents, answer the user's question.

Documents:
{context}

Question: {query}

Answer:"""
    
    print("\n❌ VULNERABLE SYSTEM:")
    print("   - No citation requirements")
    print("   - No fact verification")
    print("   - No source attribution enforcement")
    print("   - Trusts LLM to cite correctly")
    
    # Test Case 1: Fabricated citations
    print("\n" + "=" * 80)
    print("HALLUCINATION 1: Fabricated Citations")
    print("=" * 80)
    
    query = "What medications was the patient prescribed?"
    retrieved_docs = simulate_retrieval(query, knowledge_base)
    
    print(f"\n🔍 Query: '{query}'")
    print(f"   Retrieved {len(retrieved_docs)} documents")
    
    # Simulate LLM response WITH HALLUCINATIONS
    hallucinated_response = """According to Dr. Sarah Johnson's notes from February 15, 2024 (Document MED-2024-456), the patient was prescribed:

1. Prednisone 40mg daily for 5 days
2. Fluticasone/Salmeterol increased to twice daily
3. Azithromycin 500mg (as mentioned in the follow-up report by Dr. Michael Chen)
4. Montelukast 10mg nightly for asthma control [Source: Pulmonology Consult, Feb 18, 2024]

The patient also received a nebulizer treatment with ipratropium bromide in the emergency department, as documented in the ED report by Nurse Jennifer Williams."""
    
    print(f"\n❌ LLM Response (with hallucinations):")
    print(hallucinated_response)
    
    print("\n⚠️  PROBLEMS DETECTED:")
    print("   ❌ 'Dr. Sarah Johnson' - NOT in any document")
    print("   ❌ Document 'MED-2024-456' - DOES NOT EXIST")
    print("   ❌ 'Dr. Michael Chen' - FABRICATED")
    print("   ❌ 'Pulmonology Consult, Feb 18, 2024' - NOT IN KNOWLEDGE BASE")
    print("   ❌ 'Montelukast' - NOT PRESCRIBED (hallucinated medication)")
    print("   ❌ 'ipratropium bromide' - NOT MENTIONED")
    print("   ❌ 'Nurse Jennifer Williams' - DOES NOT EXIST")
    
    print("\n🔍 Actual Facts from Retrieved Documents:")
    print("   ✓ Prednisone 40mg - CORRECT")
    print("   ✓ Fluticasone/Salmeterol - CORRECT")
    print("   ✓ Azithromycin mentioned - BUT conditionally, not prescribed yet")
    print("   ✗ No doctor names provided")
    print("   ✗ No document IDs like 'MED-2024-456'")
    
    # Test Case 2: Mixing sources without attribution
    print("\n" + "=" * 80)
    print("HALLUCINATION 2: Misattribution & Mixed Sources")
    print("=" * 80)
    
    query2 = "What did the chest X-ray show?"
    
    print(f"\n🔍 Query: '{query2}'")
    
    # Simulate LLM mixing information
    mixed_response = """The chest X-ray performed on February 20, 2024 showed significant findings. According to the radiology report by Dr. Robert Martinez:

"Bilateral infiltrates consistent with pneumonia in the lower lobes. Moderate pleural effusion on the right side. Cardiomegaly noted with heart size at the upper limits of normal."

The radiologist recommended immediate antibiotic therapy and follow-up imaging in 48 hours. Comparison with previous X-ray from January 2024 showed progression of the infiltrates."""
    
    print(f"\n❌ LLM Response (with misattribution):")
    print(mixed_response)
    
    print("\n⚠️  PROBLEMS DETECTED:")
    print("   ❌ 'Dr. Robert Martinez' - FABRICATED radiologist name")
    print("   ❌ 'Bilateral infiltrates' - NOT in actual report")
    print("   ❌ 'Moderate pleural effusion' - MADE UP")
    print("   ❌ 'Cardiomegaly' - NOT MENTIONED")
    print("   ❌ Direct quote - COMPLETELY FABRICATED")
    print("   ❌ Previous X-ray comparison - DOES NOT EXIST")
    
    print("\n🔍 Actual X-ray Findings:")
    print("   ✓ Mild hyperinflation (consistent with obstructive lung disease)")
    print("   ✓ NO focal consolidation or effusion")
    print("   ✓ Heart: NORMAL size and contour")
    print("   ✓ No acute cardiopulmonary disease")
    
    # Test Case 3: Invented statistics
    print("\n" + "=" * 80)
    print("HALLUCINATION 3: Fabricated Statistics & Dates")
    print("=" * 80)
    
    query3 = "What were the patient's lab results?"
    
    print(f"\n🔍 Query: '{query3}'")
    
    fabricated_stats = """Laboratory results from February 20, 2024 (Lab Order #LAB-2024-7823):

Complete Blood Count:
- WBC: 12.5 K/uL (elevated, indicating infection) [Reference: Clinical Guidelines 2024]
- Hemoglobin: 13.2 g/dL (slightly low)
- Platelets: 189 K/uL (low-normal)

Inflammatory Markers:
- CRP: 45 mg/L (significantly elevated)
- ESR: 32 mm/hr (elevated)
- Procalcitonin: 0.8 ng/mL (elevated, suggests bacterial infection)

Per Dr. Anderson's interpretation note dated Feb 21, these results indicate acute bacterial infection requiring aggressive treatment."""
    
    print(f"\n❌ LLM Response (with fabrications):")
    print(fabricated_stats)
    
    print("\n⚠️  PROBLEMS DETECTED:")
    print("   ❌ Lab Order '#LAB-2024-7823' - DOES NOT EXIST")
    print("   ❌ WBC: 12.5 - WRONG (actual: 8.2)")
    print("   ❌ Hemoglobin: 13.2 - WRONG (actual: 14.5)")
    print("   ❌ Platelets: 189 - WRONG (actual: 245)")
    print("   ❌ CRP, ESR, Procalcitonin - NOT TESTED (completely made up)")
    print("   ❌ 'Dr. Anderson' - DOES NOT EXIST")
    print("   ❌ Interpretation note - FABRICATED")
    
    print("\n⚠️  Dangers of Citation Hallucinations:")
    print("   1. Medical misinformation - could harm patient care")
    print("   2. False confidence - citations make lies look credible")
    print("   3. Legal liability - wrong medical information is dangerous")
    print("   4. Impossible to trace - fake citations can't be verified")
    print("   5. Undermines trust - users can't rely on system")


def extract_citations_from_text(text: str) -> List[str]:
    """Extract citation patterns from text."""
    patterns = [
        r'\[Source:([^\]]+)\]',
        r'\(Document ([^\)]+)\)',
        r'according to ([^,\.]+)',
        r'as mentioned in ([^,\.]+)',
        r'per ([^,\.\'\"]+)\'s',
    ]
    
    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        citations.extend(matches)
    
    return [c.strip() for c in citations]


def verify_citations(response: str, source_docs: List[Document]) -> Dict:
    """Verify that all citations in response exist in source documents."""
    
    citations = extract_citations_from_text(response)
    
    # Create a text corpus from all source documents
    source_text = "\n".join([doc.page_content for doc in source_docs])
    source_ids = [doc.metadata.get('doc_id', '') for doc in source_docs]
    
    verification = {
        "total_citations": len(citations),
        "verified": [],
        "unverified": [],
        "fake_docs": [],
        "fake_people": []
    }
    
    for citation in citations:
        # Check if citation refers to a document ID
        if 'DOC-' in citation or 'MED-' in citation or 'LAB-' in citation:
            if citation in source_text or citation in source_ids:
                verification["verified"].append(citation)
            else:
                verification["fake_docs"].append(citation)
        
        # Check for person names (Dr., Nurse, etc.)
        elif any(title in citation for title in ['Dr.', 'Doctor', 'Nurse', 'Prof.']):
            if citation in source_text:
                verification["verified"].append(citation)
            else:
                verification["fake_people"].append(citation)
        
        else:
            # General verification
            if citation.lower() in source_text.lower():
                verification["verified"].append(citation)
            else:
                verification["unverified"].append(citation)
    
    return verification


def run_fixed():
    """
    Demonstrates CITATION-VERIFIED RAG system.
    
    Solution: Enforce structured citations, verify against source documents,
    and extract exact quotes with provenance.
    """
    print("\n" + "=" * 80)
    print("FIXED: Citation-Verified RAG System")
    print("=" * 80)
    
    knowledge_base = create_mock_knowledge_base()
    
    # Structured prompt that enforces citations
    citation_prompt = """You are a medical document assistant. You must follow these rules STRICTLY:

1. ONLY cite information that is explicitly stated in the provided documents
2. For EVERY claim, provide the document ID [DOC-XXX] where it appears
3. Use EXACT QUOTES in "quotation marks" when citing specific information
4. If information is not in the documents, say "Not found in provided documents"
5. NEVER invent document IDs, names, dates, or statistics
6. NEVER fabricate quotes or citations

Documents:
{context}

Question: {query}

Answer: Provide your response with [DOC-ID] citations after every claim."""
    
    print("\n✅ PROTECTED SYSTEM:")
    print("   - Enforces structured citations")
    print("   - Requires document IDs")
    print("   - Demands exact quotes")
    print("   - Verifies all citations")
    print("   - Rejects unfounded claims")
    
    # Test Case 1: Properly cited response
    print("\n" + "=" * 80)
    print("PROPER CITATION: Verifiable Response")
    print("=" * 80)
    
    query = "What medications was the patient prescribed?"
    retrieved_docs = simulate_retrieval(query, knowledge_base)
    
    print(f"\n🔍 Query: '{query}'")
    print(f"   Retrieved {len(retrieved_docs)} documents")
    
    # Simulate GOOD LLM response with proper citations
    proper_response = """Based on the medical record [DOC-001], the patient was prescribed:

1. "Prednisone 40mg daily for 5 days" [DOC-001]
2. "Increase fluticasone/salmeterol to twice daily dosing" [DOC-001]
3. "Continue albuterol as needed" [DOC-001]

The assessment also notes: "If X-ray positive, will start azithromycin (patient allergic to penicillin)" [DOC-001]

Note: Azithromycin was mentioned as a conditional treatment pending X-ray results, not as an active prescription at the time of this visit."""
    
    print(f"\n✅ LLM Response (with proper citations):")
    print(proper_response)
    
    # Verify citations
    verification = verify_citations(proper_response, retrieved_docs)
    
    print(f"\n✅ CITATION VERIFICATION:")
    print(f"   Total citations: {verification['total_citations']}")
    print(f"   ✓ Verified: {len(verification['verified'])}")
    print(f"   ✗ Unverified: {len(verification['unverified'])}")
    print(f"   ✗ Fake documents: {len(verification['fake_docs'])}")
    print(f"   ✗ Fake people: {len(verification['fake_people'])}")
    
    if verification['verified']:
        print(f"\n   Verified citations:")
        for cit in verification['verified'][:5]:
            print(f"      ✓ {cit}")
    
    # Test Case 2: Handling missing information
    print("\n" + "=" * 80)
    print("HONEST RESPONSE: Admitting Missing Information")
    print("=" * 80)
    
    query2 = "Who performed the chest X-ray?"
    
    print(f"\n🔍 Query: '{query2}'")
    
    honest_response = """Based on the provided documents [DOC-002], the chest X-ray report is available but the radiologist's name is not mentioned in the documents.

The X-ray report includes findings and impressions but does not contain attribution to a specific radiologist. [DOC-002]

If you need the radiologist's name, this information would need to be obtained from the original hospital records."""
    
    print(f"\n✅ LLM Response (honest):")
    print(honest_response)
    print("\n   ✓ Admits when information is not available")
    print("   ✓ Doesn't fabricate names or details")
    print("   ✓ Suggests alternative sources")
    
    # Test Case 3: Structured citation format
    print("\n" + "=" * 80)
    print("STRUCTURED FORMAT: Machine-Readable Citations")
    print("=" * 80)
    
    structured_response = {
        "answer": "The patient's vital signs on February 20, 2024 were within normal ranges with minor exceptions.",
        "citations": [
            {
                "claim": "Temperature: 98.6°F (37°C)",
                "source_doc_id": "DOC-001",
                "quote": "Temperature: 98.6°F (37°C)",
                "confidence": "exact_match"
            },
            {
                "claim": "Blood Pressure: 128/82 mmHg",
                "source_doc_id": "DOC-001",
                "quote": "Blood Pressure: 128/82 mmHg",
                "confidence": "exact_match"
            },
            {
                "claim": "Oxygen Saturation: 94% on room air",
                "source_doc_id": "DOC-001",
                "quote": "Oxygen Saturation: 94% on room air",
                "confidence": "exact_match"
            }
        ],
        "verified": True,
        "missing_info": []
    }
    
    print("\n✅ Structured Citation Format (JSON):")
    print(json.dumps(structured_response, indent=2))
    
    print("\n✅ Improvements:")
    print("   1. Every claim has a document ID")
    print("   2. Exact quotes prevent misrepresentation")
    print("   3. Machine-readable format enables verification")
    print("   4. Confidence scores indicate certainty")
    print("   5. Missing information explicitly tracked")
    print("   6. No fabricated citations possible")


def demonstrate_best_practices():
    """Show best practices for preventing citation hallucinations."""
    print("\n" + "=" * 80)
    print("BEST PRACTICES: Preventing Citation Hallucinations")
    print("=" * 80)
    
    print("""
🎯 STRATEGIES TO PREVENT CITATION HALLUCINATIONS:

1. STRUCTURED OUTPUT FORMATS:
   - Use JSON schema to enforce citation structure
   - Require document IDs for every claim
   - Mandate exact quotes in quotation marks
   - Separate claims from citations explicitly
   
   Example Schema:
   {
     "answer": "...",
     "citations": [
       {"claim": "...", "doc_id": "...", "quote": "...", "page": 5}
     ]
   }

2. PROMPT ENGINEERING:
   ✓ Explicitly forbid fabrication: "NEVER invent citations"
   ✓ Require evidence: "Every claim must cite a document ID"
   ✓ Demand quotes: "Use exact quotes in quotation marks"
   ✓ Provide examples of good citations
   ✓ Penalize hallucinations in few-shot examples

3. POST-PROCESSING VERIFICATION:
   ✓ Extract all citations from response
   ✓ Verify document IDs exist in knowledge base
   ✓ Check quotes match source documents
   ✓ Validate page numbers and sections
   ✓ Flag unverifiable claims
   ✓ Calculate citation accuracy score

4. RETRIEVAL AUGMENTATION:
   ✓ Include document metadata (ID, title, date, author)
   ✓ Add snippet provenance to each retrieved chunk
   ✓ Return relevance scores with documents
   ✓ Provide document hierarchy (section, page, paragraph)

5. UI/UX DESIGN:
   ✓ Display citations inline with claims
   ✓ Make citations clickable (link to source)
   ✓ Show confidence scores
   ✓ Highlight exact quotes differently
   ✓ Allow users to verify sources easily
   ✓ Flag unverified information

6. MODEL SELECTION & TUNING:
   ✓ Use models trained for faithful generation
   ✓ Fine-tune on citation-rich examples
   ✓ Use retrieval-augmented pretraining
   ✓ Apply RLHF for citation accuracy
   ✓ Test models on citation benchmarks

7. MONITORING & METRICS:
   ✓ Citation accuracy rate
   ✓ Hallucination detection rate
   ✓ User verification feedback
   ✓ Source document hit rate
   ✓ Quote exactness score
""")

    # Show example code
    print("\n📋 Example: Citation Verification System")
    print("=" * 40)
    
    verification_code = '''
class CitationVerifier:
    """Verify citations against source documents."""
    
    def __init__(self, knowledge_base: List[Document]):
        self.kb = knowledge_base
        self.kb_index = self._build_index()
    
    def verify_response(self, response: str, 
                       cited_docs: List[str]) -> Dict:
        """Verify all citations in response."""
        
        results = {
            "verified": [],
            "unverified": [],
            "fabricated": [],
            "accuracy_score": 0.0
        }
        
        # Extract citations
        citations = self.extract_citations(response)
        
        for citation in citations:
            # Check if document ID exists
            if not self.document_exists(citation['doc_id']):
                results["fabricated"].append(citation)
                continue
            
            # Check if quote matches source
            if self.verify_quote(citation['doc_id'], 
                                citation['quote']):
                results["verified"].append(citation)
            else:
                results["unverified"].append(citation)
        
        # Calculate accuracy
        total = len(citations)
        if total > 0:
            results["accuracy_score"] = (
                len(results["verified"]) / total
            )
        
        return results
    
    def document_exists(self, doc_id: str) -> bool:
        """Check if document ID exists in knowledge base."""
        return doc_id in self.kb_index
    
    def verify_quote(self, doc_id: str, quote: str) -> bool:
        """Verify quote appears in document."""
        doc = self.kb_index.get(doc_id)
        if not doc:
            return False
        
        # Use fuzzy matching for quotes
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(
            None, 
            quote.lower(), 
            doc.page_content.lower()
        ).ratio()
        
        return ratio > 0.85  # 85% similarity threshold

# Usage:
verifier = CitationVerifier(knowledge_base)
verification = verifier.verify_response(llm_response, cited_docs)

if verification["accuracy_score"] < 0.8:
    print("⚠️  Low citation accuracy - review response")
'''
    
    print(verification_code)


if __name__ == "__main__":
    print("\n🔥 RAG Failure Mode #4: Citation Hallucinations\n")
    
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
1. LLMs will hallucinate citations if not constrained
2. Structured outputs enforce citation discipline
3. Every claim must have verifiable provenance
4. Exact quotes prevent misrepresentation
5. Post-processing verification is essential
6. UI should make verification easy for users
7. Monitor citation accuracy in production

🎯 Citation Best Practices:
☑ Use structured output formats (JSON)
☑ Require document IDs for every claim
☑ Mandate exact quotes
☑ Verify all citations automatically
☑ Display sources prominently in UI
☑ Track citation accuracy metrics
☑ Allow users to verify sources

⚠️  Medical/Legal Applications:
- Citation hallucinations can be life-threatening
- False medical information can harm patients
- Legal misinformation has liability implications
- Always verify critical information manually
- Consider human-in-the-loop for high-stakes domains
""")
