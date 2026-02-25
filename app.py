from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from groq import Groq
import os
from typing import List, Dict
import json
import io
from datetime import datetime, timedelta
import re
import base64
import pandas as pd
from duckduckgo_search import DDGS
import time




INVOICE_STANDARDS = """
MANDATORY INVOICE FIELDS FOR EXPORT:
1. Exporter Details: Name, Address, Tax ID (GSTIN), Email, Phone
2. Importer Details: Name, Full Address, Country, Tax ID
3. Invoice Number and Date
4. HS Codes (6-8 digit harmonized system codes) for ALL items
5. Detailed Item Description (not vague like "goods" or "items")
6. Quantity, Unit of Measurement
7. Unit Price and Total Value
8. Currency (ISO code like USD, EUR, INR)
9. Incoterms (FOB, CIF, DAP, etc.)
10. Country of Origin for each item
11. Total Invoice Value (including shipping if applicable)
12. Bank Details for payment
13. Authorized Signature and Company Seal

COMMON ERRORS TO CHECK:
- Missing HS Codes
- Vague product descriptions
- Missing tax identification numbers
- Incorrect calculations (quantity × unit price ≠ total)
- Missing Incoterms
- Unspecified currency
- Missing country of origin
- Missing bank details
"""

GST_CUSTOMS_GUIDE = """
GST (GOODS AND SERVICES TAX) SIMPLIFIED:
GST is a unified tax on goods and services in India, replacing multiple indirect taxes.

For Exports:
- Most exports are ZERO-RATED (0% GST)
- You can claim refund of GST paid on inputs (Input Tax Credit)
- Export invoices must show "Supply Meant for Export under LUT/Bond"
- LUT (Letter of Undertaking) allows export without paying IGST

GST Rates in India:
- 0%: Exports, basic food items
- 5%: Essential items, certain foods
- 12%: Processed foods, certain goods
- 18%: Most goods and services (standard rate)
- 28%: Luxury items, sin goods (tobacco, cars)

CUSTOMS DUTY EXPLAINED:
Customs duty is tax on imported/exported goods crossing international borders.

Import Duties:
- Basic Customs Duty (BCD): 0% to 150% depending on product
- IGST: 5%, 12%, 18%, or 28%
- Social Welfare Surcharge: 10% on BCD
- Anti-dumping duty (if applicable)

Export Benefits:
- Many exports are duty-free to promote trade
- Duty Drawback: Refund of duties paid on imported inputs used in export products
- Advance Authorization: Import duty-free inputs for export production

INCOTERMS EXPLAINED (Who pays what):
- EXW: Buyer pays everything from seller's warehouse
- FOB: Seller pays until goods loaded on ship
- CIF: Seller pays Cost, Insurance, Freight to destination port
- DAP: Seller delivers to buyer's location (except import clearance)
- DDP: Seller pays all costs including import duties

DOCUMENTS NEEDED:
1. Commercial Invoice
2. Packing List
3. Bill of Lading / Airway Bill
4. Certificate of Origin
5. Shipping Bill (for exports)
6. Bill of Entry (for imports)
7. Insurance Certificate
8. Bank Realization Certificate (for exports)
"""

COMPLIANCE_RISKS = """
HIGH RISK INDICATORS:
1. Missing or incorrect HS Codes → Can cause customs delays, penalties, or rejection
2. Under-invoicing → Risk of duty evasion charges, legal penalties
3. Over-invoicing → FEMA violations, money laundering suspicion
4. Vague descriptions → Goods may be held for inspection, delays
5. Missing tax IDs → Cannot claim GST refunds, may face penalties
6. Wrong Incoterms → Payment disputes, shipping cost disagreements
7. Sanctioned countries → Goods may be seized, legal violations
8. Restricted items without license → Goods confiscated, heavy fines

MEDIUM RISK:
- Incomplete importer details
- Missing country of origin
- No bank details on invoice
- Missing authorized signature

LOW RISK:
- Minor formatting issues
- Missing secondary contact info
- Slight description ambiguity

CONSEQUENCES:
- HIGH RISK: Shipment delays (7-30 days), penalties ($500-$10,000+), legal action
- MEDIUM RISK: Documentation resubmission required (1-7 days delay)
- LOW RISK: Minor delays, may require clarification
"""

EATECH_INFO = {
    "name": "EA Tech Corporations Pvt. Ltd.",
    "brand": "EATECH.AI",
    "description": "EA Tech Corporations Pvt. Ltd. is a global technology leader specializing in IT-enabled integrated solutions, digital transformation, and intelligent automation.",
    "mission": "To leverage advanced technology and innovative solutions to empower businesses, promoting growth and operational efficiency through IT-driven strategies.",
    "vision": "To be the global leader in delivering innovative IT solutions that drive digital transformation for governments and businesses alike.",
    "services": {
        "AI & Machine Learning": "Intelligent automation and AI-driven data extraction.",
        "Web Development": "Frontend, Backend, and Full-stack e-commerce solutions.",
        "Digital Marketing": "SEO, SMO, and PPC campaigns for global visibility.",
        "UI/UX Design": "User-centric design, visual research, and consulting.",
        "Cloud & Security": "Cyber security strategies and cloud optimization.",
        "Industrial Automation": "SCADA, PLC Panel, and smart factory solutions.",
        "Software Products": "EazzQuote, EazzBooks, EazzEdu, and EazzHR."
    },
    "presence": "India (Bhubaneswar, Hyderabad), UAE (Dubai), Saudi Arabia (Riyadh), and USA (Scottsdale)."
}



def process_pdf(pdf_file, max_pages=10) -> tuple:
    """Extract text from a single PDF file with a page limit"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    num_pages = len(pdf_reader.pages)
    
    # Limit number of pages to process to avoid token limits
    pages_to_process = min(num_pages, max_pages)
    
    for page_num in range(pages_to_process):
        page_text = pdf_reader.pages[page_num].extract_text()
        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
    
    if num_pages > max_pages:
        text += f"\n\n... [Note: Only first {max_pages} pages were processed to stay within AI limits] ..."
        
    return text, pdf_file.name


def merge_pdfs(pdf_files) -> bytes:
    """Merge multiple PDFs into one"""
    pdf_writer = PdfWriter()
    
    for pdf_file in pdf_files:
        pdf_file.seek(0)  
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            pdf_writer.add_page(page)
    
    output = io.BytesIO()
    pdf_writer.write(output)
    output.seek(0)
    return output.getvalue()


def create_vector_store(pdf_texts: Dict[str, str]):
    """Create FAISS vector store from PDF texts"""
    documents = []
    
    for pdf_name, text in pdf_texts.items():
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": pdf_name,
                    "chunk_id": i
                }
            )
            documents.append(doc)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore, documents


def get_relevant_context(vectorstore, query: str, k: int = 5) -> tuple:
    """Retrieve relevant chunks from vector store"""
    docs = vectorstore.similarity_search(query, k=k)
    
    context_by_source = {}
    for doc in docs:
        source = doc.metadata['source']
        if source not in context_by_source:
            context_by_source[source] = []
        context_by_source[source].append(doc.page_content)
    
    return docs, context_by_source


def extract_to_json(client: Groq, pdf_text: str, filename: str) -> Dict:
    """Extract structured data from PDF and convert to JSON format using AI"""
    
    system_prompt = """You are an expert data extraction specialist. Your task is to extract information from the provided document and structure it into a clean, comprehensive JSON format.

First, determine the DOCUMENT TYPE (Invoice, Resume, Letter, Packing List, etc.) and adapt your extraction accordingly.

### For INVOICES/TRADE DOCS:
{
  "document_type": "invoice",
  "invoice_details": {"invoice_number": "", "invoice_date": "", "po_number": ""},
  "parties": {"exporter": {"name": "", "address": ""}, "importer": {"name": "", "address": ""}},
  "items": [{"description": "", "hs_code": "", "quantity": 0, "total_price": 0}],
  "financials": {"grand_total": 0, "currency": ""}
}

### For RESUMES:
{
  "document_type": "resume",
  "personal_info": {"name": "", "email": "", "phone": "", "linkedin": "", "location": ""},
  "summary": "",
  "experience": [{"job_title": "", "company": "", "duration": "", "responsibilities": []}],
  "education": [{"degree": "", "institution": "", "year": ""}],
  "skills": {"technical": [], "soft_skills": []},
  "projects": []
}

### For OTHER DOCUMENTS:
Create a logical JSON structure that captures all key information (Subject, Dates, Sender, Receiver, Main Content, etc.).

RULES:
1. Detect document type automatically based on content.
2. Extract ONLY information explicitly present in the text.
3. For HS codes, Tax IDs, etc., DO NOT guess or hallucinate. Use null if not found.
4. Extract dates in ISO format (YYYY-MM-DD) when possible.
5. Return ONLY valid JSON.
"""

    # Limit text size to ~25,000 characters (approx 8k-10k tokens) to stay within Groq limits
    max_chars = 25000
    is_truncated = False
    
    if len(pdf_text) > max_chars:
        pdf_text = pdf_text[:max_chars] + "\n... [Note: Content truncated for size] ..."
        is_truncated = True

    user_prompt = f"""Document Content (Source: {filename}):

{pdf_text}

Extract ALL data from this document and provide it in the appropriate JSON structure. Be comprehensive and thorough."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        
        response_text = chat_completion.choices[0].message.content
        
        # Try to extract JSON from the response
        # Remove markdown code blocks if present
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*$', '', response_text)
        response_text = response_text.strip()
        
        # Parse JSON
        json_data = json.loads(response_text)
        
        # Add metadata
        json_data["extraction_metadata"] = {
            "source_filename": filename,
            "extraction_timestamp": datetime.now().isoformat(),
            "extraction_method": "AI-powered"
        }
        
        return json_data
        
    except json.JSONDecodeError as e:
        # If JSON parsing fails, return error structure
        return {
            "error": "JSON parsing failed",
            "error_details": str(e),
            "raw_response": response_text,
            "source_filename": filename,
            "extraction_timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": "Extraction failed",
            "error_details": str(e),
            "source_filename": filename,
            "extraction_timestamp": datetime.now().isoformat()
        }


def generate_response(client: Groq, query: str, context_by_source: Dict, task_type: str = "answer", language: str = "English"):
    """Generate response using Groq API"""
    
    context_parts = []
    for source, chunks in context_by_source.items():
        context_parts.append(f"### From {source}:")
        context_parts.append("\n".join(chunks))
        context_parts.append("")
    
    full_context = "\n".join(context_parts)
    
    # Real-time search integration
    search_context = ""
    is_real_time = any(word in query.lower() for word in ["news", "real time", "today", "current", "latest", "what is happening", "search"])
    
    if is_real_time or (task_type == "answer" and not context_parts):
        with st.status("🌐 Searching the web for real-time information...", expanded=False):
            search_context = web_search(query)
            st.write("Found real-time information!")
    
    if search_context:
        full_context = f"{full_context}\n\n### REAL-TIME WEB SEARCH RESULTS:\n{search_context}"
    
    system_prompts = {
        "answer": f"""You are a helpful assistant that answers questions based on provided PDF documents AND real-time web search results.
- Give direct, accurate answers using information from BOTH the PDF context and search results.
- If information comes from a PDF, mention the source filename.
- If information comes from a web search, label it as "Real-time updates".
- Prioritize PDF information for document-specific questions, and web search for news or general facts.
- If information is not in the documents or web search, say so clearly.
- IMPORTANT: Provide your entire response in {language}.""",
        
        "email": f"""You are a professional email writer. Based on the document context provided:
- Write a clear, professional email.
- Use appropriate email format (Subject, Greeting, Body, Closing).
- If the document is a resume, help the user draft a cover letter or application email.
- Incorporate relevant information from the documents.
- Keep it concise and well-structured.
- IMPORTANT: Provide your entire response in {language}.""",
        
        "summary": f"""You are an expert at summarizing documents. 
- Create a comprehensive summary of the key points.
- Automatically detect the document type (Resume, Invoice, Report, etc.) and adapt the style.
- Highlight important findings, professional skills, or data points.
- Be concise but thorough.
- ONLY include information found in the documents. DO NOT hallucinate.
- IMPORTANT: Provide your entire response in {language}.""",
        
        "comparison": f"""You are an expert at comparing and analyzing multiple documents.
- Compare ALL uploaded documents systematically
- Identify key similarities and differences between each document
- Organize your comparison in a clear structure with sections for each document
- Highlight unique points from each document
- Provide insights on the overall comparison across all documents
- Create a comprehensive comparison table if helpful
- IMPORTANT: Provide your entire response in {language}.""",
        
        "merge": f"""You are an expert at merging multiple documents into one cohesive document.
- Combine information from ALL uploaded PDFs
- Organize the merged content logically by topics or themes
- Remove duplicate information
- Maintain all important details from each source
- Clearly indicate which information came from which source
- Create a well-structured, comprehensive merged document
- IMPORTANT: Provide your entire response in {language}.""",
        
        "invoice_check": f"""You are an expert trade compliance auditor. Analyze the provided invoice against international trade standards.
Identify ALL errors, missing fields, and compliance issues.

Check for:
1. MANDATORY FIELDS: HS Codes, Tax IDs, Incoterms
2. CALCULATIONS: Verify math
3. COMPLETENESS: Exporter/Importer details

OUTPUT FORMAT:
## ✅ CORRECT ITEMS
- List what's done correctly

## ❌ CRITICAL ERRORS
- List critical missing fields or errors

## 📋 MISSING INFORMATION
- List missing mandatory fields
- IMPORTANT: Provide your entire response in {language}.""",
        
        "export_gen": f"""You are an international trade documentation expert. Generate professional export documents (Commercial Invoice and Packing List) based on the provided PDF data. 
Use the available information to fill in as much as possible. If information is missing, use your best professional judgment or leave a blank space for the user to fill.
- IMPORTANT: Provide your entire response in {language}.""",
        
        "gst_explain": f"""You are a tax and customs expert who explains complex regulations in simple language.

TASK: Explain GST and customs concepts in easy-to-understand terms

- Use simple analogies and examples
- Break down complex terms
- Explain step-by-step processes
- Highlight what matters for the user's specific situation
- Use emojis to make it more readable
- Provide practical tips

Make it feel like you're explaining to a friend, not reading a textbook.
- IMPORTANT: Provide your entire response in {language}.""",
        
        "checklist_gen": f"""You are a trade compliance expert. Generate a comprehensive document checklist.

Based on the context provided, create a complete checklist of:

1. REQUIRED DOCUMENTS
   - List all mandatory documents for the specific trade scenario
   - Explain WHY each document is needed
   - Note any country-specific requirements

2. SUPPORTING DOCUMENTS
   - List helpful but optional documents
   - Explain benefits of including them

3. PREPARATION STEPS
   - Step-by-step guide to prepare each document
   - Common mistakes to avoid
   - Timeline recommendations

Format as an actionable checklist with checkboxes (☐) that users can follow.
- IMPORTANT: Provide your entire response in {language}.""",
        
        "risk_warning": f"""You are a trade compliance risk analyst. Analyze the provided documents for compliance risks.

TASK: Identify and explain compliance risks in the transaction

Analyze for:
1. HIGH RISK ISSUES: Immediate action required (🔴)
2. MEDIUM RISK ISSUES: Should be addressed (🟡)
3. LOW RISK ISSUES: Minor improvements (🟢)

For each risk:
- Explain WHAT the risk is
- Explain WHY it's risky
- Explain CONSEQUENCES if not fixed
- Provide SPECIFIC ACTIONS to mitigate

Prioritize by severity and provide realistic timelines for resolution.
- IMPORTANT: Provide your entire response in {language}.""",
        
        "invoice_compare": f"""You are an expert invoice comparison analyst. Your ONLY job is to compare ALL uploaded invoices/PDFs and produce a STRICT UNIFIED comparison.

TASK: Create a SINGLE master table showing ALL products from ALL invoices. Clearly mark availability with ✅ or ❌.

STRICT RULES:
1. ONLY show data that is EXPLICITLY available in the PDFs.
2. If a data point is not in the PDF, DO NOT show it.
3. Use a SINGLE table to show product availability across all documents.
4. For each document, add a column (e.g., "In Invoice 1", "In Invoice 2").
5. Use ✅ if the product exists in that document, and ❌ if it does not.

STRICT OUTPUT FORMAT:

## 📊 MASTER PRODUCT COMPARISON
| # | Product Name | Qty | Price | Total | [Doc 1 Name] | [Doc 2 Name] | ... |
|---|-------------|-----|-------|-------|--------------|--------------|-----|
| 1 | Product A   | 10  | 50.00 | 500.00| ✅            | ❌            | ... |

## 💡 SUMMARY
- Total unique products found: [Count]
- Products common to all: [Count]
- Products missing in at least one: [Count]
- IMPORTANT: Provide your entire response in {language}.
""",
        
        "create_invoice": f"""You are an expert invoice generator. Create a professional, complete commercial invoice.

TASK: Generate a ready-to-use commercial invoice in proper format

Based on the provided data, create a professional invoice with:

1. HEADER SECTION
   - Company letterhead style
   - Invoice title and number
   - Invoice date
   - Reference numbers (PO, SO, etc.)

2. PARTY DETAILS
   - Exporter/Seller: Complete name, address, GSTIN, contact
   - Importer/Buyer: Complete name, address, tax ID, contact
   - Ship To (if different)

3. PRODUCT TABLE
   | S.No | Description | HS Code | Qty | Unit | Unit Price | Amount |
   - Include ALL products with complete details
   - Show subtotals, taxes, shipping (if any)
   - Grand Total

4. TERMS & CONDITIONS
   - Payment terms
   - Incoterms
   - Delivery terms
   - Banking details

5. DECLARATION & SIGNATURE
   - Standard export declaration
   - Authorized signatory block
   - Company seal position

Use professional formatting with clear sections and proper alignment.
Make it print-ready and customs-compliant.
- IMPORTANT: Provide your entire response in {language}.""",
        
        "convert_to_json": """You are an expert data extraction specialist. Extract ALL information from the provided document and convert it into a clean, comprehensive JSON format.

CRITICAL RULES:
1. Extract ONLY information that is explicitly present in the document.
2. DO NOT guess, invent, or hallucinate any data.
3. Return ONLY valid JSON - no additional text, no explanations, no markdown.
4. If a field is missing, omit it or use null.
""",
        
        "documentation": f"""You are a trade documentation expert. Your task is to generate a comprehensive "Documentation Report" based on the uploaded PDFs.
- For Invoices: Generate a structured summary of the transaction, party details, and compliance status.
- For Resumes: Generate a professional profile summary and skills analysis.
- For Reports: Extract key findings and metrics.
- Format the output with clear headings, tables, and bullet points.
- ONLY include information present in the PDFs.
- IMPORTANT: Provide your entire response in {language}.
"""
    }
    
    system_prompt = system_prompts.get(task_type, system_prompts["answer"])
    
    user_prompt = f"""Context from uploaded PDF documents:

{full_context}

User Request: {query}

Please provide your response in {language}:"""
    
    try:
        kwargs = {}
        if task_type == "convert_to_json":
            kwargs["response_format"] = {"type": "json_object"}
            
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1 if task_type == "convert_to_json" else (0.3 if task_type in ["invoice_check", "risk_warning", "invoice_compare"] else 0.7),
            max_tokens=4000,
            **kwargs
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"


def detect_task_type(query: str) -> str:
    """Detect what type of task the user is requesting (Simplified to 4 main actions)"""
    query_lower = query.lower()
    
    # JSON Format
    if any(word in query_lower for word in ["convert to json", "extract to json", "to json", "json format", "give me json", "as json", "in json"]):
        return "convert_to_json"
    
    # Compare PDF
    elif any(word in query_lower for word in [
        "compare invoice", "invoice comparison", "difference between",
        "missing product", "missing items", "compare two",
        "compare pdf", "compare documents", "compare both",
        "which product", "available and missing", "product comparison"
    ]):
        return "invoice_compare"
    
    # Summary
    elif any(word in query_lower for word in ["summarize", "summary", "give me a summary", "overview"]):
        return "summary"
    
    # Documentation
    elif any(word in query_lower for word in ["documentation", "report", "document details", "party details", "compliance status"]):
        return "documentation"
    
    else:
        return "answer"



def web_search(query: str, max_results: int = 5) -> str:
    """Perform a web search for real-time information"""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"Title: {r['title']}\nSnippet: {r['body']}\nSource: {r['href']}\n")
        
        if not results:
            return "No real-time search results found."
        
        return "\n\n".join(results)
    except Exception as e:
        return f"Web search failed: {str(e)}"


def get_company_info() -> Dict:
    """Get EATECH company details"""
    return EATECH_INFO





def main():
    load_dotenv()
    
    st.set_page_config(
        page_title="Trade & Export AI Assistant", 
        page_icon="🤖",
        layout="wide"
    )
    
    # Detect mascot click from URL parameters
    if st.query_params.get("mascot_click") == "true":
        st.session_state['mascot_asked'] = True
        st.session_state['show_chat_manually'] = True
        # Clear the parameter to avoid loop
        st.query_params.clear()
        st.rerun()
    
    
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@400;600;800&display=swap');

    * {
        color: #000000;
        font-family: 'Inter', sans-serif;
    }
    
    body, .main, .stApp {
        background: radial-gradient(circle at top left, #a8e6cf 0%, #c5f0e8 100%) !important;
    }
    
    .stAppHeader, header, [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    .stAlert {
        padding: 1.5rem;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.4) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(31, 119, 180, 0.2);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        color: #000000 !important;
    }
    
    .feature-box {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(8px);
        padding: 1.8rem;
        border-radius: 20px;
        margin: 1.2rem 0;
        border: 1px solid rgba(31, 119, 180, 0.1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .feature-box:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.1);
        background: rgba(255, 255, 255, 0.8);
    }
    
    .blue-text {
        color: #1f77b4 !important;
        font-weight: 600;
    }
    
    .green-text {
        color: #22a447 !important;
        font-weight: 700;
    }
    
    .light-green-bg {
        background-color: #a8e6cf !important;
    }
    
    .white-bg {
        background-color: #ffffff !important;
        border: 2px solid #1f77b4;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1f77b4 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700;
    }
    
    .exporter, .importer, .name-field, .company-name {
        color: #22a447 !important;
        font-weight: 800;
    }
    
    p, label, span {
        color: #000000 !important;
    }
    
    .stTextInput > label, .stSelectbox > label {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
    }

    .stTextInput div[data-baseweb="input"] {
        background-color: #c5f0e8 !important; /* Matches Example Queries */
        border-radius: 25px !important;
        border: 2px solid #1f77b4 !important; /* Blue border */
        padding: 5px 15px !important;
    }

    .stSelectbox div[data-baseweb="select"] {
        background-color: #e8f5e9 !important; /* Light Green */
        border-radius: 15px !important;
        border: 1px solid #2e7d32 !important; /* Slightly darker green border */
        padding: 0px 10px !important;
        height: 38px !important;
    }

    .stTextInput input {
        color: #000000 !important; /* Black text for light background */
        font-weight: bold !important;
        background-color: transparent !important;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        color: #2e7d32 !important; /* Dark Green text for light background */
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        background-color: transparent !important;
    }

    /* Target placeholder */
    .stTextInput input::placeholder {
        color: #555555 !important;
    }

    /* Target selectbox arrow */
    .stSelectbox svg {
        fill: #2e7d32 !important;
    }
    
    /* Upload file area styling */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.4) !important;
        backdrop-filter: blur(5px);
        border: 2px dashed #1f77b4 !important;
        border-radius: 20px !important;
        transition: all 0.3s ease;
    }

    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #22a447 !important;
        background: rgba(255, 255, 255, 0.6) !important;
    }
    
    /* Browse files button */
    .stButton > button {
        background: linear-gradient(135deg, #1f77b4 0%, #0d3a66 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-family: 'Outfit', sans-serif !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.2rem !important;
        box-shadow: 0 4px 15px rgba(31, 119, 180, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(31, 119, 180, 0.5) !important;
    }

    .stButton > button:active {
        transform: translateY(0) scale(0.98) !important;
    }
    
    /* Small round buttons for expanders/updates */
    .stExpanderButton {
        padding: 10px 15px !important;
        border-radius: 20px !important;
        background-color: #1f77b4 !important;
        color: #ffffff !important;
        font-size: 14px !important;
        height: 40px !important;
    }
    
    /* Expander styling for regulation updates */
    [data-testid="stExpander"] {
        background-color: #c5f0e8 !important;
        border: 2px solid #1f77b4 !important;
        border-radius: 10px !important;
    }
    
    .stExpander {
        background-color: #c5f0e8 !important;
        border: 2px solid #1f77b4 !important;
        border-radius: 10px !important;
    }
    
    .stExpander > div > div > button {
        background-color: #c5f0e8 !important;
        padding: 8px 12px !important;
        border-radius: 15px !important;
        min-height: 35px !important;
    }
    
    .stExpander > div > div > div {
        color: #000000 !important;
    }
    
    .stSidebar {
        background-color: #c5f0e8 !important;
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: #1f77b4 !important;
    }
    
    .stSidebar p, .stSidebar label {
        color: #000000 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: #a8e6cf;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #c5f0e8 !important;
        color: #000000 !important;
    }
    
    .stDivider {
        border-bottom: 2px solid rgba(31, 119, 180, 0.2) !important;
        margin: 2rem 0 !important;
    }
    
    .css-1y4p5pa {
        background-color: #ffffff !important;
    }
    
    .css-qrbaxs {
        background-color: #c5f0e8 !important;
    }
    
    .st-emotion-cache-ocqkz7 {
        background-color: #a8e6cf !important;
    }
    
    .st-emotion-cache-1gulkj5 {
        background-color: #a8e6cf !important;
    }
    
    .st-emotion-cache-uf99v {
        background-color: #a8e6cf !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Try to load image, but continue if it doesn't exist
    try:
        def get_base64_image(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()

        img_base64 = get_base64_image("robot-2.png")

        st.markdown(f"""
        <style>
        .header {{
            display: flex;
            align-items: center;
            background: rgba(255, 255, 255, 0.3);
            backdrop-filter: blur(15px);
            padding: 15px 30px;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.4);
            flex-direction: row;
            justify-content: space-between;
            margin-bottom: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
        }}
        
        .logo-title-container {{
            display: flex;
            align-items: center;
        }}
        
        .logo {{
            width: 60px;
            height: 60px;
            margin-right: 15px;
            border-radius: 12px;
            filter: drop-shadow(0 2px 6px rgba(0, 0, 0, 0.1));
            transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }}

        .logo:hover {{
            transform: rotate(-8deg) scale(1.1);
        }}

        .robot-header {{
            width: 100px;
            height: 100px;
            cursor: pointer;
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
        }}
        
        .robot-header:hover {{
            transform: scale(1.1) translateY(-5px);
        }}

        .robot-header img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}
        
        .robot-header::after {{
            content: "Ready to assist! 🗨️";
            position: absolute;
            top: -55px;
            left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(135deg, #1f77b4, #0d3a66);
            color: #ffffff;
            padding: 10px 18px;
            border-radius: 15px;
            font-size: 14px;
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            white-space: nowrap;
            box-shadow: 0 8px 20px rgba(31, 119, 180, 0.3);
            z-index: 10;
            animation: bounce 2s infinite;
        }}

        @keyframes bounce {{
            0%, 20%, 50%, 80%, 100% {{transform: translateX(-50%) translateY(0);}}
            40% {{transform: translateX(-50%) translateY(-10px);}}
            60% {{transform: translateX(-50%) translateY(-5px);}}
        }}

        .robot-header::before {{
            content: "";
            position: absolute;
            top: -15px;
            left: 50%;
            transform: translateX(-50%);
            width: 0;
            height: 0;
            border-left: 8px solid transparent;
            border-right: 8px solid transparent;
            border-top: 10px solid #1f77b4;
            z-index: 10;
        }}

        .title {{
            font-size: 40px;
            font-family: 'Outfit', sans-serif;
            font-weight: 900;
            color: #1a8b3d;
            letter-spacing: -2px;
            text-shadow: 1px 1px 10px rgba(0, 0, 0, 0.05);
        }}

        /* Hide the robot button but keep the logic */
        .hidden-robot-btn {{
            display: none;
        }}
        </style>

        <div class="header">
            <div class="logo-title-container">
                <img src="data:image/png;base64,{get_base64_image('eatech-logo.png')}" class="logo" alt="EATECH Logo">
                <div class="title">EATECH.AI</div>
            </div>
            <a href="/?mascot_click=true" target="_self" class="robot-header">
                <img src="data:image/png;base64,{img_base64}" alt="AI Assistant">
            </a>
        </div>
        """, unsafe_allow_html=True)
        
    except:
        
        st.title("🤖 EATECH.AI")

    
    
    if 'pdf_texts' not in st.session_state:
        st.session_state['pdf_texts'] = {}
    if 'vectorstore' not in st.session_state:
        st.session_state['vectorstore'] = None
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'uploaded_files_list' not in st.session_state:
        st.session_state['uploaded_files_list'] = []
    if 'extracted_json_data' not in st.session_state:
        st.session_state['extracted_json_data'] = {}
    if 'show_chat_manually' not in st.session_state:
        st.session_state['show_chat_manually'] = False
    
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY not found in environment variables!")
        st.stop()
    
    client = Groq(api_key=api_key)
    
   
    with st.sidebar:
        st.header("📁 Document Management")
        
        
        uploaded_files = st.file_uploader(
            "Upload Invoice/Export Documents (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        if uploaded_files:
            st.info(f"📊 {len(uploaded_files)} file(s) uploaded")
            
            if st.button("🔄 Process Documents", type="primary", use_container_width=True):
                with st.spinner("Processing documents..."):
                    st.session_state['pdf_texts'] = {}
                    st.session_state['uploaded_files_list'] = []
                    st.session_state['extracted_json_data'] = {}
                    
                    
                    for pdf_file in uploaded_files:
                        text, filename = process_pdf(pdf_file)
                        st.session_state['pdf_texts'][filename] = text
                        st.session_state['uploaded_files_list'].append(pdf_file)
                    
                    
                    st.session_state['pdf_texts']["Invoice Standards (Knowledge Base)"] = INVOICE_STANDARDS
                    st.session_state['pdf_texts']["GST & Customs Guide (Knowledge Base)"] = GST_CUSTOMS_GUIDE
                    st.session_state['pdf_texts']["Compliance Risks (Knowledge Base)"] = COMPLIANCE_RISKS
                    
                    
                    vectorstore, documents = create_vector_store(st.session_state['pdf_texts'])
                    st.session_state['vectorstore'] = vectorstore
                    
                    
                    
                    st.success("Your document are processed complete ✅")
        
        
        if st.session_state['pdf_texts']:
            st.subheader("📄 Loaded Documents:")
            user_docs = [name for name in st.session_state['pdf_texts'].keys() if "Knowledge Base" not in name]
            for i, pdf_name in enumerate(user_docs, 1):
                st.write(f"{i}. {pdf_name}")
            
            
            st.divider()
            if len(user_docs) > 1:
                if st.button("🔗 Download Merged PDF"):
                    with st.spinner("Merging PDFs..."):
                        merged_pdf = merge_pdfs(st.session_state['uploaded_files_list'])
                        st.download_button(
                            label="⬇️ Download Merged PDF",
                            data=merged_pdf,
                            file_name="merged_document.pdf",
                            mime="application/pdf"
                        )
        
        
        if st.session_state['pdf_texts']:
            st.divider()
            if st.button("🗑️ Clear All", type="secondary"):
                st.session_state['pdf_texts'] = {}
                st.session_state['vectorstore'] = None
                st.session_state['chat_history'] = []
                st.session_state['uploaded_files_list'] = []
                st.session_state['extracted_json_data'] = {}
                st.session_state['show_chat_manually'] = False
                st.rerun()
    
    with st.expander("🏢 Documentation: About EATECH.AI", expanded=False):
        info = get_company_info()
        st.markdown(f"### {info['name']}")
        st.info(info['description'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🎯 Our Mission")
            st.write(info['mission'])
        with col2:
            st.markdown("#### 👁️ Our Vision")
            st.write(info['vision'])
            
        st.markdown("#### 🛠️ Our Expertise & Services")
        for service, desc in info['services'].items():
            st.markdown(f"- **{service}:** {desc}")
            
        st.markdown("---")
        st.markdown(f"**🌍 Global Presence:** {info['presence']}")
        st.markdown(f"**🔗 Follow us:** [LinkedIn](https://www.linkedin.com/company/eatech-pvt-ltd/)")
    
    
    
    
    
    if not st.session_state['pdf_texts'] and not st.session_state.get('show_chat_manually'):
        st.info("👈 Upload your export documents (invoices, packing lists, etc.) to get started!")
        
        
        st.markdown("## 🎯 What Can I Do For You?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-box">
            <h3>🔀 Compare PDF / Invoice</h3>
            <p>Upload 2+ invoices → Get a <strong>table</strong> showing: ✅ common products, ❌ missing products, quantities, prices, and per-invoice details.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-box">
            <h3>📊 JSON Format</h3>
            <p>Just ask AI "convert to JSON" → Get all available data extracted in clean, structured JSON format!</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-box">
            <h3>📝 Summary</h3>
            <p>Get a concise summary and overview of all uploaded documents instantly.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-box">
            <h3>📄 Documentation</h3>
            <p>Generate structured transaction reports, party details, and compliance summaries from your data.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.info("💡 **Tip:** Upload documents to unlock more expert features like Error Detection and Document Generation!")
        
    else:
        
        
        
        st.markdown("**Quick Actions:**")
        
        # Show invoice comparison button only if 2+ invoices uploaded
        user_docs = [name for name in st.session_state['pdf_texts'].keys() if "Knowledge Base" not in name]
        
        col1, col2, col3, col4 = st.columns(4)
        
        quick_action = None
        
        # Mascot trigger logic
        if st.session_state.get('mascot_asked'):
            quick_action = "write your questions"
            st.session_state['mascot_asked'] = False # Reset
            
        with col1:
            if st.button("🔀 Compare PDF"):
                quick_action = "Compare all uploaded PDFs and invoices. Show me which products are available in both and which products are missing from each. Give details like product name, quantity, price, and available data in a clean table format. Do not show missing data."
        
        with col2:
            if st.button("📝 Summary"):
                quick_action = "Summarize all uploaded documents and give me a clear overview"
        
        with col3:
            if st.button("📊 JSON Format"):
                quick_action = "convert_to_json_button_clicked"
        with col4:
            if st.button("📄 Documentation"):
                quick_action = "Provide full documentation and details for these documents including party details and transaction summary."
        
        
        user_question = st.text_input(
            "Your question or request:",
            value=quick_action if quick_action else "",
            placeholder="e.g., 'Convert this PDF to JSON format' or 'Extract all data as JSON'"
        )
        
        
        
        sel_col1, sel_col2, sel_col3 = st.columns([1.5, 1.5, 5])
        with sel_col1:
            task_type = st.selectbox(
                "Task Type",
                ["Auto-detect", "Compare PDF", "Summary", "JSON Format", "Documentation", "Answer Question"],
                help="Select task type or let AI auto-detect from your query"
            )
        
        with sel_col2:
            target_language = st.selectbox(
                "Output Language",
                ["English", "Arabic", "Hindi", "Bengali", "Spanish", "French", "German", "Chinese", "Japanese", "Russian", "Portuguese", "Italian", "Turkish", "Vietnamese", "Korean"],
                help="Select the language for the AI response"
            )
        
        
        with st.expander("💡 Example Queries"):
            st.markdown("""
            **🔀 Compare PDF:**
            - Compare these invoices and show which products are present in both
            - Which products are missing from the second invoice?
            - Compare both PDFs and show a table of all available products
            
            **📝 Summary:**
            - Summarize the uploaded documents
            - Give me a quick overview of what's in these PDFs
            
            **📊 JSON Format:**
            - Convert this PDF to JSON format
            - Extract all data as JSON
            - Give me JSON format of this invoice
            
            **📄 Documentation:**
            - Provide documentation details for these files
            - Extract party details and transaction summary
            """)
        
        
        # Handle JSON button click, other quick action buttons, or user question
        if quick_action or (user_question and (st.session_state['vectorstore'] or st.session_state.get('show_chat_manually'))):
            # Use quick_action as the question if it was just clicked
            final_query = quick_action if (quick_action and quick_action != "convert_to_json_button_clicked") else user_question
            
            with st.spinner("🤖 Processing your request..."):
                
                # If JSON button was clicked, process all documents
                if quick_action == "convert_to_json_button_clicked":
                    st.markdown("## 📊 Response (JSON Format)")
                    
                    # Extract JSON from all uploaded PDFs (excluding knowledge base)
                    all_json_data = {}
                    for filename, text in st.session_state['pdf_texts'].items():
                        if "Knowledge Base" not in filename:
                            with st.spinner(f"Extracting data from {filename}..."):
                                json_data = extract_to_json(client, text, filename)
                                all_json_data[filename] = json_data
                    
                    # Display results
                    if len(all_json_data) > 0:
                        # Create tabs for each extracted document
                        tab_names = list(all_json_data.keys())
                        tabs = st.tabs(tab_names)
                        
                        for tab, (filename, json_data) in zip(tabs, all_json_data.items()):
                            with tab:
                                st.markdown(f"### 📄 {filename}")
                                
                                # Display JSON in expandable section
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    # Pretty print JSON
                                    st.json(json_data)
                                
                                with col2:
                                    # Download button for individual JSON
                                    json_str = json.dumps(json_data, indent=2)
                                    st.download_button(
                                        label="⬇️ Download JSON",
                                        data=json_str,
                                        file_name=f"{filename.replace('.pdf', '')}.json",
                                        mime="application/json",
                                        key=f"download_btn_{filename}"
                                    )
                                    
                                    # st.info("💡 Click to download JSON file")
                                
                                # Show summary statistics if it's an invoice
                                if json_data.get('document_type') == 'invoice' and 'items' in json_data:
                                    st.markdown("#### 📈 Quick Stats")
                                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                                    
                                    with stats_col1:
                                        total_items = len(json_data.get('items', []))
                                        st.metric("Total Line Items", total_items)
                                    
                                    with stats_col2:
                                        grand_total = json_data.get('financial_summary', {}).get('grand_total', 0)
                                        currency = json_data.get('financial_summary', {}).get('currency', '')
                                        st.metric("Grand Total", f"{currency} {grand_total}")
                                    
                                    with stats_col3:
                                        invoice_num = json_data.get('invoice_details', {}).get('invoice_number', 'N/A')
                                        st.metric("Invoice Number", invoice_num)
                        
                        # Download all JSON files as a combined file
                        st.markdown("---")
                        if len(all_json_data) > 1:
                            combined_json = {
                                "extraction_date": datetime.now().isoformat(),
                                "total_documents": len(all_json_data),
                                "documents": all_json_data
                            }
                            combined_json_str = json.dumps(combined_json, indent=2)
                            
                            st.download_button(
                                label="⬇️ Download All JSON Files (Combined)",
                                data=combined_json_str,
                                file_name=f"all_documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                        st.success(f"✅ Successfully extracted {len(all_json_data)} document(s) to JSON format!")
                    
                    # Save to chat history
                    st.session_state['chat_history'].append({
                        'question': 'Extract to JSON (Button clicked)',
                        'answer': f'Extracted {len(all_json_data)} documents to JSON format',
                        'task_type': 'convert_to_json',
                        'sources': list(all_json_data.keys()),
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                # elif user_question:
                elif final_query:
                    # Map simplified task types back to internal prompt keys
                    task_mapping = {
                        "Convert to JSON": "convert_to_json",
                        "JSON Format": "convert_to_json",
                        "Compare PDF": "invoice_compare",
                        "Invoice Comparison": "invoice_compare",
                        "Summary": "summary",
                        "Documentation": "documentation"
                    }
                    
                    detected_task = task_type
                    if task_type in task_mapping:
                        detected_task = task_mapping[task_type]
                    elif task_type == "Auto-detect":
                        detected_task = detect_task_type(final_query)
                    
                    num_docs = len([k for k in st.session_state['pdf_texts'].keys() if "Knowledge Base" not in k])
                    k_value = min(15, (num_docs + 3) * 3)  
                    
                    if st.session_state['vectorstore']:
                        docs, context_by_source = get_relevant_context(
                            st.session_state['vectorstore'],
                            final_query,
                            k=k_value
                        )
                    else:
                        context_by_source = {}
                    
                    
                    response = generate_response(
                        client,
                        final_query,
                        context_by_source,
                        detected_task,
                        target_language
                    )
                    
                    
                    task_icons = {
                        "invoice_compare": "📊",
                        "summary": "📝",
                        "convert_to_json": "📊",
                        "documentation": "📄",
                        "answer": "💡"
                    }
                    icon = task_icons.get(detected_task, "💡")
                    
                    st.markdown(f"## {icon} Response ({detected_task.replace('_', ' ').title()})")
                    
                    # Special handling for JSON responses
                    if detected_task == "convert_to_json":
                        # Try to parse and display as proper JSON
                        try:
                            # Clean the response
                            json_text = response.strip()
                            json_text = re.sub(r'```json\s*', '', json_text)
                            json_text = re.sub(r'```\s*$', '', json_text)
                            json_text = json_text.strip()
                            
                            # Parse JSON
                            parsed_json = json.loads(json_text)
                            
                            # Display JSON viewer
                            st.json(parsed_json)
                            
                            # Provide download button
                            json_str = json.dumps(parsed_json, indent=2)
                            st.download_button(
                                label="⬇️ Download JSON File",
                                data=json_str,
                                file_name=f"extracted_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                            
                            st.success("✅ JSON extracted successfully! You can view it above and download using the button.")
                            
                        except json.JSONDecodeError:
                            # If JSON parsing fails, show the raw response
                            st.warning("⚠️ Could not parse as JSON. Showing raw response:")
                            st.code(response, language="json")
                            
                            # Still provide download option
                            st.download_button(
                                label="⬇️ Download Response",
                                data=response,
                                file_name=f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                    else:
                        # Special rendering for invoice comparison
                        if detected_task == "invoice_compare":
                            st.markdown(response)
                            st.markdown("---")
                            st.download_button(
                                label="⬇️ Download Comparison Report (.txt)",
                                data=response,
                                file_name=f"invoice_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        else:
                            # Normal markdown response for other tasks
                            st.markdown(response)
                    
                    
                    with st.expander(f"📚 View Sources ({len(context_by_source)} documents referenced)"):
                        for source, chunks in context_by_source.items():
                            if "Knowledge Base" not in source:
                                st.markdown(f"**📄 {source}** ({len(chunks)} sections)")
                                with st.expander(f"View content from {source}"):
                                    for i, chunk in enumerate(chunks, 1):
                                        st.text_area(
                                            f"Section {i}",
                                            chunk,
                                            height=150,
                                            key=f"{source}_{i}_{len(st.session_state['chat_history'])}"
                                        )
                    
                    
                    st.session_state['chat_history'].append({
                        'question': final_query,
                        'answer': response,
                        'task_type': detected_task,
                        'sources': list(context_by_source.keys()),
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
        
        
        if st.session_state['chat_history']:
            st.markdown("---")
            with st.expander(f"📜 Conversation History ({len(st.session_state['chat_history'])} items)", expanded=False):
                for i, chat in enumerate(reversed(st.session_state['chat_history']), 1):
                    st.markdown(f"**Q{i}** [{chat['timestamp']}] *({chat['task_type']})*: {chat['question']}")
                    st.markdown(f"**A{i}:** {chat['answer'][:400]}...")
                    st.markdown(f"*Sources: {', '.join([s for s in chat['sources'] if 'Knowledge Base' not in s])}*")
                    st.divider()


if __name__ == "__main__":
    main()