from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
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

REGULATION_UPDATES = {
    "2025-02": {
        "title": "New HS Code Requirements (Feb 2025)",
        "description": "Customs now requires 8-digit HS codes instead of 6-digit for electronics and textiles",
        "impact": "HIGH",
        "action_required": "Update all invoices with 8-digit HS codes for electronics/textiles"
    },
    "2025-01": {
        "title": "GST Portal Enhancement (Jan 2025)",
        "description": "New e-invoice system mandatory for exports above $10,000",
        "impact": "MEDIUM",
        "action_required": "Register on e-invoice portal and generate IRN for high-value exports"
    },
    "2024-12": {
        "title": "RCEP Trade Agreement Updates (Dec 2024)",
        "description": "Reduced duties for exports to RCEP countries (ASEAN, China, Japan, Korea)",
        "impact": "POSITIVE",
        "action_required": "Apply for Certificate of Origin to avail reduced duties"
    }
}



def process_pdf(pdf_file) -> tuple:
    """Extract text from a single PDF file"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page_num, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
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
    
    system_prompt = """You are an expert data extraction specialist. Extract ALL information from the provided document and structure it into a clean, comprehensive JSON format.

TASK: Extract and structure ALL data from the document

For INVOICES, extract:
{
  "document_type": "invoice",
  "invoice_details": {
    "invoice_number": "",
    "invoice_date": "",
    "po_number": "",
    "reference_number": ""
  },
  "exporter": {
    "name": "",
    "address": "",
    "city": "",
    "state": "",
    "country": "",
    "postal_code": "",
    "tax_id": "",
    "gstin": "",
    "email": "",
    "phone": ""
  },
  "importer": {
    "name": "",
    "address": "",
    "city": "",
    "state": "",
    "country": "",
    "postal_code": "",
    "tax_id": "",
    "email": "",
    "phone": ""
  },
  "shipping_details": {
    "ship_to_address": "",
    "shipping_method": "",
    "incoterms": "",
    "port_of_loading": "",
    "port_of_discharge": "",
    "country_of_origin": ""
  },
  "items": [
    {
      "sno": 1,
      "description": "",
      "hs_code": "",
      "quantity": 0,
      "unit": "",
      "unit_price": 0,
      "total_price": 0,
      "tax_rate": 0,
      "tax_amount": 0
    }
  ],
  "financial_summary": {
    "subtotal": 0,
    "tax_amount": 0,
    "shipping_charges": 0,
    "discount": 0,
    "grand_total": 0,
    "currency": ""
  },
  "payment_terms": {
    "terms": "",
    "due_date": "",
    "bank_details": {
      "bank_name": "",
      "account_number": "",
      "ifsc_code": "",
      "swift_code": "",
      "iban": ""
    }
  },
  "additional_info": {
    "notes": "",
    "terms_and_conditions": "",
    "authorized_signatory": ""
  }
}

For OTHER DOCUMENTS (Packing List, Bill of Lading, etc.), adapt the structure accordingly.

RULES:
1. Extract ALL fields present in the document
2. Use null or empty string for missing fields
3. Convert all numbers to appropriate numeric types
4. Maintain proper data types (strings, numbers, arrays, objects)
5. Include ALL line items/products found
6. Extract dates in ISO format (YYYY-MM-DD) when possible
7. Return ONLY valid JSON, no additional text or explanation
8. Be thorough - extract even small details

Return the complete JSON structure."""

    user_prompt = f"""Document Content:

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
            max_tokens=4000
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


def generate_response(client: Groq, query: str, context_by_source: Dict, task_type: str = "answer"):
    """Generate response using Groq API"""
    
    context_parts = []
    for source, chunks in context_by_source.items():
        context_parts.append(f"### From {source}:")
        context_parts.append("\n".join(chunks))
        context_parts.append("")
    
    full_context = "\n".join(context_parts)
    
    system_prompts = {
        "answer": """You are a helpful assistant that answers questions based on provided PDF documents. 
- Give direct, accurate answers using information from the context
- When information comes from multiple PDFs, mention the sources
- If comparing PDFs, clearly highlight similarities and differences
- If information is not in the documents, say so clearly""",
        
        "email": """You are a professional email writer. Based on the document context provided:
- Write a clear, professional email
- Use appropriate email format (Subject, Greeting, Body, Closing)
- Incorporate relevant information from the documents
- Keep it concise and well-structured""",
        
        "summary": """You are an expert at summarizing documents. 
- Create a comprehensive summary of the key points
- If multiple documents are provided, organize by source or theme
- Highlight important findings, data, or conclusions
- Be concise but thorough""",
        
        "comparison": """You are an expert at comparing and analyzing multiple documents.
- Compare ALL uploaded documents systematically
- Identify key similarities and differences between each document
- Organize your comparison in a clear structure with sections for each document
- Highlight unique points from each document
- Provide insights on the overall comparison across all documents
- Create a comprehensive comparison table if helpful""",
        
        "merge": """You are an expert at merging multiple documents into one cohesive document.
- Combine information from ALL uploaded PDFs
- Organize the merged content logically by topics or themes
- Remove duplicate information
- Maintain all important details from each source
- Clearly indicate which information came from which source
- Create a well-structured, comprehensive merged document""",
        
        "invoice_check": """You are an expert trade compliance auditor. Analyze the provided invoice against international trade standards.

TASK: Identify ALL errors, missing fields, and compliance issues

Check for:
1. MANDATORY FIELDS: HS Codes, Tax IDs (GSTIN), Incoterms, detailed descriptions, country of origin
2. CALCULATIONS: Verify quantity × unit price = total for each line item
3. COMPLETENESS: All required exporter/importer details present
4. CLARITY: Descriptions are specific (not vague like "goods" or "items")
5. COMPLIANCE: Currency specified, bank details present, authorized signature mentioned

OUTPUT FORMAT:
## ✅ CORRECT ITEMS
- List what's done correctly

## ❌ CRITICAL ERRORS (Must Fix)
- List all critical missing fields or errors with specific line references

## ⚠️ WARNINGS (Should Fix)
- List items that need improvement

## 📋 MISSING INFORMATION
- List all missing mandatory fields

## 💡 RECOMMENDATIONS
- Provide specific actionable fixes

Be thorough and specific. Reference actual line items from the invoice.""",
        
        "export_gen": """You are an international trade documentation expert. Generate professional export documents.

Based on the provided invoice data, create:

1. COMMERCIAL INVOICE
   - Formal header with exporter details
   - Complete importer details
   - Itemized product table with HS codes
   - All calculated totals
   - Payment terms and banking details
   - Declaration and signature block

2. PACKING LIST
   - Shipment details (date, reference number)
   - Complete packaging breakdown
   - Net and gross weights
   - Dimensions if available
   - Marks and numbers
   - Total packages count

Use professional formatting with clear sections, tables, and standard trade terminology.
Ensure both documents are complete, accurate, and ready for customs clearance.""",
        
        "gst_explain": """You are a tax and customs expert who explains complex regulations in simple language.

TASK: Explain GST and customs concepts in easy-to-understand terms

- Use simple analogies and examples
- Break down complex terms
- Explain step-by-step processes
- Highlight what matters for the user's specific situation
- Use emojis to make it more readable
- Provide practical tips

Make it feel like you're explaining to a friend, not reading a textbook.""",
        
        "checklist_gen": """You are a trade compliance expert. Generate a comprehensive document checklist.

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

Format as an actionable checklist with checkboxes (☐) that users can follow.""",
        
        "risk_warning": """You are a trade compliance risk analyst. Analyze the provided documents for compliance risks.

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

Prioritize by severity and provide realistic timelines for resolution.""",
        
        "invoice_compare": """You are an expert invoice comparison analyst. Compare multiple invoices and extract detailed product information.

TASK: Create a comprehensive comparison of products across ALL invoices

For each invoice, extract:
1. Invoice metadata (number, date, party details)
2. ALL products with their:
   - Product Name/Description
   - HS Code (if available)
   - Quantity
   - Unit of Measurement
   - Unit Price
   - Total Amount
   - Any other relevant details

OUTPUT FORMAT:

## 📊 INVOICE SUMMARY TABLE
Create a markdown table comparing basic invoice details (invoice numbers, dates, parties, totals)

## 🔍 DETAILED PRODUCT COMPARISON

For EACH product found across all invoices, provide:
- Product Name
- Which invoice(s) it appears in
- Quantity in each invoice
- Price differences (if any)
- Status: (Present in both/Missing from Invoice X)

## ❌ MISSING PRODUCTS
List products that appear in one invoice but NOT in others:
- Product name
- Missing from which invoice
- Quantity and value in the invoice where it exists

## 📈 QUANTITY & VALUE ANALYSIS
- Total items in each invoice
- Total value comparison
- Quantity differences for common products
- Price variance analysis

## 💡 KEY INSIGHTS & SUMMARY
- Main differences between invoices
- Products unique to each invoice
- Pricing discrepancies
- Recommendations

Be extremely thorough and extract ALL product details from the context provided.""",
        
        "create_invoice": """You are an expert invoice generator. Create a professional, complete commercial invoice.

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
Make it print-ready and customs-compliant.""",
        
        "convert_to_json": """You are an expert data extraction specialist. Extract ALL information from the provided document and convert it into a clean, comprehensive JSON format.

TASK: Extract and structure ALL data from the document into proper JSON

For INVOICES, extract and return in this JSON structure:
{
  "document_type": "invoice",
  "invoice_details": {
    "invoice_number": "",
    "invoice_date": "",
    "po_number": "",
    "reference_number": ""
  },
  "exporter": {
    "name": "",
    "address": "",
    "city": "",
    "state": "",
    "country": "",
    "postal_code": "",
    "tax_id": "",
    "gstin": "",
    "email": "",
    "phone": ""
  },
  "importer": {
    "name": "",
    "address": "",
    "city": "",
    "state": "",
    "country": "",
    "postal_code": "",
    "tax_id": "",
    "email": "",
    "phone": ""
  },
  "shipping_details": {
    "ship_to_address": "",
    "shipping_method": "",
    "incoterms": "",
    "port_of_loading": "",
    "port_of_discharge": "",
    "country_of_origin": ""
  },
  "items": [
    {
      "sno": 1,
      "description": "",
      "hs_code": "",
      "quantity": 0,
      "unit": "",
      "unit_price": 0,
      "total_price": 0,
      "tax_rate": 0,
      "tax_amount": 0
    }
  ],
  "financial_summary": {
    "subtotal": 0,
    "tax_amount": 0,
    "shipping_charges": 0,
    "discount": 0,
    "grand_total": 0,
    "currency": ""
  },
  "payment_terms": {
    "terms": "",
    "due_date": "",
    "bank_details": {
      "bank_name": "",
      "account_number": "",
      "ifsc_code": "",
      "swift_code": "",
      "iban": ""
    }
  },
  "additional_info": {
    "notes": "",
    "terms_and_conditions": "",
    "authorized_signatory": ""
  },
  "extraction_metadata": {
    "source_filename": "",
    "extraction_timestamp": "",
    "extraction_method": "AI-powered"
  }
}

For OTHER DOCUMENTS, adapt the structure accordingly.

CRITICAL RULES:
1. Extract ALL fields present in the document
2. Use null or empty string for missing fields
3. Convert all numbers to appropriate numeric types (not strings)
4. Maintain proper JSON data types (strings, numbers, arrays, objects)
5. Include ALL line items/products found
6. Extract dates in ISO format (YYYY-MM-DD) when possible
7. Return ONLY valid JSON - no additional text, no explanations, no markdown
8. Be thorough - extract even small details
9. Ensure the JSON is properly formatted and can be parsed

Return ONLY the JSON structure, nothing else."""
    }
    
    system_prompt = system_prompts.get(task_type, system_prompts["answer"])
    
    user_prompt = f"""Context from uploaded PDF documents:

{full_context}

User Request: {query}

Please provide your response:"""
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1 if task_type == "convert_to_json" else (0.3 if task_type in ["invoice_check", "risk_warning", "invoice_compare"] else 0.7),
            max_tokens=4000
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"


def detect_task_type(query: str) -> str:
    """Detect what type of task the user is requesting"""
    query_lower = query.lower()
    
    # JSON conversion detection - HIGHEST PRIORITY
    if any(word in query_lower for word in ["convert to json", "convert this to json", "extract to json", "to json", "json format", "give me json", "as json", "in json", "convert pdf to json", "extract json", "make json"]):
        return "convert_to_json"
    # Invoice comparison detection
    elif any(word in query_lower for word in ["compare invoice", "invoice comparison", "difference between invoice", "missing product", "missing items", "compare two invoice"]):
        return "invoice_compare"
    elif any(word in query_lower for word in ["create invoice", "generate invoice", "make invoice", "new invoice", "invoice from"]):
        return "create_invoice"
    elif any(word in query_lower for word in ["check error", "invoice error", "audit invoice", "check invoice", "verify invoice", "find errors"]):
        return "invoice_check"
    elif any(word in query_lower for word in ["generate export", "create export", "export document", "packing list", "commercial invoice"]):
        return "export_gen"
    elif any(word in query_lower for word in ["explain gst", "what is gst", "customs duty", "explain customs", "what is customs", "incoterms", "explain tax"]):
        return "gst_explain"
    elif any(word in query_lower for word in ["checklist", "documents needed", "what documents", "required documents", "document list"]):
        return "checklist_gen"
    elif any(word in query_lower for word in ["risk", "compliance risk", "warning", "what can go wrong", "potential issues"]):
        return "risk_warning"
    elif any(word in query_lower for word in ["merge", "combine", "join", "merge pdfs", "combine pdfs"]):
        return "merge"
    elif any(word in query_lower for word in ["email", "write an email", "draft an email", "compose email"]):
        return "email"
    elif any(word in query_lower for word in ["summarize", "summary", "give me a summary"]):
        return "summary"
    elif any(word in query_lower for word in ["compare", "comparison", "difference", "similar", "contrast"]):
        return "comparison"
    else:
        return "answer"


def get_regulation_updates() -> List[Dict]:
    """Get recent regulation updates"""
    updates = []
    for date_key in sorted(REGULATION_UPDATES.keys(), reverse=True):
        update = REGULATION_UPDATES[date_key].copy()
        update['date'] = date_key
        updates.append(update)
    return updates


def quick_invoice_scan(invoice_text: str) -> Dict:
    """Quick automated scan of invoice for common issues"""
    issues = {
        "critical": [],
        "warnings": [],
        "suggestions": []
    }
    
    invoice_lower = invoice_text.lower()
    
    
    if not re.search(r'\bhs\s*code\b|\bhscode\b|\bharmoni[sz]ed\b', invoice_lower):
        issues["critical"].append("❌ Missing HS Code - Required for customs clearance")
    
    
    if not re.search(r'\bgstin\b|\btax\s*id\b|\bein\b|\bpan\b', invoice_lower):
        issues["critical"].append("❌ Missing Tax ID/GSTIN - Required for GST compliance")
    
    
    if not re.search(r'\bfob\b|\bcif\b|\bdap\b|\bddp\b|\bexw\b|\bincoterm', invoice_lower):
        issues["warnings"].append("⚠️ Incoterms not specified - May cause payment disputes")
    
    
    if not re.search(r'country\s*of\s*origin|\borigin\b.*country', invoice_lower):
        issues["warnings"].append("⚠️ Country of origin not mentioned")
    
    
    if not re.search(r'\busd\b|\beur\b|\binr\b|\bgbp\b|\bcurrency\b', invoice_lower):
        issues["warnings"].append("⚠️ Currency not clearly specified")
    
    
    if re.search(r'\bgoods\b|\bitems\b|\bproducts\b(?!\s+\w+)', invoice_lower):
        issues["suggestions"].append("💡 Product descriptions seem vague - Add more details")
    
    
    if not re.search(r'\bbank\b|\bifsc\b|\bswift\b|\biban\b|\baccount\s*number\b', invoice_lower):
        issues["suggestions"].append("💡 Bank details not found - Required for payment processing")
    
    return issues



def main():
    load_dotenv()
    
    st.set_page_config(
        page_title="Trade & Export AI Assistant", 
        page_icon="🤖",
        layout="wide"
    )
    
    
    st.markdown("""
    <style>
    * {
        color: #000000;
    }
    
    body, .main, .stApp, .stAppHeader {
        background-color: #a8e6cf !important;
    }
    
    .stAppHeader {
        background-color: #a8e6cf !important;
    }
    
    header {
        background-color: #a8e6cf !important;
    }
    
    .stAppToolbar {
        background-color: #a8e6cf !important;
    }
    
    [data-testid="stHeader"] {
        background-color: #a8e6cf !important;
    }
    
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #c5f0e8 !important;
        border: 2px solid #1f77b4;
        color: #000000 !important;
    }
    
    .feature-box {
        background-color: #c5f0e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
        color: #000000;
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
        font-weight: 700;
    }
    
    .exporter, .importer, .name-field, .company-name {
        color: #22a447 !important;
        font-weight: 700;
    }
    
    p, label, span {
        color: #000000 !important;
    }
    
    .stTextInput, .stTextArea, .stSelectbox, .stNumberInput {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Upload file area styling */
    [data-testid="stFileUploadDropzone"] {
        background-color: #c5f0e8 !important;
        border: 3px dashed #1f77b4 !important;
        border-radius: 10px !important;
    }
    
    /* File upload area text */
    .st-emotion-cache-tv2gmp {
        background-color: #c5f0e8 !important;
    }
    
    /* Browse files button */
    .stButton > button {
        background-color: #1f77b4 !important;
        color: #ffffff !important;
        font-weight: 600;
        border: 2px solid #0d3a66;
        border-radius: 8px;
    }
    
    .stButton > button:hover {
        background-color: #0d3a66 !important;
        color: #ffffff !important;
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
        color: #1f77b4 !important;
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

        img_base64 = get_base64_image("eatech-logo.png")

        st.markdown(f"""
        <style>
        .header {{
            display: flex;
            align-items: center;
            background-color: #a8e6cf;
            padding: 20px;
            border-radius: 10px;
            flex-direction: row;
        }}

        .robot {{
            position: fixed;
            top: 20px;
            right: 20px;
            width: 200px;
            height: 200px;
            z-index: 999;
            animation: moveAround 20s infinite linear;
        }}
        
        .robot img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}
        
        .robot::after {{
            content: "Ask me anything! 💬";
            position: absolute;
            top: -60px;
            left: 50%;
            transform: translateX(-50%);
            background-color: #1f77b4;
            color: #ffffff;
            padding: 12px 18px;
            border-radius: 15px;
            font-size: 14px;
            font-weight: 600;
            white-space: nowrap;
            box-shadow: 0 4px 8px rgba(31, 119, 180, 0.3);
            z-index: 10;
            animation: fadeInDown 0.5s ease-in;
        }}
        
        .robot::before {{
            content: "";
            position: absolute;
            top: -15px;
            left: 50%;
            transform: translateX(-50%);
            width: 0;
            height: 0;
            border-left: 10px solid transparent;
            border-right: 10px solid transparent;
            border-top: 10px solid #1f77b4;
            z-index: 10;
        }}

        @keyframes moveAround {{
            0% {{ 
                top: 50px;
                right: 50px;
            }}
            10% {{ 
                top: 100px;
                right: 100px;
            }}
            20% {{ 
                top: 200px;
                right: 200px;
            }}
            30% {{ 
                top: 300px;
                right: 100px;
            }}
            40% {{ 
                top: 400px;
                right: 50px;
            }}
            50% {{ 
                top: 300px;
                right: 150px;
            }}
            60% {{ 
                top: 200px;
                right: 300px;
            }}
            70% {{ 
                top: 100px;
                right: 200px;
            }}
            80% {{ 
                top: 150px;
                right: 80px;
            }}
            90% {{ 
                top: 250px;
                right: 120px;
            }}
            100% {{ 
                top: 50px;
                right: 50px;
            }}
        }}
        
        @keyframes fadeInDown {{
            from {{
                opacity: 0;
                transform: translateX(-50%) translateY(-20px);
            }}
            to {{
                opacity: 1;
                transform: translateX(-50%) translateY(0);
            }}
        }}

        .title {{
            font-size: 60px;
            font-weight: bold;
            color: #22a447;
            text-shadow: 2px 2px 4px rgba(31, 119, 180, 0.3);
            flex: 1;
        }}
        </style>

        <div class="header">
            <div class="title">
                EATECH.AI
            </div>
        </div>
        
        <div class="robot">
            <img src="data:image/png;base64,{img_base64}" width="200">
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
    if 'quick_scan_results' not in st.session_state:
        st.session_state['quick_scan_results'] = None
    if 'extracted_json_data' not in st.session_state:
        st.session_state['extracted_json_data'] = {}
    
    
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
                    
                    
                    for filename, text in st.session_state['pdf_texts'].items():
                        if 'invoice' in filename.lower() or 'export' in filename.lower():
                            st.session_state['quick_scan_results'] = quick_invoice_scan(text)
                            break
                    
                    st.success(f"✅ Processed {len(uploaded_files)} document(s)!")
                    st.balloons()
        
        
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
                st.session_state['quick_scan_results'] = None
                st.session_state['extracted_json_data'] = {}
                st.rerun()
    
    st.markdown("---")
    with st.expander("🔔 Latest Regulation Updates & Alerts", expanded=False):
        updates = get_regulation_updates()
        for update in updates[:3]:  # Show latest 3
            impact_color = {"HIGH": "🔴", "MEDIUM": "🟡", "POSITIVE": "🟢"}.get(update['impact'], "⚪")
            st.markdown(f"""
            **{impact_color} {update['title']}** ({update['date']})  
            {update['description']}  
            *Action Required:* {update['action_required']}
            """)
            st.markdown("---")
    
    
    
    if st.session_state.get('quick_scan_results'):
        st.markdown("## 🔍 Quick Invoice Scan Results")
        results = st.session_state['quick_scan_results']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if results['critical']:
                st.error(f"**Critical Issues: {len(results['critical'])}**")
                for issue in results['critical']:
                    st.write(issue)
        with col2:
            if results['warnings']:
                st.warning(f"**Warnings: {len(results['warnings'])}**")
                for issue in results['warnings']:
                    st.write(issue)
        with col3:
            if results['suggestions']:
                st.info(f"**Suggestions: {len(results['suggestions'])}**")
                for issue in results['suggestions']:
                    st.write(issue)
        
        st.markdown("---")
    
    
    if not st.session_state['pdf_texts']:
        st.info("👈 Upload your export documents (invoices, packing lists, etc.) to get started!")
        
        
        st.markdown("## 🎯 What Can I Do For You?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-box">
            <h3>🔍 Invoice Error Detection</h3>
            <p>Upload invoice → Get instant error report with missing fields, calculation errors, and compliance issues</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-box">
            <h3>📋 Document Checklist</h3>
            <p>Generate complete checklist of required documents for your specific export scenario</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-box">
            <h3>📄 Auto-Generate Export Docs</h3>
            <p>Create professional Commercial Invoices and Packing Lists from your data</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-box">
            <h3>⚠️ Compliance Risk Warnings</h3>
            <p>Identify high-risk issues that could delay shipments or cause penalties</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-box">
            <h3>📊 Compare Invoices</h3>
            <p>Upload 2+ invoices → Get detailed comparison table showing missing products, quantity differences, and pricing analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-box">
            <h3>📊 Convert to JSON (NEW!)</h3>
            <p>Just ask AI "convert this to JSON" → Get all data extracted in clean, structured JSON format!</p>
            </div>
            """, unsafe_allow_html=True)
        
    else:
        
        st.markdown("## 💬 Ask Me Anything")
        
        
        st.markdown("**Quick Actions:**")
        
        # Show invoice comparison button only if 2+ invoices uploaded
        user_docs = [name for name in st.session_state['pdf_texts'].keys() if "Knowledge Base" not in name]
        
        if len(user_docs) >= 2:
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        else:
            col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        quick_action = None
        with col1:
            if st.button("📊 Extract JSON"):
                quick_action = "convert_to_json_button_clicked"
        
        with col2:
            if st.button("🔍 Check Errors"):
                quick_action = "Check this invoice for all errors and missing mandatory fields"
        with col3:
            if st.button("📄 Generate Docs"):
                quick_action = "Generate export documents (Commercial Invoice and Packing List) from this data"
        with col4:
            if st.button("📚 Explain GST"):
                quick_action = "Explain GST and customs duty concepts in simple language for my export"
        with col5:
            if st.button("📋 Checklist"):
                quick_action = "Generate a complete document checklist for this export transaction"
        with col6:
            if st.button("⚠️ Risk Analysis"):
                quick_action = "Analyze compliance risks in these documents"
        
        # Invoice comparison button (only show if 2+ invoices)
        if len(user_docs) >= 2:
            with col7:
                if st.button("🔀 Compare Invoices"):
                    quick_action = "Compare all uploaded invoices and show me a detailed table of products, quantities, prices, and identify missing items in each invoice"
        
        
        user_question = st.text_input(
            "Your question or request:",
            value=quick_action if quick_action else "",
            placeholder="e.g., 'Convert this PDF to JSON format' or 'Extract all data as JSON'"
        )
        
        
        task_type = st.selectbox(
            "Task Type (Auto-detect enabled)",
            ["Auto-detect", "Convert to JSON", "Invoice Comparison", "Create New Invoice", "Invoice Error Check", "Generate Export Documents", "GST/Customs Explanation", 
             "Document Checklist", "Risk Analysis", "Answer Question", "Email", "Summary", "Comparison"],
            help="Select task type or let AI auto-detect from your query"
        )
        
        
        with st.expander("💡 Example Queries"):
            st.markdown("""
            **📊 Convert to JSON (NEW!):**
            - Convert this PDF to JSON format
            - Extract all data as JSON
            - Give me JSON format of this invoice
            - Convert to JSON
            - I want JSON format
            - Extract this document to JSON
            
            **📊 Invoice Comparison:**
            - Compare both invoices and show me missing products
            - Which items are in invoice 1 but not in invoice 2?
            - Create a comparison table of all products with quantities
            - Show me quantity and price differences between invoices
            
            **🆕 Create New Invoice:**
            - Create a professional invoice from this data
            - Generate a new commercial invoice
            - Make an invoice with all proper fields
            
            **🔍 Invoice Error Checking:**
            - Check my invoice for all errors and missing fields
            - Is my invoice compliant with export standards?
            - What's wrong with this invoice?
            
            **📄 Export Document Generation:**
            - Generate a commercial invoice from this data
            - Create a packing list for these items
            - Generate export documents for customs
            
            **📚 GST & Customs Education:**
            - Explain GST for exports in simple terms
            - What are Incoterms and which should I use?
            - How does customs duty work for imports?
            - What is duty drawback?
            
            **📋 Document Checklist:**
            - What documents do I need for exporting to USA?
            - Generate export documentation checklist
            - What paperwork is required for customs clearance?
            
            **⚠️ Risk Analysis:**
            - What compliance risks exist in my invoice?
            - Can this shipment face customs delays?
            - Analyze potential issues with this transaction
            """)
        
        
        # Handle JSON button click or user question
        if quick_action == "convert_to_json_button_clicked" or (user_question and st.session_state['vectorstore']):
            with st.spinner("🤖 Processing your request..."):
                
                # If JSON button was clicked, process all documents
                if quick_action == "convert_to_json_button_clicked":
                    st.markdown("## 📊 Response (Convert To Json)")
                    
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
                elif user_question:
                    # Normal question processing
                    if task_type == "Auto-detect":
                        detected_task = detect_task_type(user_question)
                    else:
                        task_map = {
                            "Convert to JSON": "convert_to_json",
                            "Invoice Comparison": "invoice_compare",
                            "Create New Invoice": "create_invoice",
                            "Invoice Error Check": "invoice_check",
                            "Generate Export Documents": "export_gen",
                            "GST/Customs Explanation": "gst_explain",
                            "Document Checklist": "checklist_gen",
                            "Risk Analysis": "risk_warning",
                            "Answer Question": "answer",
                            "Email": "email",
                            "Summary": "summary",
                            "Comparison": "comparison"
                        }
                        detected_task = task_map.get(task_type, "answer")
                    
                    
                    num_docs = len([k for k in st.session_state['pdf_texts'].keys() if "Knowledge Base" not in k])
                    k_value = min(15, (num_docs + 3) * 3)  
                    
                    docs, context_by_source = get_relevant_context(
                        st.session_state['vectorstore'],
                        user_question,
                        k=k_value
                    )
                    
                    
                    response = generate_response(
                        client,
                        user_question,
                        context_by_source,
                        detected_task
                    )
                    
                    
                    task_icons = {
                        "invoice_check": "🔍",
                        "export_gen": "📄",
                        "gst_explain": "📚",
                        "checklist_gen": "📋",
                        "risk_warning": "⚠️",
                        "invoice_compare": "📊",
                        "create_invoice": "🆕",
                        "convert_to_json": "📊"
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
                        'question': user_question,
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