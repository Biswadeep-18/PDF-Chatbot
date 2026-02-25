 
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from groq import Groq
from dotenv import load_dotenv
import os
import io
import re
import json
import uuid
from typing import List, Dict, Optional
from duckduckgo_search import DDGS
from datetime import datetime

load_dotenv()

# --- Constants from app.py ---

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
- Incorrect calculations (quantity Ã— unit price â‰  total)
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
1. Missing or incorrect HS Codes â†’ Can cause customs delays, penalties, or rejection
2. Under-invoicing â†’ Risk of duty evasion charges, legal penalties
3. Over-invoicing â†’ FEMA violations, money laundering suspicion
4. Vague descriptions â†’ Goods may be held for inspection, delays
5. Missing tax IDs â†’ Cannot claim GST refunds, may face penalties
6. Wrong Incoterms â†’ Payment disputes, shipping cost disagreements
7. Sanctioned countries â†’ Goods may be seized, legal violations
8. Restricted items without license â†’ Goods confiscated, heavy fines

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

app = FastAPI(
    title="EAZZ.AI - Trade & Export PDF Chatbot API",
    description="Powerful API for PDF document analysis, invoice comparison, and multi-language support.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

groq_client = Groq(api_key=api_key)

# Global session store (In-memory)
knowledge_bases: Dict[str, FAISS] = {}
session_filenames: Dict[str, List[str]] = {}

# --- Pydantic Models ---

class QuestionRequest(BaseModel):
    question: str = Field(..., example="Summarize this document")
    session_id: str = Field(..., example="uuid-string")
    task_type: str = Field("Auto-detect", example="Auto-detect", description="Options: Auto-detect, Compare PDF, Summary, JSON Format, Documentation, Answer Question")
    language: str = Field("English", example="English", description="Select output language: English, Arabic, Hindi, etc.")

class AnswerResponse(BaseModel):
    answer: str
    task_detected: str
    language: str
    session_id: str
    timestamp: str

class UploadResponse(BaseModel):
    message: str
    session_id: str
    filenames: List[str]
    total_pages: int

# --- Helper Functions ---

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

def detect_task_type(query: str) -> str:
    """Detect what type of task the user is requesting"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["convert to json", "extract to json", "to json", "json format", "give me json", "as json", "in json"]):
        return "convert_to_json"
    elif any(word in query_lower for word in ["compare invoice", "invoice comparison", "difference between", "missing product", "compare two", "compare pdf"]):
        return "invoice_compare"
    elif any(word in query_lower for word in ["summarize", "summary", "give me a summary", "overview"]):
        return "summary"
    elif any(word in query_lower for word in ["documentation", "report", "document details", "party details", "compliance status"]):
        return "documentation"
    else:
        return "answer"

def generate_response(client: Groq, query: str, context: str, task_type: str = "answer", language: str = "English"):
    """Generate response using Groq API with specialized prompts"""
    
    # Real-time search integration logic
    is_real_time = any(word in query.lower() for word in ["news", "real time", "today", "current", "latest", "what is happening", "search"])
    search_context = ""
    if is_real_time or (task_type == "answer" and not context):
        search_context = web_search(query)
    
    if search_context:
        context = f"{context}\n\n### REAL-TIME WEB SEARCH RESULTS:\n{search_context}"

    system_prompts = {
        "answer": f"""You are a helpful assistant that answers questions based on provided PDF documents AND real-time web search results.
- Give direct, accurate answers using information from BOTH the PDF context and search results.
- If information comes from a PDF, mention the source filename.
- If information comes from a web search, label it as "Real-time updates".
- Prioritize PDF information for document-specific questions.
- IMPORTANT: Provide your entire response in {language}.""",
        
        "summary": f"""You are an expert at summarizing documents. 
- Create a comprehensive summary of the key points.
- Automatically detect the document type (Resume, Invoice, Report, etc.) and adapt the style.
- Highlight important findings or data points.
- Be concise but thorough.
- ONLY include information found in the documents. DO NOT hallucinate.
- IMPORTANT: Provide your entire response in {language}.""",
        
        "invoice_compare": f"""You are an expert invoice comparison analyst. Your ONLY job is to compare ALL uploaded invoices/PDFs and produce a STRICT UNIFIED comparison.
- Create a SINGLE master table showing ALL products from ALL invoices. Clearly mark availability with âœ… or âŒ.
- ONLY show data that is EXPLICITLY available in the PDFs.
- Use a SINGLE table to show product availability across all documents.
- For each document, add a column (e.g., "In Invoice 1", "In Invoice 2").
- IMPORTANT: Provide your entire response in {language}.""",
        
        "convert_to_json": """You are an expert data extraction specialist. Extract ALL information from the provided document and convert it into a clean, comprehensive JSON format.
- Extract ONLY information that is explicitly present in the document.
- Return ONLY valid JSON - no additional text, no explanations, no markdown.
""",
        
        "documentation": f"""You are a trade documentation expert. Your task is to generate a comprehensive "Documentation Report" based on the uploaded PDFs.
- For Invoices: Generate a structured summary of the transaction, party details, and compliance status.
- Format the output with clear headings, tables, and bullet points.
- ONLY include information present in the PDFs.
- IMPORTANT: Provide your entire response in {language}.
"""
    }
    
    system_prompt = system_prompts.get(task_type, system_prompts["answer"])
    
    user_prompt = f"""Context from uploaded PDF documents:
{context}

User Request: {query}

Please provide your response in {language}:"""
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1 if task_type == "convert_to_json" else 0.4,
            max_tokens=4000
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# --- API Endpoints ---

@app.get("/")
async def root():
    return {
        "name": "EAZZ.AI PDF Chatbot API",
        "version": "2.0.0",
        "status": "online",
        "documentation": "/docs"
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload one or more PDFs to create a new session"""
    session_id = str(uuid.uuid4())
    combined_texts = {}
    total_pages = 0
    filenames = []

    for file in files:
        if not file.filename.endswith('.pdf'):
            continue
        
        content = await file.read()
        pdf_reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        
        total_pages += len(pdf_reader.pages)
        combined_texts[file.filename] = text
        filenames.append(file.filename)

    if not combined_texts:
        raise HTTPException(status_code=400, detail="No valid PDF files uploaded")

    # Add knowledge bases
    combined_texts["Invoice Standards"] = INVOICE_STANDARDS
    combined_texts["GST Guide"] = GST_CUSTOMS_GUIDE
    combined_texts["Compliance Risks"] = COMPLIANCE_RISKS

    # Create Vector Store
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for name, text in combined_texts.items():
        chunks = splitter.split_text(text)
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={"source": name}))
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    knowledge_bases[session_id] = vectorstore
    session_filenames[session_id] = filenames

    return UploadResponse(
        message="Documents processed and session created",
        session_id=session_id,
        filenames=filenames,
        total_pages=total_pages
    )

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    """Ask a question or request a task (Summary, Comparison, JSON) using a session_id"""
    if request.session_id not in knowledge_bases:
        raise HTTPException(status_code=404, detail="Session not found")

    vectorstore = knowledge_bases[request.session_id]
    
    # Task mapping
    task_mapping = {
        "JSON Format": "convert_to_json",
        "Compare PDF": "invoice_compare",
        "Summary": "summary",
        "Documentation": "documentation"
    }
    
    detected_task = task_mapping.get(request.task_type, request.task_type)
    if detected_task == "Auto-detect":
        detected_task = detect_task_type(request.question)

    # Retrieval
    docs = vectorstore.similarity_search(request.question, k=10)
    context = "\n\n".join([f"Source: {d.metadata['source']}\n{d.page_content}" for d in docs])

    # Generation
    response = generate_response(
        groq_client, 
        request.question, 
        context, 
        detected_task, 
        request.language
    )

    return AnswerResponse(
        answer=response,
        task_detected=detected_task,
        language=request.language,
        session_id=request.session_id,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in knowledge_bases:
        del knowledge_bases[session_id]
        if session_id in session_filenames:
            del session_filenames[session_id]
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
