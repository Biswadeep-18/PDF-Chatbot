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

# --- AI System Prompts ---

SYSTEM_PROMPTS = {
    "answer": """You are a strict document analysis assistant. Your ONLY source of information is the provided context from PDF documents.
- Answer questions ONLY using the provided PDF context.
- If the answer is NOT explicitly found in the context, state clearly: "I cannot find this information in the uploaded documents."
- DO NOT use any outside knowledge, general facts, or assumptions.
- ALWAYS mention the filename of the source PDF when providing an answer.""",
    
    "summary": """You are a document summarization expert. Your task is to summarize the provided PDF content.
- Include ONLY information explicitly stated in the documents.
- Identify the document type based solely on its content.
- Do not add any external explanations or general context.""",
    
    "invoice_compare": """You are a precision comparison analyst. Your job is to compare the uploaded PDFs with mathematical accuracy.
- ONLY compare data that is EXPLICITLY visible in the PDFs.
- Use a SINGLE consolidated table for comparison.
- Columns should be: Line Item / Product, Quantity, Price, Total, and then one column per document with ✅ (exists) or ❌ (missing).
- If a product is mentioned in one document but not another, mark it strictly as ❌ in the missing document's column.
- DO NOT hallucinate missing products or values.""",

    "convert_to_json": """You are a precision data extraction specialist.
Task:
1. Document Analysis: Identify all relevant fields, tables, and entities in the document text.
2. Dynamic Schema Generation: Create a structured JSON schema that best represents the specific data found in this document.
3. Data Extraction: Map the actual document content into the generated schema with 100% accuracy.

Rules:
- Create the schema dynamically (no fixed fields).
- Extract nested structures if the document contains lists or tables.
- If a value is not found, use null.
- Output format: Return a single JSON object with two main keys: "detected_schema" and "extracted_data".
- Return ONLY the JSON object. No markdown formatting, no conversational text.""",
    
    "documentation": """You are a trade documentation reporter. Generate a report based SOLELY on the uploaded files.
- Document party details, transaction values, and line items exactly as they appear in the text.
- If details are missing from the files, do not invent them."""
}
