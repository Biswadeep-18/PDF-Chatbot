from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Header
import uuid
import io
from datetime import datetime
from typing import List, Dict

from ..models.schemas import QuestionRequest, AnswerResponse, UploadResponse
from ..services.pdf_service import PDFService
from ..services.vector_service import VectorService
from ..services.ai_service import AIService
from ..services.profile_service import ProfileService
from ..core.constants import INVOICE_STANDARDS, GST_CUSTOMS_GUIDE, COMPLIANCE_RISKS
from ..core.auth_utils import decode_access_token

router = APIRouter()

# In-memory storage for session data (consistent with original api.py)
knowledge_bases = {}
session_filenames = {}

pdf_service = PDFService()
vector_service = VectorService()
ai_service = AIService()
profile_service = ProfileService()

async def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    token = authorization.split(" ")[1]
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return payload["sub"]

@router.get("/")
async def root():
    return {"message": "PDF Chatbot API is running"}

@router.post("/upload", response_model=UploadResponse)
async def upload_pdfs(files: List[UploadFile] = File(...), current_user: str = Depends(get_current_user)):
    """Upload PDFs and create a session"""
    session_id = str(uuid.uuid4())
    combined_texts = {}
    total_pages = 0
    filenames = []

    for file in files:
        if not file.filename.endswith('.pdf'):
            continue
        
        content = await file.read()
        text, num_pages = pdf_service.extract_text(io.BytesIO(content), file.filename)
        
        total_pages += num_pages
        combined_texts[file.filename] = text
        filenames.append(file.filename)

    if not combined_texts:
        raise HTTPException(status_code=400, detail="No valid PDF files uploaded")

    # Create Vector Store
    vectorstore = vector_service.create_vector_store(combined_texts)
    
    knowledge_bases[session_id] = vectorstore
    session_filenames[session_id] = filenames

    return UploadResponse(
        message="Documents processed and session created",
        session_id=session_id,
        filenames=filenames,
        total_pages=total_pages
    )

@router.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest, current_user: str = Depends(get_current_user)):
    """Ask a question using session_id"""
    if request.session_id not in knowledge_bases:
        raise HTTPException(status_code=404, detail="Session not found")

    vectorstore = knowledge_bases[request.session_id]
    
    # Task detection logic
    detected_task = request.task_type
    if detected_task in ["Auto-detect", "Auto"]:
        detected_task = ai_service.detect_task_type(request.question)
    
    # Task mapping (for backward compatibility with frontend options if needed)
    task_mapping = {
        "JSON Format": "convert_to_json",
        "Compare PDF": "invoice_compare",
        "Summary": "summary",
        "Documentation": "documentation",
        "Answer Question": "answer"
    }
    detected_task = task_mapping.get(detected_task, detected_task)

    # Retrieval
    context = vector_service.get_context(vectorstore, request.question)

    # Generation
    response = ai_service.generate_response(
        query=request.question,
        context=context,
        task_type=detected_task,
        language=request.language
    )

    return AnswerResponse(
        answer=response,
        task_detected=detected_task,
        language=request.language,
        session_id=request.session_id,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in knowledge_bases:
        del knowledge_bases[session_id]
        if session_id in session_filenames:
            del session_filenames[session_id]
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")

@router.get("/profile")
async def get_profile():
    return profile_service.get_profile()

@router.put("/profile")
async def update_profile(data: Dict):
    return profile_service.update_profile(data)
