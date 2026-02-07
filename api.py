
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from dotenv import load_dotenv
import os
import io
from typing import Dict
import uuid


load_dotenv()


app = FastAPI(
    title="PDF Chatbot API",
    description="Upload the pdf and start asking Questions about the PDF",
    version="1.0.0"
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


knowledge_bases: Dict[str, FAISS] = {}



class QuestionRequest(BaseModel):
    question: str
    session_id: str


class AnswerResponse(BaseModel):
    answer: str
    context: str
    session_id: str


class UploadResponse(BaseModel):
    message: str
    session_id: str
    filename: str
    pages: int
    chunks: int


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "PDF Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "streamlit_app": "Run with: python -m streamlit run app.py (port 8501)",
        "api_docs": "http://localhost:8000/docs",
        "endpoints": {
            "GET /health": "Check API health",
            "POST /upload": "Upload a PDF file and get session_id",
            "POST /ask": "Ask a question about uploaded PDF",
            "GET /sessions": "List all active sessions",
            "DELETE /session/{session_id}": "Delete a specific session",
            "DELETE /sessions": "Delete all sessions"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "groq_api_configured": bool(api_key),
        "active_sessions": len(knowledge_bases),
        "message": "API is running successfully!"
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and process it (same logic as your Streamlit app)
    Returns a session_id to use for asking questions
    """

    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        
        session_id = str(uuid.uuid4())
        
        
        pdf_content = await file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_content))
        
        
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        
        if not text.strip():
            raise HTTPException(
                status_code=400, 
                detail="Could not extract text from PDF. File might be empty or image-based."
            )
        
        
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        
        knowledge_bases[session_id] = knowledge_base
        
        return UploadResponse(
            message="PDF processed successfully",
            session_id=session_id,
            filename=file.filename,
            pages=len(pdf_reader.pages),
            chunks=len(chunks)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about an uploaded PDF (same logic as your Streamlit app)
    Requires session_id from upload endpoint
    """
    # Check if session exists
    if request.session_id not in knowledge_bases:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload a PDF first and use the returned session_id."
        )
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
       
        knowledge_base = knowledge_bases[request.session_id]
        
      
        docs = knowledge_base.similarity_search(request.question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
       
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided document context. Give direct, concise answers using only the information from the context."
                },
                {
                    "role": "user",
                    "content": f"""Context from document:
{context}

Question: {request.question}

Instructions:
- Give a direct, exact answer
- Use ONLY information from the context
- If it's a name, date, or fact, state it clearly
- If not in context, say "I cannot find this information in the document"

Answer:"""
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=500
        )
        
        answer = chat_completion.choices[0].message.content
        
        return AnswerResponse(
            answer=answer,
            context=context,
            session_id=request.session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "active_sessions": list(knowledge_bases.keys()),
        "count": len(knowledge_bases),
        "message": f"You have {len(knowledge_bases)} active session(s)"
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session and free up memory"""
    if session_id not in knowledge_bases:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    
    del knowledge_bases[session_id]
    
    return {
        "message": f"Session deleted successfully",
        "session_id": session_id,
        "remaining_sessions": len(knowledge_bases)
    }


@app.delete("/sessions")
async def delete_all_sessions():
    """Delete all sessions at once"""
    count = len(knowledge_bases)
    knowledge_bases.clear()
    
    return {
        "message": "All sessions deleted successfully",
        "deleted_count": count
    }



if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🚀 PDF CHATBOT - FastAPI Server Starting...")
    print("="*70)
    print(f"📍 API Server:    http://localhost:8000")
    print(f"📚 API Docs:      http://localhost:8000/docs")
    print(f"📖 ReDoc:         http://localhost:8000/redoc")
    print(f"❤️  Health Check: http://localhost:8000/health")
    print("="*70)
    print("💡 Your Streamlit app is separate! Run it with:")
    print("   python -m streamlit run app.py")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)