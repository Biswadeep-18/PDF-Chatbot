from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class QuestionRequest(BaseModel):
    question: str = Field(..., example="Summarize this document")
    session_id: str = Field(..., example="uuid-string")
    task_type: str = Field("Auto-detect", example="Auto-detect")
    language: str = Field("English", example="English")

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
