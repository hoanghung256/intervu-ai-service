from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime

class UserSession(BaseModel):
    id: str
    history: List[Dict] = []
    last_cv: Optional[Dict] = None
    last_cv_pdf_url: Optional[str] = None
    transcript: Optional[Dict] = None
    updated_at: Optional[datetime] = None

class InterviewQuestion(BaseModel):
    id: str
    type: str
    difficulty: str
    target_skill: str
    experience_ref: Optional[str] = None
    question: str
    answer_guide: str
