from pydantic import BaseModel
from typing import Optional, List, Dict, Union

class AskRequest(BaseModel):
    prompt: str

class AnswerResponse(BaseModel):
    answer: str

class AssessmentRequest(BaseModel):
    role: str
    level: str
    techstack: Union[List[str], str]
    domain: Union[List[str], str]
    free_text: str

    class Config:
        @staticmethod
        def alias_generator(string: str) -> str:
            parts = string.split("_")
            return parts[0] + "".join(p.title() for p in parts[1:])

        allow_population_by_field_name = True

class QuestionDto(BaseModel):
    Title: str
    Content: str

class SimilarityCheckRequest(BaseModel):
    SimilarMatchQuestionList: List[QuestionDto]
    Question: QuestionDto

class AssessmentResponse(BaseModel):
    status: str
    assessment: Optional[dict] = None
    question: Optional[str] = None
    context_question: Optional[str] = None
    phaseA: Optional[List[dict]] = None
    phaseB: Optional[List[dict]] = None

class CVResponse(BaseModel):
    apply_for: Optional[dict] = None
    skills: Optional[list] = None
    languages: Optional[list] = None
    experiences: Optional[list] = None
    certifications: Optional[list] = None
    total_years_of_experience: Optional[float] = None
    error: Optional[str] = None

class JDResponse(BaseModel):
    job_title: Optional[str] = None
    must_have_skills: Optional[list] = None
    nice_to_have_skills: Optional[list] = None
    required_yoe: Optional[float] = None
    core_responsibilities: Optional[list] = None
    benefits: Optional[list] = None
    company_culture: Optional[str] = None
    error: Optional[str] = None

class TranscriptResponse(BaseModel):
    status: str
    transcript: str
    question_list: List[dict]

class ExtractQuestionsRequest(BaseModel):
    transcript_text: str

class ExtractQuestionsResponse(BaseModel):
    status: str
    question_list: List[dict]
