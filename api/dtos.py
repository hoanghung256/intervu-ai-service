from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict, Union

class AskRequest(BaseModel):
    prompt: str

class AnswerResponse(BaseModel):
    answer: str

class AssessmentRequest(BaseModel):
    role: Optional[str] = ""
    level: Optional[str] = ""
    techstack: Optional[Union[List[str], str]] = None
    domain: Optional[Union[List[str], str]] = None
    # selected_options: Optional[List[str]] = Field(default_factory=list)
    free_text: Optional[str] = None

    class Config:
        @staticmethod
        def alias_generator(string: str) -> str:
            parts = string.split("_")
            return parts[0] + "".join(p.title() for p in parts[1:])

        validate_by_name = True

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
    tags: Optional[List[str]] = Field(default_factory=list)

class ExtractQuestionsRequest(BaseModel):
    transcript_text: str
    tags: Optional[List[str]] = Field(default_factory=list)

class ExtractQuestionsResponse(BaseModel):
    status: str
    question_list: List[dict]
    tags: Optional[List[str]] = Field(default_factory=list)
    
class CvEvaluationResponse(BaseModel):
    status: str
    summary: str
    strengths: List[str]
    gaps: List[str]
    reasoning: str
    final_verdict: str


class SurveyAnswerProfileDto(BaseModel):
    role: str = ""
    level: str = ""
    techstack: List[str] = Field(default_factory=list)
    domain: List[str] = Field(default_factory=list)
    freeText: str = ""


class SurveyAnswerResponseDto(BaseModel):
    questionId: str = ""
    question: str = ""
    phase: str = ""
    skill: str = ""
    answer: str = ""
    selectedLevel: str = ""


class SurveyAnswerJsonDto(BaseModel):
    profile: SurveyAnswerProfileDto = Field(default_factory=SurveyAnswerProfileDto)
    responses: List[SurveyAnswerResponseDto] = Field(default_factory=list)


class ResponseItemDto(BaseModel):
    phase: str = ""
    skill: str = ""
    selectedLevel: str = ""


class SurveyTargetDto(BaseModel):
    roles: List[str] = Field(default_factory=list)
    level: str = ""
    skillsTarget: List[str] = Field(default_factory=list)


class SurveySkillLevelDto(BaseModel):
    skill: str = ""
    level: str = ""
    sfiaLevel: Optional[int] = None


class SurveyCurrentDto(BaseModel):
    skills: List[SurveySkillLevelDto] = Field(default_factory=list)


class SurveyGapDto(BaseModel):
    missing: List[str] = Field(default_factory=list)
    weak: List[str] = Field(default_factory=list)


class SurveyRoadmapDto(BaseModel):
    phases: List[Dict[str, Any]] = Field(default_factory=list)
    roadmap_metadata: Dict[str, Any] = Field(default_factory=dict)


class SurveyResponsesDto(BaseModel):
    userId: Optional[str] = None
    assessmentName: str = ""
    responses: List[ResponseItemDto] = Field(default_factory=list)
    target: Optional[SurveyTargetDto] = None
    current: Optional[SurveyCurrentDto] = None
    gap: Optional[SurveyGapDto] = None
    roadmap: Optional[SurveyRoadmapDto] = None
    answer: Optional[SurveyAnswerJsonDto] = None

    class Config:
        @staticmethod
        def alias_generator(string: str) -> str:
            parts = string.split("_")
            return parts[0] + "".join(p.title() for p in parts[1:])

        validate_by_name = True


class SurveySummaryResultDto(BaseModel):
    userId: Optional[str] = None
    summaryText: str = ""
    answer: Dict[str, Any] = Field(default_factory=dict)
    target: Dict[str, Any] = Field(default_factory=dict)
    current: Dict[str, Any] = Field(default_factory=dict)
    gap: Dict[str, Any] = Field(default_factory=lambda: {"weak": [], "missing": []})


class EvaluateAssessmentRequestDto(BaseModel):
    answer: SurveyAnswerJsonDto
    target: Optional[Dict[str, Any]] = None
    targetJson: Optional[Dict[str, Any]] = None
    targetjson: Optional[Dict[str, Any]] = None
    gap: Optional[Dict[str, Any]] = None
    gapJson: Optional[Dict[str, Any]] = None
    gapjson: Optional[Dict[str, Any]] = None
