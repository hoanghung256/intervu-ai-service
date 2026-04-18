from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

class SkillOption(BaseModel):
    text: str
    level: str

class SkillQuestion(BaseModel):
    skill: str
    question: str
    options: List[SkillOption]

class Assessment(BaseModel):
    contextQuestion: str
    phaseA: List[SkillQuestion]
    phaseB: List[SkillQuestion]


class SurveyAnswerProfile(BaseModel):
    role: str = ""
    level: str = ""
    techstack: List[str] = Field(default_factory=list)
    domain: List[str] = Field(default_factory=list)
    freeText: str = ""


class SurveyAnswerResponse(BaseModel):
    questionId: str = ""
    question: str = ""
    phase: str = ""
    skill: str = ""
    answer: str = ""
    selectedLevel: str = ""


class SurveyAnswerJson(BaseModel):
    profile: SurveyAnswerProfile = Field(default_factory=SurveyAnswerProfile)
    responses: List[SurveyAnswerResponse] = Field(default_factory=list)


class ResponseItem(BaseModel):
    phase: str = ""
    skill: str = ""
    selectedLevel: str = ""


class SurveyTarget(BaseModel):
    roles: List[str] = Field(default_factory=list)
    level: str = ""
    skillsTarget: List[str] = Field(default_factory=list)


class SurveySkillLevel(BaseModel):
    skill: str = ""
    level: str = ""
    sfiaLevel: Optional[int] = None


class SurveyCurrent(BaseModel):
    skills: List[SurveySkillLevel] = Field(default_factory=list)


class SurveyGap(BaseModel):
    missing: List[str] = Field(default_factory=list)
    weak: List[str] = Field(default_factory=list)


class SurveyRoadmap(BaseModel):
    phases: List[Dict[str, Any]] = Field(default_factory=list)
    roadmap_metadata: Dict[str, Any] = Field(default_factory=dict)


class SurveyResponses(BaseModel):
    userId: Optional[str] = None
    assessmentName: str = ""
    responses: List[ResponseItem] = Field(default_factory=list)
    target: Optional[SurveyTarget] = None
    current: Optional[SurveyCurrent] = None
    gap: Optional[SurveyGap] = None
    roadmap: Optional[SurveyRoadmap] = None
    answer: Optional[SurveyAnswerJson] = None


class SurveySummaryResult(BaseModel):
    userId: Optional[str] = None
    summaryText: str = ""
    summaryObject: Dict[str, Any] = Field(default_factory=dict)
