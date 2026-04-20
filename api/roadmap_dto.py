import json
from typing import Optional

from pydantic import BaseModel, model_validator
from api.dtos import LLMUsage


class TargetSkillDto(BaseModel):
    level: str
    roles: list[str]
    skillsTarget: list[str]


class SkillLevelDto(BaseModel):
    level: str
    skill: str
    sfiaLevel: int


class CurrentLevelDto(BaseModel):
    skills: list[SkillLevelDto]


class GapDto(BaseModel):
    weak: list[str]
    missing: list[str]


class CoachServiceEntry(BaseModel):
    id: str
    interview_type_name: str
    price: int = 0
    duration_minutes: int = 60
    aim_level_hint: Optional[str] = ""


class CoachCatalogEntry(BaseModel):
    id: str
    name: str
    slug_profile_url: Optional[str] = ""
    avatar_url: Optional[str] = ""
    skills: list[str] = []
    bio: Optional[str] = ""
    services: list[CoachServiceEntry] = []


class RoadmapRequest(BaseModel):
    target_skill: TargetSkillDto
    current_level: CurrentLevelDto
    gap: GapDto
    coach_catalog: list[CoachCatalogEntry] = []

    @model_validator(mode="before")
    @classmethod
    def parse_stringified_body(cls, data):
        # Some clients double-serialize JSON and send the request body as a string.
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError as exc:
                raise ValueError("Request body must be a valid JSON object") from exc
        return data


class RoadmapResponse(BaseModel):
    status: str
    roadmap: Optional[dict] = None
    error: Optional[str] = None
    usage: Optional[LLMUsage] = None


class EvaluationItemDto(BaseModel):
    type: str
    score: int
    question: Optional[str] = ""
    answer: Optional[str] = ""


class RoadmapProgressUpdateRequest(BaseModel):
    current_roadmap: dict
    interview_type: str
    aim_level: Optional[str] = ""
    evaluation: list[EvaluationItemDto]
    target_node_id: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def parse_stringified_body(cls, data):
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError as exc:
                raise ValueError("Request body must be a valid JSON object") from exc
        return data
