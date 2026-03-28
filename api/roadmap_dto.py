import json
from typing import Optional

from pydantic import BaseModel, model_validator


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


class RoadmapRequest(BaseModel):
    target_skill: TargetSkillDto
    current_level: CurrentLevelDto
    gap: GapDto

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
