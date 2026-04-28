import json
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from api.dtos import LLMUsage


class TargetSkillDto(BaseModel):
    level: str
    roles: list[str]
    skillsTarget: list[str]


class SkillLevelDto(BaseModel):
    level: str
    skill: str
    sfiaLevel: int
    score: float = 0.0


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
    # Optional pass-through of the assessment AnswerJson (per-question data:
    # score, desc/rationale, confidence, etc.). When present, the strategy
    # service uses it to populate per-mission interview_themes and
    # failed_question_excerpts. When absent (legacy callers), missions are still
    # built — they just lack the per-question colour.
    answer_json: Optional[dict] = None

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


# ── Roadmap response schema (Phase 4) ────────────────────────────────────
#
# These models exist purely for VALIDATION of the LLM output. They are loose
# where they have to be (extra fields, mixed-shape child_skills) and strict
# where it matters (skill_id, status enum, progress 0-100). The endpoint still
# returns `dict` for backward compatibility — the FE shape doesn't change.


class RoadmapNodeAssessmentSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    current_level: str = ""
    target_level: str = ""
    sfia_level: int = 0
    status: Literal["Weak", "Missing", "Complete"] = "Weak"
    progress: int = Field(default=0, ge=0, le=100)
    score: float = 0.0


class RoadmapChildSkillSchema(BaseModel):
    """Child skill — accept either a bare string or `{name, ...}` dict.
    The downstream snapshot mapper already tolerates both shapes; we
    normalize here so validation doesn't reject either."""

    model_config = ConfigDict(extra="allow")

    name: str = ""


class RoadmapNodeSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    skill_id: str
    skill_name: str = ""
    assessment: RoadmapNodeAssessmentSchema = Field(default_factory=RoadmapNodeAssessmentSchema)
    child_skills: List[Union[str, RoadmapChildSkillSchema]] = Field(default_factory=list)
    mentor_note: str = ""
    interview_drills: List[str] = Field(default_factory=list)


class RoadmapPhaseSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    phase_id: str
    phase_name: str
    phase_description: str = ""
    nodes: List[RoadmapNodeSchema] = Field(default_factory=list)


class RoadmapMetadataSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    target_role: str = ""
    target_level: str = ""
    total_phases: int = 0


class RoadmapDocumentSchema(BaseModel):
    """Top-level shape an LLM response is validated against. Anything outside
    these fields is accepted (extra="allow" on phases/nodes) because we do
    NOT want to drop coach recommendations the recommender attaches later."""

    model_config = ConfigDict(extra="allow")

    roadmap_metadata: RoadmapMetadataSchema = Field(default_factory=RoadmapMetadataSchema)
    phases: List[RoadmapPhaseSchema] = Field(default_factory=list)


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
