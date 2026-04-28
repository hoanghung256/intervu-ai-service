"""LearningMission — the deterministic input the roadmap LLM consumes.

A mission is one (skill, role, level) triple expressed as a unit of work the
candidate needs to complete to reach interview-ready. Built by
`StrategyService.build_learning_missions()` from the assessment summary and the
competency matrix, never by an LLM.

The roadmap LLM groups missions into phases and writes mentor copy on top of
them — it does NOT decide which skills exist, which are mastered, which are
weak, or how big the gap is. That work happens here, in code, where it can be
unit-tested.
"""

from typing import List, Literal

from pydantic import BaseModel, Field


MissionStatus = Literal["mastered", "weak", "missing"]


class LearningMission(BaseModel):
    skill_id: str
    skill_name: str
    current_level: int
    target_level: int
    delta: int
    """Levels of growth needed (target_level - current_level), clamped at 0."""

    weight: float
    """Matrix weight (0-1): how important this skill is for the role/level."""

    priority_weight: float
    """weight × delta — the canonical sort key for "what to learn first"."""

    status: MissionStatus
    interview_critical: bool
    score: float = 0.0
    """0-100 from the assessment. Lower = more remedial work needed."""

    confidence_in_assessment: float = 1.0
    """Min LLM scoring confidence across responses for this skill (0-1).
    1.0 means we're sure about the level; lower means recommend re-assessment."""

    time_estimate_hours: int = 0
    """Heuristic: round(delta × weight × 8). Refine later if we measure actuals."""

    interview_themes: List[str] = Field(default_factory=list)
    """Short phrases extracted from per-question AI rationale (response.desc).
    Empty when the assessment didn't carry per-question data."""

    failed_question_excerpts: List[str] = Field(default_factory=list)
    """Question texts for low-scoring responses on this skill — feeds the
    roadmap LLM's mentor_note so it can reference what the candidate actually
    got wrong instead of generic advice."""
