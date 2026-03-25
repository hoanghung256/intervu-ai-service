from pydantic import BaseModel
from typing import List, Optional

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
