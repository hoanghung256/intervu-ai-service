"""Role × level competency matrix.

Single source of truth for "what does a Backend Middle need?" — combines the
existing `role-skill-framework.json` (min_level per role/level) with the
`role-competency-overlay.json` sidecar (weight, interview_critical).

Used by:
- roadmap_service to compute per-skill target levels and priority weights
- (future) assessment_service to drive which skills are surveyed
- /api/competency-matrix endpoint, so the .NET BE can stop hardcoding skill
  scope in C#

Loaded once on first access and cached on the class.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRAMEWORK_PATH = _REPO_ROOT / "role-skill-framework.json"
_OVERLAY_PATH = _REPO_ROOT / "role-competency-overlay.json"


class SkillCompetency(BaseModel):
    """A single skill expectation for a (role, level) pair."""

    skill: str
    target_level: int
    weight: float = 0.7
    interview_critical: bool = False


class CompetencyMatrix(BaseModel):
    role_key: str
    level_key: str
    skills: List[SkillCompetency]


def _normalize(text: Optional[str]) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_role_key(role: str, framework: Dict[str, Any]) -> str:
    """Mirror of `AssessmentService._normalize_role_key` so the matrix accepts
    the same free-form role strings the assessment already understands."""
    role_normalized = _normalize(role).lower()
    role_aliases = framework.get("role_aliases", {}) if isinstance(framework, dict) else {}
    role_definitions = framework.get("role_definitions", {}) if isinstance(framework, dict) else {}

    if role_normalized in role_definitions:
        return role_normalized
    if role_normalized in role_aliases:
        return role_aliases[role_normalized]

    for alias, canonical in role_aliases.items():
        if alias in role_normalized:
            return canonical

    if "front" in role_normalized:
        return "frontend_engineer"
    if "devops" in role_normalized or "sre" in role_normalized:
        return "devops_engineer"
    if "qa" in role_normalized or "test" in role_normalized:
        return "qa_engineer"
    if "architect" in role_normalized:
        return "solution_architect"
    if "data" in role_normalized or "database" in role_normalized:
        return "database_designer"
    return "backend_engineer"


def normalize_level_key(level: str) -> str:
    """Mirror of `AssessmentService._normalize_level_key`."""
    lvl = _normalize(level).lower()
    if not lvl:
        return "junior"
    if any(k in lvl for k in ["fresher", "entry", "intern", "new grad", "beginner"]):
        return "fresher"
    if any(k in lvl for k in ["junior", "basic"]):
        return "junior"
    if any(k in lvl for k in ["middle", "mid", "intermediate"]):
        return "middle"
    if any(k in lvl for k in ["senior", "lead", "staff", "principal", "advanced", "expert"]):
        return "senior"
    return "junior"


# Numeric mapping for the Phase 1 deterministic-progress formula. Junior=2 etc.
_LEVEL_TO_BAND = {"fresher": 1, "junior": 2, "middle": 3, "senior": 4}


def level_key_to_band(level_key: str) -> int:
    """Numeric target band for a normalized level key (1-4)."""
    return _LEVEL_TO_BAND.get(level_key, 3)


class CompetencyMatrixService:
    _framework: Optional[Dict[str, Any]] = None
    _overlay: Optional[Dict[str, Any]] = None

    @classmethod
    def _load_json(cls, path: Path) -> Dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning("Competency matrix file missing: %s", path)
            return {}
        except Exception as ex:
            logging.error("Failed to load competency matrix file %s: %s", path, ex)
            return {}

    @classmethod
    def _framework_data(cls) -> Dict[str, Any]:
        if cls._framework is None:
            cls._framework = cls._load_json(_FRAMEWORK_PATH)
        return cls._framework or {}

    @classmethod
    def _overlay_data(cls) -> Dict[str, Any]:
        if cls._overlay is None:
            cls._overlay = cls._load_json(_OVERLAY_PATH)
        return cls._overlay or {}

    @classmethod
    def get_matrix(cls, role: str, level: str) -> CompetencyMatrix:
        """Return the competency matrix slice for a (role, level) pair.

        Falls back across (a) the requested role/level, (b) sibling levels for
        the same role, (c) backend_engineer as the catch-all role. Always
        returns a CompetencyMatrix — never raises.
        """
        framework = cls._framework_data()
        overlay = cls._overlay_data()

        role_key = normalize_role_key(role, framework)
        level_key = normalize_level_key(level)

        role_def = (framework.get("role_definitions") or {}).get(role_key) or {}
        skills_for_level = role_def.get(level_key)
        if not isinstance(skills_for_level, list) or not skills_for_level:
            for fallback_level in ("junior", "middle", "senior", "fresher"):
                fallback = role_def.get(fallback_level)
                if isinstance(fallback, list) and fallback:
                    skills_for_level = fallback
                    level_key = fallback_level
                    break
            else:
                skills_for_level = []

        defaults = (overlay.get("defaults") or {}) if isinstance(overlay, dict) else {}
        default_weight = float(defaults.get("weight", 0.7))
        default_critical = bool(defaults.get("interview_critical", False))

        overlay_for_role = ((overlay.get("overlays") or {}).get(role_key) or {}) if isinstance(overlay, dict) else {}
        overlay_for_level = overlay_for_role.get(level_key) or {}

        items: List[SkillCompetency] = []
        for entry in skills_for_level:
            if not isinstance(entry, dict):
                continue
            skill_name = _normalize(entry.get("skill"))
            if not skill_name:
                continue
            target_level = int(entry.get("min_level") or 1)
            weight = default_weight
            interview_critical = default_critical
            ov = overlay_for_level.get(skill_name)
            if isinstance(ov, dict):
                if "weight" in ov:
                    try:
                        weight = max(0.0, min(1.0, float(ov["weight"])))
                    except (TypeError, ValueError):
                        pass
                if "interview_critical" in ov:
                    interview_critical = bool(ov["interview_critical"])
            items.append(
                SkillCompetency(
                    skill=skill_name,
                    target_level=target_level,
                    weight=weight,
                    interview_critical=interview_critical,
                )
            )

        return CompetencyMatrix(role_key=role_key, level_key=level_key, skills=items)

    @classmethod
    def lookup(cls, role: str, level: str, skill: str) -> Optional[SkillCompetency]:
        """Find one skill's competency expectation. Returns None if absent."""
        skill_normalized = _normalize(skill).lower()
        if not skill_normalized:
            return None
        for item in cls.get_matrix(role, level).skills:
            if item.skill.lower() == skill_normalized:
                return item
        return None
