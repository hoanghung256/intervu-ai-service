"""StrategyService — turn an assessment summary into ordered LearningMissions.

This is the deterministic middle layer between assessment and roadmap. It is
NOT an LLM call. Every transformation is pure code so we can unit-test "Backend
Middle, REST=2, EF=1, missing=Cache → 3 missions in this order" without spinning
up a model.

The roadmap LLM never sees raw assessment data anymore — it only sees the
mission list this service produces, plus a small narrative payload. That is the
core of the Phase 3 audit fix: the LLM narrates a pre-computed plan, it doesn't
design one.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from api.roadmap_dto import RoadmapRequest, SkillLevelDto
from application.services.competency_matrix_service import (
    CompetencyMatrix,
    CompetencyMatrixService,
    SkillCompetency,
    level_key_to_band,
    normalize_level_key,
)
from domain.entities.learning_mission import LearningMission


# How many characters of a question to keep when it lands as a "failed_question_excerpt".
# Long enough to be useful as context, short enough that 5-10 of these fit in a prompt.
_QUESTION_EXCERPT_LIMIT = 220

# Score threshold (0-100). Responses at/below this are treated as "failed" for
# the purpose of surfacing as excerpts. Tuned to catch genuinely weak answers
# without flagging every middling response.
_FAILED_SCORE_THRESHOLD = 50.0


def _parse_level(value: Any) -> int:
    if value is None:
        return 0
    try:
        return max(0, min(4, int(str(value).strip())))
    except (TypeError, ValueError):
        return 0


def _normalize_skill_key(skill: str) -> str:
    return re.sub(r"\s+", " ", str(skill or "")).strip().lower()


def _extract_responses(answer_json: Optional[Any]) -> List[Dict[str, Any]]:
    """Defensively pull a list of response dicts from whatever shape we got.

    The BE persists `AnswerJson` as a string blob; AI service callers may pass
    it pre-parsed. Either way we want a list of dicts with at least `skill`,
    `score`, `desc`, `confidence`, and `question` keys.
    """
    if answer_json is None:
        return []
    payload = answer_json
    if isinstance(payload, str):
        try:
            import json as _json
            payload = _json.loads(payload)
        except Exception:
            return []
    if not isinstance(payload, dict):
        return []
    responses = payload.get("responses")
    if isinstance(responses, list):
        return [r for r in responses if isinstance(r, dict)]
    return []


def _index_responses_by_skill(responses: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group responses by canonical skill key."""
    by_skill: Dict[str, List[Dict[str, Any]]] = {}
    for r in responses:
        key = _normalize_skill_key(r.get("skill"))
        if not key:
            continue
        by_skill.setdefault(key, []).append(r)
    return by_skill


def _theme_from_response(response: Dict[str, Any]) -> Optional[str]:
    """Distill a one-line interview theme from a response.

    Prefers the LLM-generated `desc` (Python evaluator emits this), falls back
    to a clipped question. We strip leading verbs like 'Candidate' so the theme
    reads as a topic, not a narration.
    """
    raw = response.get("desc") or response.get("description") or ""
    if isinstance(raw, str) and raw.strip():
        cleaned = re.sub(r"^(the candidate|candidate|user)\s+", "", raw.strip(), flags=re.IGNORECASE)
        # Cap to one sentence so we don't dump prose into the prompt.
        cleaned = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0]
        return cleaned[:160]
    question = response.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()[:160]
    return None


def _excerpt_from_response(response: Dict[str, Any]) -> Optional[str]:
    text = response.get("question")
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= _QUESTION_EXCERPT_LIMIT:
        return cleaned
    return cleaned[: _QUESTION_EXCERPT_LIMIT - 1].rstrip() + "…"


def _confidence_for_responses(responses: List[Dict[str, Any]]) -> float:
    confidences = []
    for r in responses:
        c = r.get("confidence")
        if isinstance(c, (int, float)) and 0 <= c <= 1:
            confidences.append(float(c))
    if not confidences:
        return 1.0
    return round(min(confidences), 3)


def _score_for_skill(
    skill_name: str,
    skill_index: Dict[str, SkillLevelDto],
    responses_for_skill: List[Dict[str, Any]],
) -> float:
    """Prefer the explicit per-skill score (Phase 1.1 plumbed this through).
    Fall back to averaging response.score when the skill DTO has score=0 (e.g.
    the BE didn't forward it on this call)."""
    key = _normalize_skill_key(skill_name)
    direct = skill_index.get(key)
    if direct is not None and direct.score:
        return float(direct.score)
    if not responses_for_skill:
        return 0.0
    nums = []
    for r in responses_for_skill:
        s = r.get("score")
        if isinstance(s, (int, float)):
            nums.append(float(s))
    if not nums:
        return 0.0
    return round(sum(nums) / len(nums), 1)


def _time_estimate(delta: int, weight: float) -> int:
    """Rough hours-to-target estimate. 8h per level × weight × delta.

    The constant is intentionally simple — we don't have telemetry yet to fit
    a better curve. Replace once we measure actual completion times."""
    return int(round(max(delta, 0) * max(weight, 0.0) * 8))


def _classify_status(current: int, target: int) -> str:
    if target <= 0 or current >= target:
        return "mastered" if current >= max(target, 1) else "weak"
    if current == 0:
        return "missing"
    return "weak"


class StrategyService:
    """Pure functions only — no LLM, no I/O beyond the matrix file load.

    Kept as a class so future state (telemetry, caches) has a home, but every
    public entry point is a `@staticmethod` today.
    """

    @staticmethod
    def build_learning_missions(
        request: RoadmapRequest,
        answer_json: Optional[Any] = None,
    ) -> Tuple[List[LearningMission], List[LearningMission]]:
        """Return `(active_missions, mastered_missions)`.

        - `active_missions` is sorted by priority_weight desc; mastered skills are
          excluded so the roadmap LLM never sees them as candidates for nodes.
        - `mastered_missions` is the ledger of "you already met target on these"
          for the FE to show as a "skills you have" badge.

        Two skills with identical priority_weight: `interview_critical=True`
        wins; further ties broken by larger `delta`, then by skill name for
        stability.
        """
        primary_role = (request.target_skill.roles or [""])[0] or ""
        target_level_str = request.target_skill.level or ""

        matrix: CompetencyMatrix = CompetencyMatrixService.get_matrix(primary_role, target_level_str)
        matrix_index: Dict[str, SkillCompetency] = {
            _normalize_skill_key(item.skill): item for item in matrix.skills
        }

        skill_index: Dict[str, SkillLevelDto] = {
            _normalize_skill_key(s.skill): s for s in (request.current_level.skills or []) if s.skill
        }

        responses = _extract_responses(answer_json)
        responses_by_skill = _index_responses_by_skill(responses)

        band_fallback = level_key_to_band(normalize_level_key(target_level_str))

        # Universe of skills to consider: every skill in current_level, every
        # gap entry, plus every matrix skill (so missing-from-assessment skills
        # the matrix expects still surface).
        candidate_keys: Dict[str, str] = {}  # key -> display name
        for s in request.current_level.skills or []:
            if s.skill:
                candidate_keys[_normalize_skill_key(s.skill)] = s.skill
        for gap_skill in (request.gap.weak or []) + (request.gap.missing or []):
            if gap_skill:
                candidate_keys.setdefault(_normalize_skill_key(gap_skill), gap_skill)
        for matrix_skill in matrix.skills:
            candidate_keys.setdefault(_normalize_skill_key(matrix_skill.skill), matrix_skill.skill)

        active: List[LearningMission] = []
        mastered: List[LearningMission] = []

        for key, display_name in candidate_keys.items():
            current = _parse_level(skill_index[key].level) if key in skill_index else 0

            matrix_entry = matrix_index.get(key)
            if matrix_entry is not None:
                target = matrix_entry.target_level
                weight = matrix_entry.weight
                interview_critical = matrix_entry.interview_critical
            else:
                target = band_fallback
                weight = 0.7
                interview_critical = False

            delta = max(target - current, 0)
            priority_weight = round(weight * delta, 3)
            status = _classify_status(current, target)

            responses_for_skill = responses_by_skill.get(key, [])
            score = _score_for_skill(display_name, skill_index, responses_for_skill)
            confidence = _confidence_for_responses(responses_for_skill)

            interview_themes: List[str] = []
            failed_excerpts: List[str] = []
            for r in responses_for_skill:
                theme = _theme_from_response(r)
                if theme and theme not in interview_themes:
                    interview_themes.append(theme)
                resp_score = r.get("score")
                if isinstance(resp_score, (int, float)) and float(resp_score) <= _FAILED_SCORE_THRESHOLD:
                    excerpt = _excerpt_from_response(r)
                    if excerpt and excerpt not in failed_excerpts:
                        failed_excerpts.append(excerpt)
            # Cap so the prompt doesn't bloat — top 3 themes / 3 excerpts is plenty
            # for an LLM to write a mentor_note.
            interview_themes = interview_themes[:3]
            failed_excerpts = failed_excerpts[:3]

            mission = LearningMission(
                skill_id=display_name,
                skill_name=display_name,
                current_level=current,
                target_level=target,
                delta=delta,
                weight=round(weight, 2),
                priority_weight=priority_weight,
                status=status,  # type: ignore[arg-type]
                interview_critical=interview_critical,
                score=score,
                confidence_in_assessment=confidence,
                time_estimate_hours=_time_estimate(delta, weight),
                interview_themes=interview_themes,
                failed_question_excerpts=failed_excerpts,
            )

            if mission.status == "mastered":
                mastered.append(mission)
            else:
                active.append(mission)

        active.sort(
            key=lambda m: (
                m.priority_weight,
                1 if m.interview_critical else 0,
                m.delta,
                # Negative score so lower scores (more remedial) sort earlier
                # within the same priority_weight band.
                -m.score,
                m.skill_name,
            ),
            reverse=True,
        )

        # Mastered list ordering doesn't drive logic; sort by name for stable output.
        mastered.sort(key=lambda m: m.skill_name)

        return active, mastered
