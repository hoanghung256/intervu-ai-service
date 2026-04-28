import json
import logging
import re
from typing import Optional

from pydantic import ValidationError

from api.roadmap_dto import CoachCatalogEntry, RoadmapDocumentSchema, RoadmapRequest
from application.services.strategy_service import StrategyService
from domain.entities.learning_mission import LearningMission
from infrastructure.model_provider.llm_provider import LLMProvider
from infrastructure.model_provider.model_constants import HUGGINGFACE_LLAMA_3_3_70B_INSTRUCT_GROQ


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _normalize_skill_key(skill: str) -> str:
    return (skill or "").strip().lower()


def _map_to_sfia(level: int) -> int:
    """Mirror BE's `MapToSfia` so node.assessment.sfia_level stays consistent
    across services without requiring the BE to forward it. Levels: 0→0,
    1→2, 2→3, 3→5, 4→6."""
    if level <= 0:
        return 0
    if level == 1:
        return 2
    if level == 2:
        return 3
    if level == 3:
        return 5
    return 6


def _build_mission_index(missions: list[LearningMission]) -> dict[str, LearningMission]:
    return {_normalize_skill_key(m.skill_id): m for m in missions}


def _normalize_assessment_block(roadmap: dict, missions: list[LearningMission]) -> dict:
    """Override every node's assessment block from the matching mission.

    `progress = round(min(current/target, 1) * 100)`. The LLM may emit any
    placeholder values; we replace them. Nodes that don't match any mission
    are left untouched (the coercion step will have warned about them).
    """
    mission_index = _build_mission_index(missions)

    for phase in roadmap.get("phases", []) or []:
        for node in phase.get("nodes", []) or []:
            mission = mission_index.get(_normalize_skill_key(node.get("skill_id") or ""))
            if mission is None:
                continue

            assessment = node.setdefault("assessment", {})
            assessment["current_level"] = str(mission.current_level)
            assessment["target_level"] = str(mission.target_level)
            assessment["sfia_level"] = _map_to_sfia(mission.current_level)
            assessment["score"] = round(mission.score, 1)

            ratio = (mission.current_level / mission.target_level) if mission.target_level > 0 else 0.0
            ratio = max(0.0, min(1.0, ratio))
            assessment["progress"] = round(ratio * 100)

            if mission.current_level >= mission.target_level and mission.target_level > 0:
                assessment["status"] = "Complete"
            elif mission.current_level == 0:
                assessment["status"] = "Missing"
            else:
                assessment["status"] = "Weak"

    return roadmap


def _mission_to_prompt_dict(mission: LearningMission) -> dict:
    """Trim a LearningMission to the fields the LLM actually needs in the
    prompt. Excludes book-keeping like `confidence_in_assessment` to keep tokens down."""
    payload = {
        "skill_id": mission.skill_id,
        "current_level": mission.current_level,
        "target_level": mission.target_level,
        "delta": mission.delta,
        "priority_weight": mission.priority_weight,
        "interview_critical": mission.interview_critical,
        "score": mission.score,
        "time_estimate_hours": mission.time_estimate_hours,
    }
    if mission.interview_themes:
        payload["interview_themes"] = mission.interview_themes
    if mission.failed_question_excerpts:
        payload["failed_question_excerpts"] = mission.failed_question_excerpts
    return payload


def _render_missions_for_prompt(missions: list[LearningMission]) -> str:
    if not missions:
        return "(no active missions — candidate already meets every target)"
    return "\n".join(json.dumps(_mission_to_prompt_dict(m), ensure_ascii=False) for m in missions)


def _render_mastered_for_prompt(mastered: list[LearningMission]) -> str:
    if not mastered:
        return "(none)"
    return ", ".join(m.skill_name for m in mastered)


def _attach_mastered_summary(roadmap: dict, mastered: list[LearningMission]) -> dict:
    """Surface mastered skills as a top-level summary so the FE can render a
    "skills you already have" section without re-deriving it."""
    roadmap["mastered_summary"] = [
        {
            "skill_id": m.skill_id,
            "skill_name": m.skill_name,
            "current_level": m.current_level,
            "target_level": m.target_level,
        }
        for m in mastered
    ]
    return roadmap


def _tokenize(text: Optional[str]) -> set[str]:
    if not text:
        return set()
    return {t for t in _TOKEN_PATTERN.findall(text.lower()) if len(t) > 1}


def _node_tokens(node: dict) -> set[str]:
    parts: list[str] = [node.get("skill_name", ""), node.get("skill_id", "")]
    for child in node.get("child_skills", []) or []:
        if isinstance(child, str):
            parts.append(child)
        elif isinstance(child, dict):
            parts.append(child.get("name", ""))
    return _tokenize(" ".join(parts))


def _coach_tokens(coach: CoachCatalogEntry) -> set[str]:
    parts = [coach.name or "", coach.bio or ""]
    parts.extend(coach.skills or [])
    for service in coach.services or []:
        parts.append(service.interview_type_name or "")
    return _tokenize(" ".join(parts))


def _score(node_tokens: set[str], coach_tokens: set[str]) -> float:
    if not node_tokens or not coach_tokens:
        return 0.0
    overlap = len(node_tokens & coach_tokens)
    # Jaccard-like, but weighted toward node token coverage so a broadly-skilled
    # coach doesn't automatically win every node.
    return overlap / (len(node_tokens) + 1e-6)


def _pick_service(coach: CoachCatalogEntry, node_tokens: set[str], target_level: str) -> Optional[dict]:
    if not coach.services:
        return None

    target = (target_level or "").strip().lower()
    best = None
    best_score = -1.0
    for service in coach.services:
        type_tokens = _tokenize(service.interview_type_name)
        overlap = len(type_tokens & node_tokens)
        aim = (service.aim_level_hint or "").strip().lower()
        level_bonus = 1 if aim and target and aim == target else 0
        score = overlap * 2 + level_bonus
        if score > best_score:
            best_score = score
            best = service

    if best is None:
        best = coach.services[0]

    return {
        "id": best.id,
        "interview_type_name": best.interview_type_name,
        "price": best.price,
        "duration_minutes": best.duration_minutes,
    }


def _attach_recommendations(roadmap: dict, catalog: list[CoachCatalogEntry]) -> dict:
    if not catalog:
        return roadmap

    prepared = [(c, _coach_tokens(c)) for c in catalog if c.services]
    if not prepared:
        return roadmap

    for phase in roadmap.get("phases", []) or []:
        for node in phase.get("nodes", []) or []:
            n_tokens = _node_tokens(node)
            ranked = sorted(
                prepared,
                key=lambda pair: _score(n_tokens, pair[1]),
                reverse=True,
            )
            coach, _ = ranked[0]
            target_level = (node.get("assessment") or {}).get("target_level", "")
            service = _pick_service(coach, n_tokens, target_level)
            if service is None:
                continue
            node["recommended_coach"] = {
                "id": coach.id,
                "name": coach.name,
                "slug_profile_url": coach.slug_profile_url or "",
                "avatar_url": coach.avatar_url or "",
            }
            node["recommended_service"] = service

    return roadmap


def _validate_roadmap(
    roadmap: dict,
    active_missions: list[LearningMission],
    mastered_missions: list[LearningMission],
) -> list[str]:
    """Validate an LLM roadmap. Returns a list of human-readable error strings.
    Empty list means the roadmap is acceptable.

    Two layers:
    1. Pydantic structural validation against `RoadmapDocumentSchema`. This
       catches missing fields, wrong types, invalid status enum values, etc.
    2. Semantic checks that Pydantic can't express:
       - Every node.skill_id must match an active mission (post-coercion).
       - No node may reference a mastered skill.
       - At least one phase must exist when active missions exist.

    The progress/status formula is NOT validated because we overwrite those
    values deterministically in `_normalize_assessment_block` after this step.
    """
    errors: list[str] = []

    try:
        RoadmapDocumentSchema.model_validate(roadmap)
    except ValidationError as ex:
        # Pydantic gives us a list of dicts; format each as one error line.
        for issue in ex.errors():
            loc = ".".join(str(p) for p in issue.get("loc", ()))
            errors.append(f"schema {loc}: {issue.get('msg', 'invalid')}")
        # If the structure is broken, semantic checks are unreliable — bail early.
        return errors

    active_ids = {_normalize_skill_key(m.skill_id) for m in active_missions}
    mastered_ids = {_normalize_skill_key(m.skill_id) for m in mastered_missions}

    phases = roadmap.get("phases") or []

    if active_missions and not phases:
        errors.append("missing phases: there are active learning missions but the roadmap has no phases")

    seen_node_ids: set[str] = set()
    for phase_idx, phase in enumerate(phases):
        for node_idx, node in enumerate(phase.get("nodes") or []):
            node_path = f"phases[{phase_idx}].nodes[{node_idx}]"
            skill_id = (node.get("skill_id") or "").strip()
            if not skill_id:
                errors.append(f"{node_path}: skill_id is empty")
                continue

            key = _normalize_skill_key(skill_id)
            if key in mastered_ids:
                errors.append(
                    f"{node_path}: skill_id {skill_id!r} is in the mastered list and must NOT appear as a node"
                )
            elif key not in active_ids:
                errors.append(
                    f"{node_path}: skill_id {skill_id!r} does not match any active mission"
                )

            if key in seen_node_ids:
                errors.append(f"{node_path}: skill_id {skill_id!r} is duplicated across nodes")
            else:
                seen_node_ids.add(key)

    return errors


def _coerce_skill_ids(roadmap: dict, missions: list[LearningMission]) -> None:
    """Best-effort: ensure every node's `skill_id` matches an active mission.

    The LLM is instructed to use canonical mission ids, but it occasionally
    drifts. We try to repair the link by matching the LLM's `skill_id` or
    `skill_name` to the mission list with a case-insensitive containment
    heuristic. Drifted nodes that can't be repaired are kept (with a logged
    warning) so the user still gets a roadmap; the schema-level validation in
    Phase 4 will fail closed instead.
    """
    known_lookup = {_normalize_skill_key(m.skill_id): m.skill_id for m in missions}

    def _resolve(candidate: str) -> Optional[str]:
        if not candidate:
            return None
        key = candidate.strip().lower()
        if key in known_lookup:
            return known_lookup[key]
        # Containment fallback: mission id appears inside the LLM's id, or vice versa.
        for known_key, known_value in known_lookup.items():
            if known_key in key or key in known_key:
                return known_value
        return None

    for phase in roadmap.get("phases", []) or []:
        for node in phase.get("nodes", []) or []:
            current_id = node.get("skill_id") or ""
            resolved = _resolve(current_id) or _resolve(node.get("skill_name") or "")
            if resolved:
                node["skill_id"] = resolved
            else:
                logging.warning(
                    "Roadmap node skill_id %r could not be matched to any active mission",
                    current_id,
                )


class RoadmapService:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def generate_roadmap(self, request: RoadmapRequest) -> dict:
        # Phase 3: assessment data is collapsed into a deterministic mission list
        # before we even talk to the LLM. The model narrates this plan; it does
        # not design one. Mastered skills are removed from the input entirely so
        # the LLM has no opportunity to add nodes for things the candidate
        # already knows.
        active_missions, mastered_missions = StrategyService.build_learning_missions(
            request, request.answer_json
        )

        rendered_missions = _render_missions_for_prompt(active_missions)
        rendered_mastered = _render_mastered_for_prompt(mastered_missions)

        target_role = (request.target_skill.roles or [""])[0] or ""
        target_level = request.target_skill.level or ""

        base_prompt = """
# ROLE
You are a Senior Career Architect. Your job is to NARRATE a pre-computed
learning plan, not to design one. The candidate's gap analysis, target levels,
priority weights, and time estimates have already been calculated. You group
the missions into phases and write the human-facing copy.

# INPUT
You receive:
1. `target_role` and `target_level` — career band the candidate is reaching for.
2. `active_missions` — JSON-per-line list, sorted by `priority_weight` desc.
   Each mission is a skill the candidate has NOT yet met target on. Fields:
   - `skill_id` (canonical — use verbatim, do not rename)
   - `current_level`, `target_level`, `delta` (target - current)
   - `priority_weight` (= matrix_weight × delta) and `interview_critical`
   - `score` (0-100; lower = needs more remediation)
   - `time_estimate_hours`
   - `interview_themes` (optional, max 3): one-line topics the candidate
     struggled with — use these in mentor notes.
   - `failed_question_excerpts` (optional, max 3): actual interview questions
     they fumbled — quote or paraphrase in mentor_note when relevant.
3. `mastered_skills` — comma-separated list. Already at/above target. NEVER
   create nodes for these.

# YOUR JOB
1. Group active missions into 3-5 phases:
   - Phase 1: "Foundations & Remediation" — the highest-priority_weight missions
     and any mission with `interview_critical: true` that has a non-trivial gap.
   - Phase 2-3: "Core Tech & Specialized" — remaining gaps in priority order.
   - Phase 4 (optional): "Production Readiness & Career Simulation" — only if
     remaining missions are about deployment/testing/observability.
   Preserve priority order WITHIN each phase. Highest priority_weight first.
2. Each phase: max 4 nodes. Consolidate related missions if needed (e.g. group
   "HTML"/"CSS"/"Responsive UI" under one parent node) — the parent's
   `skill_id` is the highest-priority mission, the rest become `child_skills`.
3. For each node:
   - Use the mission's `skill_id` verbatim. Inventing ids is forbidden.
   - Write a `mentor_note` (1-2 sentences) that references the candidate's
     `interview_themes` or `failed_question_excerpts` when present. If neither
     is present, give targeted general advice based on `score` and `delta`.
   - If applicable, list `interview_drills` (1-3 short bullet points) — concrete
     things to practice for this skill in interviews. Prioritize drills that
     directly address the failed_question_excerpts.

# OUTPUT FORMAT (MANDATORY JSON)
Return ONLY valid JSON, no preamble or commentary. The `assessment` block's
`current_level`, `target_level`, `sfia_level`, `status`, and `progress` will
be overwritten downstream from the mission data — emit any plausible values.
{
  "roadmap_metadata": {
    "target_role": "string",
    "target_level": "string",
    "total_phases": number
  },
  "phases": [
    {
      "phase_id": "p1",
      "phase_name": "string",
      "phase_description": "1-2 sentences: why this phase, what unlocks after it",
      "nodes": [
        {
          "skill_id": "string (verbatim mission skill_id)",
          "skill_name": "string (broad domain — may differ from skill_id when consolidating)",
          "assessment": {
            "current_level": "string",
            "target_level": "string",
            "sfia_level": 0,
            "status": "Weak",
            "progress": 0
          },
          "child_skills": ["Detailed sub-skill 1", "Detailed sub-skill 2"],
          "mentor_note": "1-2 sentences referencing the candidate's interview_themes / failed_question_excerpts when present.",
          "interview_drills": ["Drill A", "Drill B"]
        }
      ]
    }
  ]
}

# TONE
Professional, specific, and grounded in the candidate's actual evidence — not
generic advice. If a mission has interview_themes, reference them. If not, fall
back to score-driven framing ("you scored 35/100 — start with X").
"""

        request_payload = f"""

# CONTEXT
target_role: {target_role}
target_level: {target_level}

# active_missions (one JSON object per line; sorted by priority_weight desc — DO NOT REORDER ACROSS PHASES)
{rendered_missions}

# mastered_skills (DO NOT create nodes for these)
{rendered_mastered}

# FINAL REMINDER
- Use ONLY skill_ids that appear in active_missions (verbatim).
- Mastered skills MUST NOT appear anywhere in the output.
- Highest-priority_weight missions go into the earliest phase.
- mentor_note should reference interview_themes / failed_question_excerpts when the mission has them.
"""

        prompt = base_prompt + request_payload

        # Phase 4: call the LLM, validate, retry once with the validation errors
        # appended to the prompt. Fail closed (ValueError → status=failed) if the
        # second attempt also has unresolvable issues — better a regenerate
        # button on the FE than a silently-wrong roadmap.
        roadmap, usage = await self._generate_validated_roadmap(
            base_prompt=prompt,
            active_missions=active_missions,
            mastered_missions=mastered_missions,
        )

        _normalize_assessment_block(roadmap, active_missions)
        _attach_mastered_summary(roadmap, mastered_missions)

        try:
            roadmap = _attach_recommendations(roadmap, request.coach_catalog or [])
        except Exception as ex:
            logging.warning(f"Failed to attach per-node recommendations: {ex}")

        return roadmap, usage

    async def _generate_validated_roadmap(
        self,
        base_prompt: str,
        active_missions: list[LearningMission],
        mastered_missions: list[LearningMission],
        max_retries: int = 1,
    ) -> tuple[dict, dict]:
        """Run the LLM, parse, coerce skill ids, validate. On validation failure,
        retry once with the error list appended to the prompt. After retries are
        exhausted, raise ValueError with the residual issues so the endpoint can
        return status=failed and the FE can offer a regenerate button.

        Returns the validated `(roadmap_dict, llm_usage)`. Total token usage
        across the original + retry call is summed so cost reporting stays
        truthful.
        """
        prompt = base_prompt
        accumulated_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        last_errors: list[str] = []

        for attempt in range(max_retries + 1):
            response_text, usage = await self.llm_provider.generate_content(
                prompt, model=HUGGINGFACE_LLAMA_3_3_70B_INSTRUCT_GROQ
            )
            if isinstance(usage, dict):
                for k in accumulated_usage:
                    accumulated_usage[k] += int(usage.get(k, 0))

            try:
                cleaned_json = self.llm_provider.clean_json_string(response_text)
                roadmap = json.loads(cleaned_json)
            except Exception as ex:
                logging.warning(
                    "Roadmap attempt %d: model returned non-JSON output: %s", attempt + 1, ex
                )
                last_errors = [f"Output was not valid JSON: {ex}"]
                if attempt < max_retries:
                    prompt = self._build_retry_prompt(base_prompt, last_errors)
                    continue
                raise ValueError("Model did not return a valid roadmap JSON")

            _coerce_skill_ids(roadmap, active_missions)
            errors = _validate_roadmap(roadmap, active_missions, mastered_missions)
            if not errors:
                return roadmap, accumulated_usage

            logging.warning(
                "Roadmap attempt %d failed validation with %d issue(s): %s",
                attempt + 1,
                len(errors),
                errors[:5],
            )
            last_errors = errors
            if attempt < max_retries:
                prompt = self._build_retry_prompt(base_prompt, errors)

        # Out of retries. Refuse to ship the broken roadmap.
        summary = "; ".join(last_errors[:5])
        if len(last_errors) > 5:
            summary += f" (+{len(last_errors) - 5} more)"
        raise ValueError(f"Roadmap failed validation after retry: {summary}")

    @staticmethod
    def _build_retry_prompt(base_prompt: str, errors: list[str]) -> str:
        """Append a "fix these specific issues" block to the original prompt.

        We replay the full base prompt rather than just sending the diff because
        models tend to drop context across turns; restating the rules + giving
        the concrete error list gets us a much higher success rate on the retry.
        """
        bullet_list = "\n".join(f"- {e}" for e in errors[:20])
        suffix = f"""

# RETRY — YOUR PREVIOUS RESPONSE WAS REJECTED
Your previous output failed validation with the following issues. Re-emit the
ENTIRE roadmap JSON, fixing every issue. Do NOT add prose, do NOT acknowledge
this message — return only the corrected JSON document.

ISSUES:
{bullet_list}
"""
        return base_prompt + suffix
