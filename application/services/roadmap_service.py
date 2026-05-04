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
_MAX_NODES_PER_PHASE = 6
_PASS_THRESHOLD = 70
_STAGE_ORDER = {"foundation": 0, "core": 1, "specialized": 2, "production": 3}
_STAGE_PHASE_COPY = {
    "foundation": (
        "Foundations & Remediation",
        "Close the prerequisite gaps first so later interview practice has a stable base.",
    ),
    "core": (
        "Core Role Skills",
        "Build the day-to-day skills expected for the target role and level.",
    ),
    "specialized": (
        "Specialized Role Practice",
        "Apply the core skills to higher-context role scenarios without jumping ahead of the target level.",
    ),
    "production": (
        "Production Readiness & Career Simulation",
        "Round out the roadmap with testing, deployment, observability, and interview simulation practice.",
    ),
}
_SPLIT_PHASE_NAMES = {
    "foundation": [
        ("Foundations & Remediation", "Close the prerequisite gaps first so later interview practice has a stable base."),
        ("Core Foundations", "Continue strengthening the baseline skills needed for your target role."),
    ],
    "core": [
        ("Core Role Skills", "Build the day-to-day skills expected for the target role and level."),
        ("Core Competencies", "Deepen the remaining core competencies to reach interview readiness."),
    ],
    "specialized": [
        ("Applied Architecture", "Apply design and integration skills to realistic role scenarios."),
        ("System Design & Scalability", "Tackle advanced system-level challenges for your target role."),
    ],
    "production": [
        ("Production Readiness", "Round out the roadmap with testing, deployment, and observability."),
        ("Career Simulation & Polish", "Fine-tune production skills and prepare for real interview settings."),
    ],
}


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


def _mission_progression_key(mission: LearningMission) -> tuple:
    return (
        _STAGE_ORDER.get(mission.progression_stage, 1),
        mission.sequence_index,
        -mission.priority_weight,
        -1 if mission.interview_critical else 0,
        -mission.delta,
        mission.score,
        mission.skill_name,
    )


def _dedupe_missions(missions: list[LearningMission]) -> list[LearningMission]:
    seen: set[str] = set()
    ordered: list[LearningMission] = []
    for mission in sorted(missions, key=_mission_progression_key):
        key = _normalize_skill_key(mission.skill_id)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(mission)
    return ordered


def _phase_sort_key(mission: LearningMission) -> tuple:
    """Within a stage, sort Weak (has progress) before Missing (quick wins first),
    then by priority_weight descending, then by sequence_index."""
    has_progress = 1 if mission.current_level > 0 else 0
    return (-has_progress, -mission.priority_weight, mission.sequence_index)


def _build_phase_plan(missions: list[LearningMission]) -> list[dict]:
    """Build the roadmap structure in code so the LLM only writes copy.

    Phases are grouped by progression_stage, then split if a stage has too many
    skills. Split phases get unique names from _SPLIT_PHASE_NAMES instead of
    generic 'Part N' suffixes. Within each stage, skills with existing progress
    (Weak) are placed before Missing ones so learners get quick wins early.

    Soft skills are NOT added per phase; they go into the final phase only.
    """
    ordered = _dedupe_missions(missions)
    if not ordered:
        return []

    grouped: dict[str, list[LearningMission]] = {}
    for mission in ordered:
        stage = mission.progression_stage if mission.progression_stage in _STAGE_ORDER else "core"
        grouped.setdefault(stage, []).append(mission)

    phases: list[dict] = []
    for stage in sorted(grouped, key=lambda s: _STAGE_ORDER.get(s, 1)):
        stage_missions = sorted(grouped[stage], key=_phase_sort_key)
        split_names = _SPLIT_PHASE_NAMES.get(stage, _SPLIT_PHASE_NAMES["core"])
        default_name, default_desc = _STAGE_PHASE_COPY.get(stage, _STAGE_PHASE_COPY["core"])

        for chunk_start in range(0, len(stage_missions), _MAX_NODES_PER_PHASE):
            chunk = stage_missions[chunk_start : chunk_start + _MAX_NODES_PER_PHASE]
            split_index = chunk_start // _MAX_NODES_PER_PHASE

            if split_index < len(split_names):
                display_name, phase_description = split_names[split_index]
            else:
                display_name = f"{default_name} ({split_index + 1})"
                phase_description = default_desc

            phase_id = f"p{len(phases) + 1}"
            phase_nodes = [
                {
                    "node_id": f"{phase_id}-hard-{idx + 1}",
                    "skill_id": mission.skill_id,
                    "skill_name": mission.skill_name,
                    "pillar_type": "HARD_SKILL",
                }
                for idx, mission in enumerate(chunk)
            ]

            phase_nodes.append(
                {
                    "node_id": f"{phase_id}-checkpoint",
                    "skill_id": f"{phase_id}-live-checkpoint",
                    "skill_name": "Live Mock Interview Checkpoint",
                    "pillar_type": "LIVE_CHECKPOINT",
                    "assessment": {
                        "current_level": "0",
                        "target_level": "2",
                        "sfia_level": 0,
                        "status": "Unlocked",
                        "progress": 0,
                        "score": 0,
                    },
                    "child_skills": [],
                    "mentor_note": "Book this live mock to validate this phase's skills in a real interview setting.",
                    "interview_drills": [],
                    "checkpoint": {
                        "title": "Live Mock Interview Checkpoint",
                        "objective": "Validate this phase's hard skills in a live interview.",
                        "pass_threshold": _PASS_THRESHOLD,
                    },
                },
            )

            phases.append(
                {
                    "phase_id": phase_id,
                    "phase_name": display_name,
                    "phase_description": phase_description,
                    "status": "Unlocked",
                    "is_unlocked": True,
                    "nodes": phase_nodes,
                }
            )

    if phases:
        last_phase = phases[-1]
        last_phase_id = last_phase["phase_id"]
        soft_node = {
            "node_id": f"{last_phase_id}-soft-1",
            "skill_id": f"{last_phase_id}-soft-skills",
            "skill_name": "Communication & Interview Storytelling",
            "pillar_type": "SOFT_SKILL",
            "assessment": {
                "current_level": "0",
                "target_level": "2",
                "sfia_level": 0,
                "status": "Missing",
                "progress": 0,
                "score": 0,
            },
            "child_skills": [
                "Self-pitching & elevator pitch",
                "STAR method storytelling",
                "Conflict resolution & teamwork",
            ],
            "mentor_note": "Prepare concise stories that connect your technical work to business impact.",
            "interview_drills": [
                "Practice a 60-second self pitch",
                "Draft one STAR story from a difficult project",
                "Prepare a conflict-resolution example",
            ],
        }
        checkpoint_idx = next(
            (i for i, n in enumerate(last_phase["nodes"]) if n.get("pillar_type") == "LIVE_CHECKPOINT"),
            len(last_phase["nodes"]),
        )
        last_phase["nodes"].insert(checkpoint_idx, soft_node)

    return phases


def _render_phase_plan_for_prompt(phase_plan: list[dict]) -> str:
    if not phase_plan:
        return "[]"
    return json.dumps(phase_plan, ensure_ascii=False, indent=2)


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
        "sequence_index": mission.sequence_index,
        "progression_stage": mission.progression_stage,
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


def _phase_tokens(phase: dict) -> set[str]:
    """Collect tokens from all nodes in a phase for coach matching."""
    parts: list[str] = [phase.get("phase_name", ""), phase.get("phase_description", "")]
    for node in phase.get("nodes", []) or []:
        parts.append(node.get("skill_name", ""))
        parts.append(node.get("skill_id", ""))
        for child in node.get("child_skills", []) or []:
            if isinstance(child, str):
                parts.append(child)
            elif isinstance(child, dict):
                parts.append(child.get("name", ""))
    return _tokenize(" ".join(parts))


def _attach_recommendations(roadmap: dict, catalog: list[CoachCatalogEntry]) -> dict:
    """Assign one coach per phase (not per node) and ensure diversity across phases."""
    if not catalog:
        return roadmap

    prepared = [(c, _coach_tokens(c)) for c in catalog if c.services]
    if not prepared:
        return roadmap

    used_coach_ids: set[str] = set()

    for phase in roadmap.get("phases", []) or []:
        p_tokens = _phase_tokens(phase)
        ranked = sorted(
            prepared,
            key=lambda pair: _score(p_tokens, pair[1]),
            reverse=True,
        )

        chosen = None
        for coach, _ in ranked:
            if coach.id not in used_coach_ids:
                chosen = coach
                break
        if chosen is None:
            chosen = ranked[0][0]

        used_coach_ids.add(chosen.id)

        coach_payload = {
            "id": chosen.id,
            "name": chosen.name,
            "slug_profile_url": chosen.slug_profile_url or "",
            "avatar_url": chosen.avatar_url or "",
        }

        phase["recommended_coach"] = coach_payload
        phase.setdefault("recommended_coaches", [])
        if not phase["recommended_coaches"]:
            phase["recommended_coaches"] = [coach_payload]

    return roadmap


def _child_skill_display(child: object) -> str:
    if isinstance(child, str):
        return child
    if isinstance(child, dict):
        return str(child.get("name") or child.get("Name") or "")
    return ""


def _dedupe_child_skills(roadmap: dict) -> dict:
    for phase in roadmap.get("phases", []) or []:
        for node in phase.get("nodes", []) or []:
            deduped: list[object] = []
            seen: set[str] = set()
            for child in node.get("child_skills", []) or []:
                display = _child_skill_display(child)
                key = _normalize_skill_key(display)
                if not key or key in seen:
                    continue
                seen.add(key)
                deduped.append(child)
            node["child_skills"] = deduped
    return roadmap


def _validate_roadmap(
    roadmap: dict,
    active_missions: list[LearningMission],
    mastered_missions: list[LearningMission],
    expected_phase_plan: Optional[list[dict]] = None,
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
       - When a phase plan is supplied, phase/node order must match it exactly.

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
    active_names = {_normalize_skill_key(m.skill_name) for m in active_missions}
    mastered_ids = {_normalize_skill_key(m.skill_id) for m in mastered_missions}
    mastered_names = {_normalize_skill_key(m.skill_name) for m in mastered_missions}
    mission_keys = active_ids | active_names | mastered_ids | mastered_names

    phases = roadmap.get("phases") or []
    expected_phase_plan = expected_phase_plan or []

    if active_missions and not phases:
        errors.append("missing phases: there are active learning missions but the roadmap has no phases")

    if expected_phase_plan and len(phases) != len(expected_phase_plan):
        errors.append(
            f"phase count mismatch: expected {len(expected_phase_plan)} phase(s), got {len(phases)}"
        )

    expected_flat: list[str] = []
    actual_flat: list[str] = []
    seen_node_ids: set[str] = set()
    seen_child_keys_global: set[str] = set()
    for phase_idx, phase in enumerate(phases):
        expected_phase = expected_phase_plan[phase_idx] if phase_idx < len(expected_phase_plan) else None
        if expected_phase is not None:
            expected_phase_id = expected_phase.get("phase_id")
            actual_phase_id = phase.get("phase_id")
            if actual_phase_id != expected_phase_id:
                errors.append(
                    f"phases[{phase_idx}]: phase_id {actual_phase_id!r} does not match planned {expected_phase_id!r}"
                )
            expected_phase_name = expected_phase.get("phase_name")
            actual_phase_name = phase.get("phase_name")
            if actual_phase_name != expected_phase_name:
                errors.append(
                    f"phases[{phase_idx}]: phase_name {actual_phase_name!r} does not match planned {expected_phase_name!r}"
                )

        phase_expected_keys = [
            _normalize_skill_key(node.get("skill_id") or "")
            for node in (expected_phase or {}).get("nodes", [])
        ]
        expected_flat.extend(phase_expected_keys)
        phase_actual_keys: list[str] = []

        checkpoint_count = 0
        for node_idx, node in enumerate(phase.get("nodes") or []):
            node_path = f"phases[{phase_idx}].nodes[{node_idx}]"
            skill_id = (node.get("skill_id") or "").strip()
            pillar_type = node.get("pillar_type") or "HARD_SKILL"
            if not skill_id:
                errors.append(f"{node_path}: skill_id is empty")
                continue

            key = _normalize_skill_key(skill_id)
            phase_actual_keys.append(key)
            actual_flat.append(key)
            if pillar_type == "LIVE_CHECKPOINT":
                checkpoint_count += 1
                checkpoint = node.get("checkpoint") or {}
                if int(checkpoint.get("pass_threshold") or _PASS_THRESHOLD) != _PASS_THRESHOLD:
                    errors.append(f"{node_path}: live checkpoint pass_threshold must be {_PASS_THRESHOLD}")
            elif pillar_type == "HARD_SKILL":
                if key in mastered_ids:
                    errors.append(
                        f"{node_path}: skill_id {skill_id!r} is in the mastered list and must NOT appear as a node"
                    )
                elif key not in active_ids:
                    errors.append(
                        f"{node_path}: skill_id {skill_id!r} does not match any active mission"
                    )
            elif pillar_type != "SOFT_SKILL":
                errors.append(f"{node_path}: pillar_type {pillar_type!r} is invalid")

            if key in seen_node_ids:
                errors.append(f"{node_path}: skill_id {skill_id!r} is duplicated across nodes")
            else:
                seen_node_ids.add(key)

            seen_child_keys: set[str] = set()
            for child_idx, child in enumerate(node.get("child_skills") or []):
                child_name = _child_skill_display(child)
                child_key = _normalize_skill_key(child_name)
                if not child_key:
                    continue
                if child_key in seen_child_keys:
                    errors.append(f"{node_path}.child_skills[{child_idx}]: {child_name!r} is duplicated")
                else:
                    seen_child_keys.add(child_key)
                if child_key in seen_child_keys_global:
                    errors.append(
                        f"{node_path}.child_skills[{child_idx}]: {child_name!r} is duplicated across phases"
                    )
                else:
                    seen_child_keys_global.add(child_key)
                if child_key in mission_keys:
                    errors.append(
                        f"{node_path}.child_skills[{child_idx}]: {child_name!r} duplicates a roadmap mission"
                    )

        if expected_phase is not None and phase_actual_keys != phase_expected_keys:
            expected_ids = [node.get("skill_id") for node in expected_phase.get("nodes", [])]
            actual_ids = [node.get("skill_id") for node in phase.get("nodes", []) or []]
            errors.append(
                f"phases[{phase_idx}]: node order mismatch; expected {expected_ids}, got {actual_ids}"
            )

        if checkpoint_count != 1:
            errors.append(f"phases[{phase_idx}]: expected exactly one LIVE_CHECKPOINT node, got {checkpoint_count}")

    if expected_phase_plan and actual_flat != expected_flat:
        expected_ids = [
            node.get("skill_id")
            for phase in expected_phase_plan
            for node in phase.get("nodes", [])
        ]
        actual_ids = [
            node.get("skill_id")
            for phase in phases
            for node in phase.get("nodes", []) or []
        ]
        errors.append(f"roadmap node sequence mismatch; expected {expected_ids}, got {actual_ids}")

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
            if (node.get("pillar_type") or "HARD_SKILL") != "HARD_SKILL":
                continue
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
        active_missions = _dedupe_missions(active_missions)
        phase_plan = _build_phase_plan(active_missions)

        rendered_missions = _render_missions_for_prompt(active_missions)
        rendered_mastered = _render_mastered_for_prompt(mastered_missions)
        rendered_phase_plan = _render_phase_plan_for_prompt(phase_plan)

        target_role = (request.target_skill.roles or [""])[0] or ""
        target_level = request.target_skill.level or ""

        base_prompt = """
# ROLE
You are a Senior Career Architect. Your job is to NARRATE a pre-computed
learning plan, not to design one. The candidate's gap analysis, target levels,
priority weights, time estimates, and phase/node order have already been
calculated. You write the human-facing copy without changing the structure.

# INPUT
You receive:
1. `target_role` and `target_level` — career band the candidate is reaching for.
2. `active_missions` — JSON-per-line list, already sorted in learning order.
   Each mission is a skill the candidate has NOT yet met target on. Fields:
   - `skill_id` (canonical — use verbatim, do not rename)
   - `current_level`, `target_level`, `delta` (target - current)
   - `priority_weight` (= matrix_weight × delta) and `interview_critical`
   - `sequence_index` and `progression_stage` (deterministic learning order)
   - `score` (0-100; lower = needs more remediation)
   - `time_estimate_hours`
   - `interview_themes` (optional, max 3): one-line topics the candidate
     struggled with — use these in mentor notes.
   - `failed_question_excerpts` (optional, max 3): actual interview questions
     they fumbled — quote or paraphrase in mentor_note when relevant.
3. `mastered_skills` — comma-separated list. Already at/above target. NEVER
   create nodes for these.
4. `phase_plan` — the exact phase/node skeleton you must emit.

# YOUR JOB
1. Copy `phase_plan` exactly:
   - Same number of phases.
   - Same `phase_id`, `phase_name`, and node order.
   - Same node `skill_id` values, verbatim. Hard-skill nodes use active mission ids; soft/checkpoint nodes use the synthetic ids in `phase_plan`.
   - Same `node_id`, `pillar_type`, phase status fields, and checkpoint object. Soft skill nodes only appear in the last phase.
   - Do not move, merge, remove, duplicate, or invent nodes.
2. For each planned phase:
   - Keep the supplied phase name.
   - You may polish `phase_description`, but it must still describe that phase.
3. For each planned node:
   - Use the mission's `skill_id` verbatim. Inventing ids is forbidden.
   - Use a clear `skill_name` based on the mission skill, not a broader level or phase.
   - Keep `child_skills` empty or use only fine-grained subtopics. Do NOT list
     another active or mastered mission as a child skill.
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
  "schema_version": 2,
  "roadmap_metadata": {
    "target_role": "string",
    "target_level": "string",
    "total_phases": number
  },
  "phases": [
    {
      "phase_id": "p1",
      "phase_name": "string",
      "phase_description": "1-2 sentences: why this phase and what the learner will gain",
      "status": "Unlocked",
      "is_unlocked": true,
      "nodes": [
        {
          "node_id": "string (copy from phase_plan)",
          "skill_id": "string (verbatim mission skill_id)",
          "skill_name": "string (same skill domain as skill_id)",
          "pillar_type": "HARD_SKILL",
          "assessment": {
            "current_level": "string",
            "target_level": "string",
            "sfia_level": 0,
            "status": "Weak",
            "progress": 0
          },
          "child_skills": ["Detailed sub-skill 1", "Detailed sub-skill 2"],
          "mentor_note": "1-2 sentences referencing the candidate's interview_themes / failed_question_excerpts when present.",
          "interview_drills": ["Drill A", "Drill B"],
          "checkpoint": null
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

# active_missions (one JSON object per line; sorted in the planned learning order)
{rendered_missions}

# mastered_skills (DO NOT create nodes for these)
{rendered_mastered}

# phase_plan (copy this structure exactly; only fill descriptions, notes, drills, and child subtopics)
{rendered_phase_plan}

# FINAL REMINDER
- For HARD_SKILL nodes, use ONLY skill_ids that appear in active_missions (verbatim). For SOFT_SKILL and LIVE_CHECKPOINT nodes, copy the synthetic skill_id from phase_plan.
- Mastered skills MUST NOT appear anywhere in the output.
- Phase and node order MUST match phase_plan exactly. Soft skill nodes only appear in the last phase.
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
            phase_plan=phase_plan,
        )

        _dedupe_child_skills(roadmap)
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
        phase_plan: list[dict],
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
            errors = _validate_roadmap(roadmap, active_missions, mastered_missions, phase_plan)
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
