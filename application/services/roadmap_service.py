import json
import logging
import re
from typing import Optional

from api.roadmap_dto import CoachCatalogEntry, RoadmapRequest
from infrastructure.model_provider.llm_provider import LLMProvider
from infrastructure.model_provider.model_constants import HUGGINGFACE_LLAMA_3_3_70B_INSTRUCT_GROQ


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


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


class RoadmapService:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def generate_roadmap(self, request: RoadmapRequest) -> dict:
        base_prompt = """
# ROLE
You are a Senior Career Architect and Software Architect. Your mission is to analyze a user's technical skill data and generate a highly personalized, structured Learning Roadmap. You must transform fragmented skills into a cohesive, high-level professional journey.

# INPUT DATA
You will receive a JSON object containing:
1. TargetJson: Target role, level (Junior/Mid/Senior), and priority skills.
2. CurrentJson: Existing skills with proficiency levels and SFIA Levels (0-3).
3. GapJson: Categorized "Weak" and "Missing" skills.

# GROUPING & CONSOLIDATION RULES (CRITICAL)
To avoid UI clutter and ensure a professional architectural view, follow these abstraction rules:
1. High-Level Encapsulation: Do NOT create standalone Nodes for individual tools or syntax (e.g., No separate nodes for "Git", "HTML", or "Flexbox").
2. Competency-Based Mapping: Consolidate related skills into broad "Parent Nodes" (Competency Areas).
   - Example: Group "HTML", "CSS", "Responsive Design" -> Node: "Modern Web Foundations".
   - Example: Group "PostgreSQL", "Redis", "Prisma" -> Node: "Data Architecture & Persistence".
3. The 4-Node Limit: Strictly aim for 3-4 primary Nodes per Phase. If there are more skills, merge them into a broader logical category.
4. Node vs. Child Skill:
   - A Node (skill_name) must represent a "Technical Domain".
   - Child Skills (child_skills) must list the specific tools, libraries, or sub-topics from the GapJson.

# ROADMAP GENERATION LOGIC
1. Structure: Divide the Roadmap into 3-4 logical "Phases".
2. Sequencing:
   - Phase 1: "Foundations & Remediation" (Fixing Weak gaps).
   - Phase 2 & 3: "Core Tech & Specialized Architecture" (Learning Missing domains).
   - Phase 4: "Production Readiness & Career Simulation" (Deployment, Testing, Mock Interviews).
3. Status & Progress Calculation:
   - If Current < Target: Status = "Weak" (Progress ~33%).
   - If Current = "None": Status = "Missing" (Progress = 0%).
   - If Current >= Target: Status = "Complete" (Progress = 100%).

# OUTPUT FORMAT (MANDATORY JSON)
Return ONLY a valid JSON. No conversational filler. Use this structure:
{
  "roadmap_metadata": {
    "target_role": "string",
    "target_level": "string",
    "total_phases": number
  },
  "phases": [
    {
      "phase_id": "string",
      "phase_name": "string",
      "phase_description": "1-2 sentence narrative explaining why this phase exists, how it builds on the previous phase, and what the learner will be able to do after completing it.",
      "nodes": [
        {
          "skill_id": "string",
          "skill_name": "string (Broad Domain Name)",
          "assessment": {
            "current_level": "string",
            "target_level": "string",
            "sfia_level": number,
            "status": "Weak | Missing | Complete",
            "progress": number
          },
          "child_skills": ["Detailed Skill 1", "Detailed Skill 2", "Detailed Skill 3"],
          "mentor_note": "A 1-sentence tip connecting their background to this new node."
        }
      ]
    }
  ]
}

# TONE & STYLE
Professional, analytical, and aligned with global engineering standards (SFIA/Big Tech levels).
"""

        request_payload = f"""

# REQUEST PAYLOAD
TargetSkill: {request.target_skill}
CurrentLevel: {request.current_level}
Gap: {request.gap}

# FINAL REMINDER
- Consolidate all individual skills into high-level Nodes.
- Each phase must have a maximum of 4 nodes.
- Use 'child_skills' to preserve the specific details from the Gap list.
"""

        prompt = base_prompt + request_payload

        # response_text, usage = await self.llm_provider.generate_content(prompt, model=HUGGINGFACE_DEFAULT_MODEL)
        response_text, usage = await self.llm_provider.generate_content(prompt, model=HUGGINGFACE_LLAMA_3_3_70B_INSTRUCT_GROQ)

        try:
            cleaned_json = self.llm_provider.clean_json_string(response_text)
            roadmap = json.loads(cleaned_json)
        except Exception as ex:
            logging.error(f"Failed to parse roadmap JSON: {ex}")
            raise ValueError("Model did not return a valid roadmap JSON")

        try:
            roadmap = _attach_recommendations(roadmap, request.coach_catalog or [])
        except Exception as ex:
            logging.warning(f"Failed to attach per-node recommendations: {ex}")

        return roadmap, usage
