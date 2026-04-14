import json
import logging
from api.roadmap_dto import RoadmapRequest
from infrastructure.model_provider.llm_provider import LLMProvider
from infrastructure.model_provider.model_constants import HUGGINGFACE_DEFAULT_MODEL


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

        response_text = await self.llm_provider.generate_content(prompt, model=HUGGINGFACE_DEFAULT_MODEL)

        try:
            cleaned_json = self.llm_provider.clean_json_string(response_text)
            return json.loads(cleaned_json)
        except Exception as ex:
            logging.error(f"Failed to parse roadmap JSON: {ex}")
            raise ValueError("Model did not return a valid roadmap JSON")
