import json
import logging
from api.roadmap_dto import RoadmapProgressUpdateRequest
from infrastructure.model_provider.llm_provider import LLMProvider
from infrastructure.model_provider.model_constants import HUGGINGFACE_DEFAULT_MODEL

class RoadmapProgressService:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def update_roadmap_progress(self, request: RoadmapProgressUpdateRequest) -> dict:
        avg_score = (
            sum(e.score for e in request.evaluation) / len(request.evaluation)
            if request.evaluation
            else 0
        )

        base_prompt = """
# ROLE
You are a Senior Career Progress Analyst. Your job is to update a candidate's Learning Roadmap based on their most recent mock interview evaluation scores. You must recalculate node progress and status to reflect real growth — do NOT reset or invent data.

# INPUT DATA
You will receive:
1. current_roadmap: The candidate's existing roadmap JSON (phases, nodes, assessments).
2. interview_type: The type of interview just completed (e.g., "Technical Backend", "Behavioral").
3. aim_level: The target level aimed for in this interview (e.g., "Junior", "Mid", "Senior").
4. evaluation: A list of scored evaluation criteria (type, score out of 10, question, answer).

# UPDATE RULES (CRITICAL)
1. Identify Relevant Nodes: Based on interview_type and aim_level, identify which phase and nodes this interview most likely covers. Use the node skill_name and child_skills as context clues.
2. Update Progress:
   - avg_score = average of all evaluation scores (0–10 scale).
   - New progress for matched nodes = round((avg_score / 10) * 100).
   - If current node progress is already HIGHER than the new value, keep the higher value (never regress).
3. Update Status:
   - progress >= 80 → "Complete"
   - progress >= 40 → "Weak"
   - progress < 40  → "Missing"
4. Untouched Nodes: Any node NOT matched to this interview type must remain EXACTLY as-is (same progress, same status, same child_skills).
5. Metadata: Do NOT change roadmap_metadata.

# OUTPUT FORMAT (MANDATORY JSON)
Return ONLY the complete updated roadmap JSON with the same structure as current_roadmap. No explanation, no markdown fences.
"""

        request_payload = f"""
# REQUEST PAYLOAD
interview_type: {request.interview_type}
aim_level: {request.aim_level}
avg_score: {avg_score:.1f}/10

evaluation:
{json.dumps([e.model_dump() for e in request.evaluation], indent=2)}

current_roadmap:
{json.dumps(request.current_roadmap, indent=2)}

# FINAL REMINDER
- Only update nodes relevant to the interview_type.
- Never lower progress that is already higher.
- Return the FULL roadmap JSON unchanged except for updated nodes.
"""

        prompt = base_prompt + request_payload

        response_text = await self.llm_provider.generate_content(
            prompt, 
            model=HUGGINGFACE_DEFAULT_MODEL
        )

        try:
            cleaned_json = self.llm_provider.clean_json_string(response_text)
            return json.loads(cleaned_json)
        except Exception as ex:
            logging.error(f"Failed to parse updated roadmap JSON: {ex}")
            raise ValueError("Model did not return a valid roadmap JSON")
