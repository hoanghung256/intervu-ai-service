import json
import os
import logging
from infrastructure.model_provider.llm_provider import LLMProvider
from api.dtos import AssessmentRequest

class AssessmentService:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def generate_assessment(self, request: AssessmentRequest):
        skill_reference_path = os.path.join(os.getcwd(), "skill-references.json")
        with open(skill_reference_path, "r", encoding="utf-8") as f:
            skill_reference = json.load(f)

        skill_reference_str = json.dumps(skill_reference, indent=2)

        prompt = f"""
You are an intelligent technical interviewer inside a career development platform.
Your job is to guide a new user through a short assessment to understand:
1. Their CURRENT level (Phase A)
2. Their TARGET expectations (Phase B)

SKILL REFERENCE:
{skill_reference_str}

INPUT:
Role: {request.role}
Level: {request.level}
TechStack: {request.techstack}
Domain: {request.domain}
User Intent: {request.free_text}

OUTPUT FORMAT:
{{
    "contextQuestion": "",
    "phaseA": [
        {{
            "skill": "",
            "question": "",
            "options": [
                {{ "text": "", "level": "None" }},
                {{ "text": "", "level": "Basic" }},
                {{ "text": "", "level": "Intermediate" }},
                {{ "text": "", "level": "Advanced" }}
            ]
        }}
    ],
    "phaseB": [
        {{
            "skill": "",
            "question": "",
            "options": [
                {{ "text": "", "level": "Beginner" }},
                {{ "text": "", "level": "Comfortable" }},
                {{ "text": "", "level": "Confident" }},
                {{ "text": "", "level": "Expert" }}
            ]
        }}
    ]
}}
"""
        response_text = await self.llm_provider.generate_content(prompt, model="gemma-3-27b-it")
        cleaned_json = self.llm_provider.clean_json_string(response_text)
        return json.loads(cleaned_json)
