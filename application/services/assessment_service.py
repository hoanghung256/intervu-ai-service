import json
import os
import logging
import re
from typing import Union
from infrastructure.model_provider.llm_provider import LLMProvider
from api.dtos import AssessmentRequest

CONTRACTIONS = {
    "i'm": "i am",
    "i've": "i have",
    "i'd": "i would",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "it's": "it is",
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "isn't": "is not",
    "aren't": "are not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "there's": "there is",
    "that's": "that is",
    "we've": "we have",
    "they've": "they have",
}
CONTRACTIONS_PATTERN = re.compile(r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS.keys()) + r")\b", flags=re.IGNORECASE)

def normalize_text(text: Union[str, None]) -> str:
    if not text:
        return ""
    txt = str(text).replace("\u2019", "'").replace("\u2018", "'").replace("’", "'")
    txt = CONTRACTIONS_PATTERN.sub(lambda m: CONTRACTIONS.get(m.group(0).lower(), m.group(0)), txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

class AssessmentService:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.logger = logging.getLogger(__name__)

    def is_empty_request(self, req: AssessmentRequest):
        def _empty(v):
            if v is None:
                return True
            if isinstance(v, str):
                s = normalize_text(v)
                return s == "" or s.lower() == "string"
            if isinstance(v, (list, tuple)):
                def _elem_empty(x):
                    nx = normalize_text(x)
                    return not nx or nx.lower() == "string"
                return all(_elem_empty(x) for x in v)
            return False

        return (
            _empty(req.role) and
            _empty(req.level) and
            _empty(req.techstack) and
            _empty(req.free_text)
        )

    async def generate_assessment(self, request: AssessmentRequest):
        skill_reference_path = os.path.join(os.getcwd(), "skill-references.json")
        with open(skill_reference_path, "r", encoding="utf-8") as f:
            skill_reference = json.load(f)

        skill_reference_str = json.dumps(skill_reference, indent=2)

        # normalize free text using module-level helper

        techstack_str = (
            ", ".join(request.techstack) if isinstance(request.techstack, (list, tuple)) else (request.techstack or "")
        )
        domain_str = (
            ", ".join(request.domain) if isinstance(request.domain, (list, tuple)) else (request.domain or "")
        )

        free_text_normalized = normalize_text(request.free_text)

        prompt = f"""
You are a technical interviewer in a career development platform.

Goal:
- Generate a personalized assessment to identify:
    1) CURRENT capability (Phase A)
    2) TARGET capability (Phase B)

Hard requirements (must pass all):
- Total questions = 20.
- phaseA length = 15.
- phaseB length = 5.
- Return strict JSON only (no text outside JSON).
- Keep the exact key names and level values defined below.

User context:
- Role: {request.role}
- Level: {request.level}
- TechStack: {techstack_str}
- Domain: {domain_str}
- User Intent: {free_text_normalized}

Skill source (do not use skills outside this source):
{skill_reference_str}

TechStack filtering rules:
1) Use only technologies present in TechStack.
2) Do not mention any technology outside TechStack.
3) Remove or rewrite any question that depends on out-of-scope technology.
4) Questions must be practical and scenario-based, not generic survey wording.

Phase construction:
- contextQuestion: exactly 1 friendly question.

- phaseA:
    - Exactly 15 items.
    - Use 5-7 skills from skill reference.
    - Skill repetition per selected skill: 2-3 questions.
    - Difficulty progression:
        - Q1-Q5 basic usage
        - Q6-Q10 real project usage
        - Q11-Q15 optimization/problem-solving/design
    - Each item needs options with levels exactly:
        None, Basic, Intermediate, Advanced

- phaseB:
    - Exactly 5 items.
    - Skills must be different from phaseA skills.
    - Focus on target capability: scalability, performance, system design, architecture decisions.
    - Each item needs options with levels exactly:
        Beginner, Comfortable, Confident, Expert

Option writing rules:
- Exactly 4 options per question.
- 5-12 words each option.
- Actionable and scenario-specific.
- Clear progression from lower to higher capability.
- No vague self-rating text.

JSON output rules for parser safety:
- Output one JSON object only.
- First char must be '{{' and last char must be '}}'.
- No markdown code block.
- No comments.
- No trailing commas.
- No placeholder tokens like "...".

Before final output, silently validate:
- len(phaseA) == 15
- len(phaseB) == 5
- every question has 4 options
- all option levels match allowed values exactly

Output schema:
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
        response_text = await self.llm_provider.generate_content(
            prompt=prompt,
                        model = "meta-llama/Llama-3.1-8B-Instruct:novita"
        )

        self.logger.info(f"LLM Response: {response_text}")

        cleaned_json = self.llm_provider.clean_json_string(response_text)
        data = json.loads(cleaned_json)

        if len(data.get("phaseA", [])) != 15:
            raise Exception("Invalid Phase A")

        if len(data.get("phaseB", [])) != 5:
            raise Exception("Invalid Phase B")

        return data
