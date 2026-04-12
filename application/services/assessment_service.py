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
ROLE
You are a technical interviewer in a career development platform.

OBJECTIVE
Generate one personalized assessment that measures:
1. Current capability (phaseA)
2. Target capability (phaseB)

INPUT
- Role: {request.role}
- Level: {request.level}
- TechStack: {techstack_str}
- Domain: {domain_str}
- User Intent: {free_text_normalized}

SKILL SOURCE (ONLY ALLOWED SOURCE)
{skill_reference_str}

STRICT CONSTRAINTS
1. Total questions must be exactly 20.
2. phaseA must contain exactly 15 items.
3. phaseB must contain exactly 5 items.
4. Return exactly one valid JSON object and nothing else.
5. Do not use markdown code fences.
6. Do not add comments or trailing commas.
7. Use valid JSON syntax with double quotes for all keys and string values.
8. Escape any inner double quotes inside text values.
9. Do not output JavaScript object literals.
10. Use only technologies listed in TechStack.
11. Do not mention out-of-scope technologies.
12. Use only skills from the skill source.
13. phaseB skills must not duplicate phaseA skills.

QUESTION DESIGN RULES
- All questions must be practical and scenario-based.
- Avoid generic survey wording.
- phaseA progression:
    - Q1-Q5: basic usage
    - Q6-Q10: real project usage
    - Q11-Q15: optimization / problem solving / design

OPTION RULES
- Every question must have exactly 4 options.
- phaseA levels must be exactly: None, Basic, Intermediate, Advanced.
- phaseB levels must be exactly: Beginner, Comfortable, Confident, Expert.
- Option text should be concise, actionable, and progressively harder.

SELF-CHECK BEFORE OUTPUT
- Ensure len(phaseA) == 15.
- Ensure len(phaseB) == 5.
- Ensure each question has exactly 4 options.
- Ensure option level names match allowed values exactly.
- If counts are wrong, fix counts before returning final JSON.
- Ensure output is parseable by Python json.loads without modification.

RESPONSE SCHEMA (MUST MATCH)
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
            model="meta-llama/Llama-3.1-8B-Instruct:novita"
        )

        self.logger.info(f"LLM Response: {response_text}")

        async def _parse_or_repair(candidate_text: str, context: str) -> dict:
            cleaned = self.llm_provider.clean_json_string(candidate_text)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as ex:
                self.logger.warning(f"JSON parse failed in {context}: {ex}")
                repair_prompt = f"""
Convert the following content into STRICT valid JSON.

Rules:
- Output ONLY one JSON object.
- Use double quotes for all keys and string values.
- No markdown code fences, no comments, no trailing commas.
- Keep original meaning and schema.

Content:
{cleaned}
"""
                repaired_text = await self.llm_provider.generate_content(
                    prompt=repair_prompt,
                    model="meta-llama/Llama-3.1-8B-Instruct:novita"
                )
                repaired_json = self.llm_provider.clean_json_string(repaired_text)
                return json.loads(repaired_json)

        data = await _parse_or_repair(response_text, "initial assessment generation")

        retry_count = 0
        while (len(data.get("phaseA", [])) != 15 or len(data.get("phaseB", [])) != 5) and retry_count < 3:
            count_fix_prompt = f"""
You will repair one assessment JSON object.

REQUIREMENTS
- Return valid JSON only.
- Keep the same schema and keys.
- phaseA length must be exactly 15.
- phaseB length must be exactly 5.
- Preserve original items as much as possible.
- If items are missing, add consistent items using the same style/context.
- If items are extra, remove the least relevant items.
- Keep level enums strictly valid.
- Use double quotes for all keys and string values.
- Ensure result is directly parseable by Python json.loads.

CONTEXT
- Role: {request.role}
- Level: {request.level}
- TechStack: {techstack_str}
- Domain: {domain_str}
- User Intent: {free_text_normalized}

ALLOWED LEVELS
- phaseA: None, Basic, Intermediate, Advanced
- phaseB: Beginner, Comfortable, Confident, Expert

JSON CANDIDATE
{json.dumps(data, ensure_ascii=False)}
"""

            fixed_text = await self.llm_provider.generate_content(
                prompt=count_fix_prompt,
                model="meta-llama/Llama-3.1-8B-Instruct:novita"
            )
            data = await _parse_or_repair(fixed_text, "count-fix generation")
            retry_count += 1

        phase_a = data.get("phaseA", []) if isinstance(data.get("phaseA", []), list) else []
        phase_b = data.get("phaseB", []) if isinstance(data.get("phaseB", []), list) else []

        if len(phase_a) < 15:
            missing_a = 15 - len(phase_a)
            add_phase_a_prompt = f"""
Generate ONLY a JSON array for missing phaseA items.

Requirements:
- Return a JSON array with exactly {missing_a} objects.
- Each object schema:
    {{
        "skill": "",
        "question": "",
        "options": [
            {{"text": "", "level": "None"}},
            {{"text": "", "level": "Basic"}},
            {{"text": "", "level": "Intermediate"}},
            {{"text": "", "level": "Advanced"}}
        ]
    }}
- Use only TechStack technologies.
- Use skills from skill reference.
- Do not repeat existing questions.

Context:
Role: {request.role}
Level: {request.level}
TechStack: {techstack_str}
Domain: {domain_str}
User Intent: {free_text_normalized}

Skill source:
{skill_reference_str}

Existing phaseA:
{json.dumps(phase_a, ensure_ascii=False)}
"""

            add_phase_a_text = await self.llm_provider.generate_content(
                prompt=add_phase_a_prompt,
                model="meta-llama/Llama-3.1-8B-Instruct:novita"
            )
            add_phase_a_data = await _parse_or_repair(add_phase_a_text, "phaseA missing-items generation")
            if isinstance(add_phase_a_data, dict):
                add_phase_a_data = add_phase_a_data.get("phaseA", [])
            if isinstance(add_phase_a_data, list):
                phase_a.extend(add_phase_a_data)

        if len(phase_b) < 5:
            missing_b = 5 - len(phase_b)
            add_phase_b_prompt = f"""
Generate ONLY a JSON array for missing phaseB items.

Requirements:
- Return a JSON array with exactly {missing_b} objects.
- Each object schema:
    {{
        "skill": "",
        "question": "",
        "options": [
            {{"text": "", "level": "Beginner"}},
            {{"text": "", "level": "Comfortable"}},
            {{"text": "", "level": "Confident"}},
            {{"text": "", "level": "Expert"}}
        ]
    }}
- Use only TechStack technologies.
- Use skills from skill reference.
- phaseB skills should differ from phaseA skills when possible.

Context:
Role: {request.role}
Level: {request.level}
TechStack: {techstack_str}
Domain: {domain_str}
User Intent: {free_text_normalized}

Skill source:
{skill_reference_str}

Existing phaseA:
{json.dumps(phase_a, ensure_ascii=False)}

Existing phaseB:
{json.dumps(phase_b, ensure_ascii=False)}
"""

            add_phase_b_text = await self.llm_provider.generate_content(
                prompt=add_phase_b_prompt,
                model="meta-llama/Llama-3.1-8B-Instruct:novita"
            )
            add_phase_b_data = await _parse_or_repair(add_phase_b_text, "phaseB missing-items generation")
            if isinstance(add_phase_b_data, dict):
                add_phase_b_data = add_phase_b_data.get("phaseB", [])
            if isinstance(add_phase_b_data, list):
                phase_b.extend(add_phase_b_data)

        data["phaseA"] = phase_a[:15]
        data["phaseB"] = phase_b[:5]

        if len(data.get("phaseA", [])) != 15:
            raise Exception("Invalid Phase A")

        if len(data.get("phaseB", [])) != 5:
            raise Exception("Invalid Phase B")

        return data
