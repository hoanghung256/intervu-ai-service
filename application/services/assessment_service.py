import json
import os
import logging
import re
from typing import Any, Dict, List, Set, Union
from infrastructure.model_provider.llm_provider import LLMProvider
from infrastructure.model_provider.model_constants import HUGGINGFACE_DEFAULT_MODEL
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
    _skill_reference = None
    PHASE_A_LEVELS = ["None", "Basic", "Intermediate", "Advanced"]
    PHASE_B_LEVELS = ["Beginner", "Comfortable", "Confident", "Expert"]

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.logger = logging.getLogger(__name__)

    @classmethod
    def _load_skill_reference(cls):
        if cls._skill_reference is None:
            path = os.path.join(os.getcwd(), "skill-references.json")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cls._skill_reference = json.load(f)
            except Exception as e:
                logging.error(f"Failed to load skill-references.json: {e}")
                cls._skill_reference = []
        return cls._skill_reference

    @staticmethod
    def _to_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
        txt = str(value).strip()
        return [txt] if txt else []

    def _build_skill_source_for_prompt(self, request: AssessmentRequest, skill_reference: Any) -> List[Dict[str, str]]:
        if not isinstance(skill_reference, list):
            return []

        role = normalize_text(request.role).lower()
        tech_items = [t.lower() for t in self._to_list(request.techstack)]

        target_categories: Set[str] = set()
        if any(k in role for k in ["front", "ui", "web"]):
            target_categories.update(["Frontend", "Fullstack"])
        if any(k in role for k in ["back", "api", "server"]):
            target_categories.update(["Backend", "Fullstack"])
        if any(k in role for k in ["data", "ml", "ai"]):
            target_categories.update(["Data", "AI"])
        if any(k in role for k in ["devops", "infra", "platform", "sre"]):
            target_categories.update(["DevOps"])
        if any(k in role for k in ["mobile", "android", "ios"]):
            target_categories.update(["Mobile"])
        if any(k in role for k in ["qa", "test"]):
            target_categories.update(["QA", "Testing"])

        for tech in tech_items:
            if any(k in tech for k in ["react", "angular", "vue", "typescript", "javascript", "html", "css"]):
                target_categories.update(["Frontend", "Fullstack"])
            if any(k in tech for k in ["node", "java", "python", "dotnet", "c#", "go", "postgres", "mysql", "mongo", "redis", "sql"]):
                target_categories.update(["Backend", "Fullstack", "Data"])

        compact_all = [
            {
                "name": str(item.get("name", "")).strip(),
                "category": str(item.get("category", "General")).strip() or "General",
            }
            for item in skill_reference
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        ]

        if not compact_all:
            return []

        if not target_categories:
            return compact_all[:40]

        matched = [s for s in compact_all if s.get("category") in target_categories]
        if len(matched) < 20:
            needed = 40 - len(matched)
            extras = [s for s in compact_all if s not in matched][:max(0, needed)]
            matched.extend(extras)
        return matched[:40]

    @staticmethod
    def _normalize_options(raw_options: Any, levels: List[str]) -> List[Dict[str, str]]:
        level_index = {lvl.lower(): i for i, lvl in enumerate(levels)}
        by_level: List[str] = [""] * len(levels)

        if isinstance(raw_options, list):
            for idx, opt in enumerate(raw_options):
                if not isinstance(opt, dict):
                    continue
                text = normalize_text(opt.get("text"))
                lvl = normalize_text(opt.get("level")).lower()
                if lvl in level_index:
                    by_level[level_index[lvl]] = text or by_level[level_index[lvl]]
                elif idx < len(by_level) and text and not by_level[idx]:
                    by_level[idx] = text

        return [
            {
                "text": by_level[i] or f"Can demonstrate {levels[i].lower()} capability in this scenario.",
                "level": levels[i],
            }
            for i in range(len(levels))
        ]

    @staticmethod
    def _fallback_question(skill: str, techstack_str: str, phase: str) -> str:
        if phase == "phaseA":
            return f"In a {techstack_str or 'project'} task, how would you apply {skill} to deliver a working feature?"
        return f"For your next growth step in {techstack_str or 'this stack'}, how would you strengthen {skill} in a real production scenario?"

    def _normalize_phase(
        self,
        items: Any,
        target_count: int,
        levels: List[str],
        fallback_skills: List[str],
        techstack_str: str,
        phase: str,
        avoid_skills: Set[str],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        seen_questions: Set[str] = set()
        available_skills = [s for s in fallback_skills if s and s not in avoid_skills] or fallback_skills or ["Problem Solving"]
        skill_cursor = 0

        raw_items = items if isinstance(items, list) else []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue

            skill = normalize_text(raw.get("skill"))
            question = normalize_text(raw.get("question"))
            if not skill:
                skill = available_skills[skill_cursor % len(available_skills)]
                skill_cursor += 1
            if not question:
                question = self._fallback_question(skill, techstack_str, phase)
            if question in seen_questions:
                continue

            if phase == "phaseB" and skill in avoid_skills:
                replacement_candidates = [s for s in available_skills if s not in avoid_skills]
                if replacement_candidates:
                    skill = replacement_candidates[skill_cursor % len(replacement_candidates)]
                    skill_cursor += 1

            normalized.append(
                {
                    "skill": skill,
                    "question": question,
                    "options": self._normalize_options(raw.get("options"), levels),
                }
            )
            seen_questions.add(question)
            if len(normalized) >= target_count:
                return normalized

        while len(normalized) < target_count:
            skill = available_skills[skill_cursor % len(available_skills)]
            skill_cursor += 1
            question = self._fallback_question(skill, techstack_str, phase)
            if question in seen_questions:
                question = f"{question} (Focus area {len(normalized) + 1})"
            normalized.append(
                {
                    "skill": skill,
                    "question": question,
                    "options": self._normalize_options([], levels),
                }
            )
            seen_questions.add(question)

        return normalized

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
        logging.info(f"Generating assessment for role: {request.role}, level: {request.level}")
        
        skill_reference = self._load_skill_reference()
        skill_source = self._build_skill_source_for_prompt(request, skill_reference)
        skill_reference_str = json.dumps(skill_source, separators=(",", ":"), ensure_ascii=False)

        techstack_str = (
            ", ".join(request.techstack) if isinstance(request.techstack, (list, tuple)) else (request.techstack or "")
        )
        domain_str = (
            ", ".join(request.domain) if isinstance(request.domain, (list, tuple)) else (request.domain or "")
        )

        free_text_normalized = normalize_text(request.free_text)

        prompt = f"""
Return ONE valid JSON object only.

Input:
- role: {request.role}
- level: {request.level}
- techstack: {techstack_str}
- domain: {domain_str}
- user_intent: {free_text_normalized}

Allowed skills (name/category only):
{skill_reference_str}

Requirements:
- Total questions = 20 exactly.
- phaseA length = 15 exactly; options levels = None, Basic, Intermediate, Advanced.
- phaseB length = 5 exactly; options levels = Beginner, Comfortable, Confident, Expert.
- Questions must be practical and scenario-based.
- Use only tech from techstack input.
- Use only skills from allowed skills.
- phaseB skills should not duplicate phaseA skills when possible.
- No markdown fences, no comments.

Output schema:
{{
  "contextQuestion": "",
  "phaseA": [{{"skill": "", "question": "", "options": [{{"text": "", "level": "None"}}, {{"text": "", "level": "Basic"}}, {{"text": "", "level": "Intermediate"}}, {{"text": "", "level": "Advanced"}}]}}],
  "phaseB": [{{"skill": "", "question": "", "options": [{{"text": "", "level": "Beginner"}}, {{"text": "", "level": "Comfortable"}}, {{"text": "", "level": "Confident"}}, {{"text": "", "level": "Expert"}}]}}]
}}
"""
        response_text, usage = await self.llm_provider.generate_content(
            prompt=prompt,
            model=HUGGINGFACE_DEFAULT_MODEL
        )

        self.logger.info("LLM response received for assessment generation")

        async def _parse_or_repair(candidate_text: str, context: str) -> dict:
            cleaned = self.llm_provider.clean_json_string(candidate_text)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as ex:
                self.logger.warning(f"JSON parse failed in {context}: {ex}")
                repair_prompt = f"""
Convert this to one strict valid JSON object only.
Use double quotes, no markdown, no comments, no trailing commas.

Content:
{cleaned}
"""
                repaired_text, _ = await self.llm_provider.generate_content(
                    prompt=repair_prompt,
                    model=HUGGINGFACE_DEFAULT_MODEL
                )
                repaired_json = self.llm_provider.clean_json_string(repaired_text)
                return json.loads(repaired_json)

        data = await _parse_or_repair(response_text, "initial assessment generation")

        if not isinstance(data, dict):
            data = {}

        fallback_skills = [s.get("name", "") for s in skill_source if isinstance(s, dict) and s.get("name")]
        phase_a = self._normalize_phase(
            items=data.get("phaseA"),
            target_count=15,
            levels=self.PHASE_A_LEVELS,
            fallback_skills=fallback_skills,
            techstack_str=techstack_str,
            phase="phaseA",
            avoid_skills=set(),
        )
        phase_a_skills = {normalize_text(item.get("skill")).lower() for item in phase_a}
        phase_b = self._normalize_phase(
            items=data.get("phaseB"),
            target_count=5,
            levels=self.PHASE_B_LEVELS,
            fallback_skills=fallback_skills,
            techstack_str=techstack_str,
            phase="phaseB",
            avoid_skills=phase_a_skills,
        )

        data = {
            "contextQuestion": normalize_text(data.get("contextQuestion")) or "Tell us about a recent technical challenge you solved.",
            "phaseA": phase_a,
            "phaseB": phase_b,
        }

        if len(data.get("phaseA", [])) != 15:
            raise Exception("Invalid Phase A")

        if len(data.get("phaseB", [])) != 5:
            raise Exception("Invalid Phase B")

        return data, usage
