import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Set, Union

from api.dtos import AssessmentRequest, SurveyAnswerJsonDto, SurveyResponsesDto, SurveySummaryResultDto
from infrastructure.model_provider.llm_provider import LLMProvider
from infrastructure.model_provider.model_constants import HUGGINGFACE_LLAMA_3_3_70B_INSTRUCT_GROQ

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
CONTRACTIONS_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS.keys()) + r")\b",
    flags=re.IGNORECASE,
)


LEVEL_ORDER = ["none", "basic", "intermediate", "advanced"]
LEVEL_KEY_BY_INDEX = ["1", "2", "3", "4"]


def normalize_text(text: Union[str, None]) -> str:
    if not text:
        return ""
    txt = str(text).replace("\u2019", "'").replace("\u2018", "'").replace("’", "'")
    txt = CONTRACTIONS_PATTERN.sub(lambda m: CONTRACTIONS.get(m.group(0).lower(), m.group(0)), txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


class AssessmentService:
    _role_skill_framework: Optional[Dict[str, Any]] = None
    _score_rules: Optional[Dict[str, Any]] = None
    _skill_reference: Optional[Any] = None
    _skill_reference_index: Optional[Dict[str, Dict[str, Any]]] = None

    PHASE_A_LEVELS = ["1", "2", "3", "4"]
    PHASE_B_LEVELS = ["1", "2", "3", "4"]

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.logger = logging.getLogger(__name__)
        self._repo_root = Path(__file__).resolve().parents[2]

    @classmethod
    def _load_json_file(cls, path: Path, fallback: Any) -> Any:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as ex:
            logging.error(f"Failed to load JSON file {path}: {ex}")
            return fallback

    def _load_role_skill_framework(self) -> Dict[str, Any]:
        if self.__class__._role_skill_framework is None:
            path = self._repo_root / "role-skill-framework.json"
            self.__class__._role_skill_framework = self._load_json_file(path, {})
        return self.__class__._role_skill_framework or {}

    def _load_score_rules(self) -> Dict[str, Any]:
        if self.__class__._score_rules is None:
            path = self._repo_root / "assessment-score-rules.json"
            self.__class__._score_rules = self._load_json_file(path, {})
        return self.__class__._score_rules or {}

    def _load_skill_reference(self) -> Any:
        if self.__class__._skill_reference is None:
            path = self._repo_root / "skill-references.json"
            self.__class__._skill_reference = self._load_json_file(path, {})
        return self.__class__._skill_reference or {}

    def _iter_skill_reference_items(self) -> List[Dict[str, Any]]:
        data = self._load_skill_reference()
        items: List[Dict[str, Any]] = []

        if isinstance(data, list):
            items.extend([x for x in data if isinstance(x, dict)])
            return items

        if not isinstance(data, dict):
            return items

        if isinstance(data.get("skills"), list):
            for skill in data.get("skills", []):
                if isinstance(skill, dict):
                    items.append(skill)
            return items

        for role_payload in data.values():
            if not isinstance(role_payload, dict):
                continue
            skills = role_payload.get("skills")
            if not isinstance(skills, list):
                continue
            for skill in skills:
                if isinstance(skill, dict):
                    items.append(skill)
        return items

    def _extract_level_descriptions(self, ref: Optional[Dict[str, Any]]) -> Dict[str, str]:
        if not isinstance(ref, dict):
            return {}

        levels = ref.get("levels")
        descriptions: Dict[str, str] = {}

        if isinstance(levels, list):
            for level_item in levels:
                if not isinstance(level_item, dict):
                    continue
                level_label = normalize_text(level_item.get("level")).lower()
                description = normalize_text(level_item.get("description") or level_item.get("text"))
                if not level_label or not description:
                    continue
                idx = self.map_level(level_label)
                descriptions[LEVEL_ORDER[idx]] = description
            return descriptions

        if isinstance(levels, dict):
            for level_key, value in levels.items():
                level_label = normalize_text(str(level_key)).lower()
                description = normalize_text(value if isinstance(value, str) else value.get("description", ""))
                if not level_label or not description:
                    continue
                idx = self.map_level(level_label)
                descriptions[LEVEL_ORDER[idx]] = description

        return descriptions

    def _get_skill_reference_index(self) -> Dict[str, Dict[str, Any]]:
        if self.__class__._skill_reference_index is not None:
            return self.__class__._skill_reference_index

        index: Dict[str, Dict[str, Any]] = {}
        for item in self._iter_skill_reference_items():
            name = normalize_text(item.get("name")).lower()
            if not name:
                continue
            index[name] = item

        self.__class__._skill_reference_index = index
        return index

    @staticmethod
    def _to_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [normalize_text(v) for v in value if normalize_text(v)]
        txt = normalize_text(str(value))
        return [txt] if txt else []

    @staticmethod
    def _tokenize(value: str) -> Set[str]:
        return {t for t in re.findall(r"[a-z0-9\+#\.]+", normalize_text(value).lower()) if len(t) >= 2}

    @staticmethod
    def _level_key_for_index(idx: int) -> str:
        idx = max(0, min(3, idx))
        return LEVEL_KEY_BY_INDEX[idx]

    def _normalize_role_key(self, role: str, framework: Dict[str, Any]) -> str:
        role_normalized = normalize_text(role).lower()
        role_aliases = framework.get("role_aliases", {}) if isinstance(framework, dict) else {}
        role_definitions = framework.get("role_definitions", {}) if isinstance(framework, dict) else {}

        if role_normalized in role_definitions:
            return role_normalized
        if role_normalized in role_aliases:
            return role_aliases[role_normalized]

        for alias, canonical in role_aliases.items():
            if alias in role_normalized:
                return canonical

        if "front" in role_normalized:
            return "frontend_engineer"
        if "devops" in role_normalized or "sre" in role_normalized:
            return "devops_engineer"
        if "qa" in role_normalized or "test" in role_normalized:
            return "qa_engineer"
        if "architect" in role_normalized:
            return "solution_architect"
        if "data" in role_normalized or "database" in role_normalized:
            return "database_designer"
        return "backend_engineer"

    @staticmethod
    def _normalize_level_key(level: str) -> str:
        lvl = normalize_text(level).lower()
        if not lvl:
            return "junior"
        if any(k in lvl for k in ["fresher", "entry", "intern", "new grad", "beginner"]):
            return "fresher"
        if any(k in lvl for k in ["junior", "basic"]):
            return "junior"
        if any(k in lvl for k in ["middle", "mid", "intermediate"]):
            return "middle"
        if any(k in lvl for k in ["senior", "lead", "staff", "principal", "advanced", "expert"]):
            return "senior"
        return "junior"

    def _find_skill_reference(self, skill_name: str) -> Optional[Dict[str, Any]]:
        skill_key = normalize_text(skill_name).lower()
        index = self._get_skill_reference_index()
        if skill_key in index:
            return index[skill_key]

        for ref_key, ref in index.items():
            if skill_key in ref_key or ref_key in skill_key:
                return ref
        return None

    def _get_role_skills(self, role_key: str, level_key: str, framework: Dict[str, Any]) -> List[Dict[str, Any]]:
        role_definitions = framework.get("role_definitions", {}) if isinstance(framework, dict) else {}
        role_data = role_definitions.get(role_key, {}) if isinstance(role_definitions, dict) else {}
        if not isinstance(role_data, dict):
            return []

        if level_key in role_data and isinstance(role_data[level_key], list):
            return role_data[level_key]

        for fallback in ["junior", "middle", "senior", "fresher"]:
            if fallback in role_data and isinstance(role_data[fallback], list):
                return role_data[fallback]
        return []

    def _prioritize_skills(
        self,
        role_skills: List[Dict[str, Any]],
        techstack: List[str],
        domain: List[str],
        free_text: str,
    ) -> List[Dict[str, Any]]:
        free_tokens = self._tokenize(free_text)
        tech_tokens = self._tokenize(" ".join(techstack))
        domain_tokens = self._tokenize(" ".join(domain))

        scored: List[Dict[str, Any]] = []
        for item in role_skills:
            skill_name = normalize_text(item.get("skill"))
            if not skill_name:
                continue

            skill_tokens = self._tokenize(skill_name)
            score = 0

            if free_tokens and (skill_tokens & free_tokens):
                score += 50
            if tech_tokens and (skill_tokens & tech_tokens):
                score += 30
            if domain_tokens and (skill_tokens & domain_tokens):
                score += 20

            min_level = int(item.get("min_level") or 1)
            score += min_level

            scored.append({
                "skill": skill_name,
                "min_level": min_level,
                "priority_score": score,
            })

        scored.sort(key=lambda x: (x["priority_score"], x["min_level"]), reverse=True)
        return scored

    def _build_level_hints(self, skill_name: str, question: str) -> List[str]:
        text = normalize_text(f"{skill_name} {question}").lower()
        if any(k in text for k in ["performance", "latency", "throughput", "optimize", "slow"]):
            return ["measure baseline latency", "optimize DB/query hotspots", "add caching and monitor p95/p99"]
        if any(k in text for k in ["auth", "authorization", "authentication", "security", "token", "jwt"]):
            return ["validate auth flow", "enforce RBAC and token checks", "harden threat model with audit and monitoring"]
        if any(k in text for k in ["database", "sql", "schema", "index", "query"]):
            return ["review schema/query plan", "add indexes and tune queries", "set migration + consistency + capacity strategy"]
        if any(k in text for k in ["api", "endpoint", "rest", "integration"]):
            return ["define request/response contract", "handle validation + errors + retries", "design versioning, observability, and SLA guards"]
        if any(k in text for k in ["deploy", "container", "kubernetes", "ci/cd", "release"]):
            return ["prepare reproducible build/deploy", "add health checks and rollback", "design progressive rollout with reliability gates"]
        if any(k in text for k in ["test", "testing", "bug", "qa"]):
            return ["write core happy-path tests", "cover edge cases + integration tests", "enforce quality gates and flake control"]
        return ["implement a clear baseline solution", "add validation and reliability controls", "design for scale with observability and risk management"]

    def _contextualize_option(
        self,
        level_idx: int,
        base_text: str,
        skill_name: str,
        question: str,
    ) -> str:
        hints = self._build_level_hints(skill_name, question)
        base = normalize_text(base_text)

        if level_idx == 0:
            return (
                f"{base} The response cannot yet describe a clear and relevant approach for this scenario, "
                "including key constraints or expected outcomes."
            )
        if level_idx == 1:
            return (
                f"{base} Start by {hints[0]}, explain core implementation steps, and run basic verification "
                "to confirm the solution works as expected."
            )
        if level_idx == 2:
            return (
                f"{base} Execute {hints[0]}, then {hints[1]}, and validate results with clear metrics, "
                "trade-off analysis, and a practical mitigation plan."
            )
        return (
            f"{base} Lead end-to-end by {hints[0]}, {hints[1]}, and {hints[2]}, while coordinating risk controls "
            "and a safe production rollout strategy."
        )

    def _is_generic_option_text(self, text: str, question: str) -> bool:
        content = normalize_text(text)
        if not content:
            return True
        lowered = content.lower()
        if "can demonstrate" in lowered and "capability" in lowered:
            return True
        if len(content.split()) < 8:
            return True
        option_tokens = self._tokenize(content)
        question_tokens = self._tokenize(question)
        if question_tokens and len(option_tokens & question_tokens) == 0:
            return True
        return False

    def _coerce_option_level_index(self, raw_level: Any) -> Optional[int]:
        level_text = normalize_text(str(raw_level))
        if not level_text:
            return None
        if level_text in {"1", "2", "3", "4"}:
            return int(level_text) - 1
        mapped = self.map_level(level_text)
        if mapped < 0 or mapped > 3:
            return None
        return mapped

    def _normalize_question_options(
        self,
        raw_options: Any,
        skill_name: str,
        phase: str,
        question: str,
    ) -> List[Dict[str, str]]:
        phase_levels = self.PHASE_A_LEVELS if phase == "phaseA" else self.PHASE_B_LEVELS
        default_options = self._build_options_from_skill_levels(skill_name, phase, question)
        by_idx: List[Optional[str]] = [None, None, None, None]

        if isinstance(raw_options, list):
            for idx, opt in enumerate(raw_options):
                if not isinstance(opt, dict):
                    continue
                text = normalize_text(opt.get("text"))
                if not text:
                    continue
                level_idx = self._coerce_option_level_index(opt.get("level"))
                if level_idx is None and idx < 4:
                    level_idx = idx
                if level_idx is None or level_idx < 0 or level_idx > 3:
                    continue
                if by_idx[level_idx] is None:
                    by_idx[level_idx] = text

        options: List[Dict[str, str]] = []
        for i in range(4):
            candidate = by_idx[i] or default_options[i]["text"]
            # If LLM returns generic filler, rewrite to scenario-aware hint.
            if self._is_generic_option_text(candidate, question):
                candidate = self._contextualize_option(i, default_options[i]["text"], skill_name, question)
            options.append({"text": candidate, "level": phase_levels[i]})
        return options

    def _build_options_from_skill_levels(self, skill_name: str, phase: str, question: str = "") -> List[Dict[str, str]]:
        ref = self._find_skill_reference(skill_name)
        phase_levels = self.PHASE_A_LEVELS if phase == "phaseA" else self.PHASE_B_LEVELS

        descriptions_by_level: Dict[str, str] = {}
        if ref:
            descriptions_by_level = self._extract_level_descriptions(ref)

        options: List[Dict[str, str]] = []
        for idx, phase_level in enumerate(phase_levels):
            base_level = LEVEL_ORDER[idx]
            base_text = descriptions_by_level.get(base_level)
            if not base_text:
                base_text = {
                    0: f"Limited hands-on experience with {skill_name}.",
                    1: f"Can apply {skill_name} at a basic implementation level.",
                    2: f"Can apply {skill_name} with solid structure and reliability.",
                    3: f"Can lead advanced {skill_name} decisions in production.",
                }[idx]
            text = self._contextualize_option(
                level_idx=idx,
                base_text=base_text,
                skill_name=skill_name,
                question=question,
            )
            options.append({"text": text, "level": phase_level})

        return options

    @staticmethod
    def _fallback_question(skill: str, techstack_str: str, practical: bool = False) -> str:
        if practical:
            return (
                f"In a production incident on {techstack_str or 'your current stack'}, "
                f"how would you apply {skill} to resolve the issue with minimal risk?"
            )
        return f"How do you apply {skill} in day-to-day delivery for a {techstack_str or 'real project'}?"

    def _normalize_phase(
        self,
        items: Any,
        target_count: int,
        skill_pool: List[str],
        techstack_str: str,
        phase: str,
        avoid_skills: Set[str],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        seen_questions: Set[str] = set()

        allowed_skills = [s for s in skill_pool if s]
        if not allowed_skills:
            allowed_skills = ["Problem Solving"]

        cursor = 0
        raw_items = items if isinstance(items, list) else []

        for raw in raw_items:
            if not isinstance(raw, dict):
                continue

            skill = normalize_text(raw.get("skill"))
            question = normalize_text(raw.get("question"))

            if skill not in allowed_skills:
                skill = allowed_skills[cursor % len(allowed_skills)]
                cursor += 1

            if phase == "phaseB" and skill.lower() in avoid_skills:
                candidates = [s for s in allowed_skills if s.lower() not in avoid_skills]
                if candidates:
                    skill = candidates[cursor % len(candidates)]
                    cursor += 1

            if not question:
                question = self._fallback_question(skill, techstack_str, practical=(phase == "phaseB"))

            if question in seen_questions:
                continue

            normalized.append(
                {
                    "skill": skill,
                    "question": question,
                    "options": self._normalize_question_options(raw.get("options"), skill, phase, question),
                }
            )
            seen_questions.add(question)

            if len(normalized) >= target_count:
                return normalized

        while len(normalized) < target_count:
            skill = allowed_skills[cursor % len(allowed_skills)]
            cursor += 1

            if phase == "phaseB" and skill.lower() in avoid_skills:
                candidates = [s for s in allowed_skills if s.lower() not in avoid_skills]
                if candidates:
                    skill = candidates[cursor % len(candidates)]
                    cursor += 1

            question = self._fallback_question(skill, techstack_str, practical=(phase == "phaseB"))
            if question in seen_questions:
                question = f"{question} (Scenario {len(normalized) + 1})"

            normalized.append(
                {
                    "skill": skill,
                    "question": question,
                    "options": self._build_options_from_skill_levels(skill, phase, question),
                }
            )
            seen_questions.add(question)

        return normalized

    def _build_prompt_skill_payload(self, prioritized_skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for skill_item in prioritized_skills:
            skill_name = skill_item.get("skill", "")
            ref = self._find_skill_reference(skill_name)
            levels: Dict[str, str] = {}
            descriptions = self._extract_level_descriptions(ref) if ref else {}
            for idx, level_name in enumerate(LEVEL_ORDER):
                levels[self._level_key_for_index(idx)] = descriptions.get(
                    level_name,
                    f"Can demonstrate {level_name} capability in {skill_name}.",
                )

            payload.append(
                {
                    "name": skill_name,
                    "min_level": int(skill_item.get("min_level") or 1),
                    "levels": levels,
                }
            )
        return payload

    @staticmethod
    def _build_question_templates(practical: bool) -> List[str]:
        if practical:
            return [
                "Your team reports a production issue in {skill}. How would you diagnose and fix it in {stack}?",
                "A high-traffic release in {domain} stresses {skill}. What trade-offs would you make first?",
                "How would you design a resilient approach for {skill} to reduce incidents in {stack}?",
                "In a post-mortem for {skill}, what actions would you prioritize for the next sprint?",
                "How would you prevent recurring failures related to {skill} in a real production workflow?",
            ]
        return [
            "How do you apply {skill} in day-to-day engineering work for {stack}?",
            "What is your standard approach for implementing {skill} in a feature delivery cycle?",
            "How do you validate quality and correctness when working on {skill}?",
            "Which design decisions matter most when applying {skill} in {domain} systems?",
            "How would you explain your practical implementation process for {skill}?",
        ]

    def _build_rule_based_phase_items(
        self,
        skill_pool: List[str],
        target_count: int,
        practical: bool,
        techstack_str: str,
        domain_items: List[str],
    ) -> List[Dict[str, str]]:
        skills = [s for s in skill_pool if s] or ["Problem Solving"]
        templates = self._build_question_templates(practical=practical)
        domain_text = ", ".join(domain_items) if domain_items else "general"
        stack_text = techstack_str or "your current stack"

        items: List[Dict[str, str]] = []
        for idx in range(target_count):
            skill = skills[idx % len(skills)]
            template = templates[idx % len(templates)]
            question = template.format(skill=skill, stack=stack_text, domain=domain_text)
            if idx >= len(skills):
                question = f"{question} (Focus {idx + 1})"
            items.append({"skill": skill, "question": question})
        return items

    def is_empty_request(self, req: AssessmentRequest):
        def _empty(v):
            if v is None:
                return True
            if isinstance(v, str):
                s = normalize_text(v)
                return s == "" or s.lower() == "string"
            if isinstance(v, (list, tuple)):
                return all((normalize_text(x) == "" or normalize_text(x).lower() == "string") for x in v)
            return False

        return _empty(req.role) and _empty(req.level) and _empty(req.techstack) and _empty(req.free_text)

    async def generate_assessment(self, request: AssessmentRequest):
        total_start = perf_counter()
        logging.info(f"Generating assessment for role: {request.role}, level: {request.level}")
        timings_ms: Dict[str, float] = {}

        t0 = perf_counter()
        framework = self._load_role_skill_framework()
        score_rules = self._load_score_rules()

        role_key = self._normalize_role_key(request.role or "", framework)
        level_key = self._normalize_level_key(request.level or "")
        role_skills = self._get_role_skills(role_key, level_key, framework)

        tech_items = self._to_list(request.techstack)
        domain_items = self._to_list(request.domain)
        free_text = normalize_text(request.free_text)

        prioritized_skills = self._prioritize_skills(role_skills, tech_items, domain_items, free_text)
        if not prioritized_skills:
            prioritized_skills = [
                {"skill": item.get("name", "General Skill"), "min_level": 1, "priority_score": 0}
                for item in self._iter_skill_reference_items()[:10]
            ]

        core_count = int(score_rules.get("question_plan", {}).get("core_count", 10))
        practical_count = int(score_rules.get("question_plan", {}).get("practical_count", 5))
        techstack_str = ", ".join(tech_items)
        domain_str = ", ".join(domain_items)
        fallback_skills = [normalize_text(s.get("skill")) for s in prioritized_skills if normalize_text(s.get("skill"))]
        timings_ms["prepare_inputs"] = round((perf_counter() - t0) * 1000, 3)

        generation_mode = "llm"
        data: Dict[str, Any] = {}

        t0 = perf_counter()
        skill_payload = self._build_prompt_skill_payload(prioritized_skills)
        skill_payload_str = json.dumps(skill_payload, separators=(",", ":"), ensure_ascii=False)
        prompt = f"""
Return ONE valid JSON object only.

Input:
- role_key: {role_key}
- role_level: {level_key}
- techstack: {techstack_str}
- domain: {domain_str}
- user_intent: {free_text}

Role skills and level semantics:
{skill_payload_str}

Requirements:
- Total questions = {core_count + practical_count}.
- phaseA length = {core_count} (core skills).
- phaseB length = {practical_count} (practical scenarios).
- Use only the listed role skills.
- Keep questions concrete and production-oriented.
- For EACH question, return 4 options with levels "1","2","3","4".
- Option text must be scenario-specific and directly related to the question context.
- Options must be distinct in action depth, not just rewording level labels.
- Each option should be a complete sentence with medium length (around 18-30 words).
- Do not truncate option text and do not use trailing ellipsis.
- Do NOT output generic phrases like "Can demonstrate ... capability".
- Do not return markdown or comments.

Output schema:
{{
  "contextQuestion": "",
  "phaseA": [{{"skill": "", "question": "", "options": [{{"text":"", "level":"1"}}, {{"text":"", "level":"2"}}, {{"text":"", "level":"3"}}, {{"text":"", "level":"4"}}]}}],
  "phaseB": [{{"skill": "", "question": "", "options": [{{"text":"", "level":"1"}}, {{"text":"", "level":"2"}}, {{"text":"", "level":"3"}}, {{"text":"", "level":"4"}}]}}]
}}
"""
        timings_ms["build_prompt"] = round((perf_counter() - t0) * 1000, 3)
        usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if self.llm_provider and hasattr(self.llm_provider, "generate_content"):
            t0 = perf_counter()
            response_text, usage = await self.llm_provider.generate_content(
                prompt=prompt,
                model=HUGGINGFACE_LLAMA_3_3_70B_INSTRUCT_GROQ,
            )
            timings_ms["llm_generate"] = round((perf_counter() - t0) * 1000, 3)

            async def _parse_or_repair(candidate_text: str, context: str) -> dict:
                parse_start = perf_counter()
                cleaned = self.llm_provider.clean_json_string(candidate_text)
                try:
                    parsed = json.loads(cleaned)
                    timings_ms["parse_json"] = round((perf_counter() - parse_start) * 1000, 3)
                    return parsed
                except json.JSONDecodeError as ex:
                    self.logger.warning(f"JSON parse failed in {context}: {ex}")
                    repair_prompt = f"""
Convert this content to one strict valid JSON object only.
Use double quotes, no markdown, no comments, no trailing commas.

Content:
{cleaned}
"""
                    repair_start = perf_counter()
                    repaired_text, _ = await self.llm_provider.generate_content(
                        prompt=repair_prompt,
                        model=HUGGINGFACE_LLAMA_3_3_70B_INSTRUCT_GROQ,
                    )
                    timings_ms["llm_repair"] = round((perf_counter() - repair_start) * 1000, 3)
                    repaired_json = self.llm_provider.clean_json_string(repaired_text)
                    parsed = json.loads(repaired_json)
                    timings_ms["parse_json"] = round((perf_counter() - parse_start) * 1000, 3)
                    return parsed

            parsed_data = await _parse_or_repair(response_text, "assessment generation")
            data = parsed_data if isinstance(parsed_data, dict) else {}
        else:
            generation_mode = "rule_based_fallback"
            timings_ms["llm_generate"] = 0.0
            data = {
                "contextQuestion": "",
                "phaseA": self._build_rule_based_phase_items(
                    skill_pool=fallback_skills,
                    target_count=core_count,
                    practical=False,
                    techstack_str=techstack_str,
                    domain_items=domain_items,
                ),
                "phaseB": self._build_rule_based_phase_items(
                    skill_pool=fallback_skills,
                    target_count=practical_count,
                    practical=True,
                    techstack_str=techstack_str,
                    domain_items=domain_items,
                ),
            }

        t0 = perf_counter()
        phase_a = self._normalize_phase(
            items=data.get("phaseA"),
            target_count=core_count,
            skill_pool=fallback_skills,
            techstack_str=techstack_str,
            phase="phaseA",
            avoid_skills=set(),
        )
        phase_a_skills = {normalize_text(item.get("skill")).lower() for item in phase_a}
        phase_b = self._normalize_phase(
            items=data.get("phaseB"),
            target_count=practical_count,
            skill_pool=fallback_skills,
            techstack_str=techstack_str,
            phase="phaseB",
            avoid_skills=phase_a_skills,
        )
        timings_ms["normalize_output"] = round((perf_counter() - t0) * 1000, 3)

        elapsed_ms = round((perf_counter() - total_start) * 1000, 3)
        slowest_stage = max(timings_ms, key=timings_ms.get) if timings_ms else None

        result = {
            "contextQuestion": normalize_text(data.get("contextQuestion"))
            or "Tell us about a recent technical challenge you solved.",
            "phaseA": phase_a,
            "phaseB": phase_b,
            "metadata": {
                "roleKey": role_key,
                "levelKey": level_key,
                "coreCount": core_count,
                "practicalCount": practical_count,
                "generationMode": generation_mode,
                "generationModel": HUGGINGFACE_LLAMA_3_3_70B_INSTRUCT_GROQ if generation_mode == "llm" else None,
                "executionMs": elapsed_ms,
                "timingsMs": timings_ms,
                "slowestStage": slowest_stage,
            },
        }

        if len(result.get("phaseA", [])) != core_count:
            raise Exception("Invalid Phase A")
        if len(result.get("phaseB", [])) != practical_count:
            raise Exception("Invalid Phase B")

        if slowest_stage:
            self.logger.info(
                "Assessment generation timing: total=%sms, slowest=%s(%sms)",
                elapsed_ms,
                slowest_stage,
                timings_ms.get(slowest_stage, 0.0),
            )

        return result, usage

    def _level_alias_map(self) -> Dict[str, int]:
        rules = self._load_score_rules()
        aliases = rules.get("level_aliases", {}) if isinstance(rules, dict) else {}
        output: Dict[str, int] = {}
        for key, value in aliases.items():
            output[normalize_text(key).lower()] = int(value)
        if not output:
            output = {
                "none": 0,
                "basic": 1,
                "intermediate": 2,
                "advanced": 3,
                "beginner": 0,
                "comfortable": 1,
                "confident": 2,
                "expert": 3,
                "1": 0,
                "2": 1,
                "3": 2,
                "4": 3,
            }
        return output

    def map_level(self, level: str) -> int:
        aliases = self._level_alias_map()
        return aliases.get(normalize_text(level).lower(), 0)

    def _display_level(self, score: float) -> str:
        rules = self._load_score_rules()
        thresholds = rules.get("score_thresholds", {}) if isinstance(rules, dict) else {}
        none_max = float(thresholds.get("none_max", 0.5))
        basic_max = float(thresholds.get("basic_max", 1.5))
        intermediate_max = float(thresholds.get("intermediate_max", 2.5))

        if score <= none_max:
            return "None"
        if score <= basic_max:
            return "Basic"
        if score <= intermediate_max:
            return "Intermediate"
        return "Advanced"

    def _base_level_label(self, level: str) -> str:
        idx = self.map_level(level)
        return LEVEL_ORDER[max(0, min(3, idx))]

    def _profile_target_level(self, level: str) -> int:
        lv = normalize_text(level).lower()
        if not lv or lv in {"none", "0"}:
            return 0
        if any(k in lv for k in ["fresher", "entry", "intern", "beginner", "1"]):
            return 1
        if any(k in lv for k in ["junior", "2"]):
            return 2
        if any(k in lv for k in ["middle", "mid", "intermediate", "3"]):
            return 3
        if any(k in lv for k in ["senior", "lead", "staff", "principal", "advanced", "expert", "4"]):
            return 4
        return 2

    def _to_actual_level(self, selected_level: str) -> Optional[int]:
        raw = normalize_text(selected_level).lower()
        if not raw:
            return None
        if raw in {"0", "1", "2", "3", "4"}:
            return int(raw)
        if raw == "none":
            return 0
        if raw in {"fresher", "beginner"}:
            return 1
        if raw in {"junior", "basic", "comfortable"}:
            return 2
        if raw in {"middle", "intermediate", "confident"}:
            return 3
        if raw in {"senior", "advanced", "expert"}:
            return 4
        return None

    @staticmethod
    def _achievement_score(actual_level: float, target_level: int) -> float:
        if target_level <= 0:
            return 0.0
        return round(max(0.0, min(100.0, (actual_level / target_level) * 100.0)), 2)

    @staticmethod
    def _achievement_band(score_pct: float) -> str:
        if score_pct <= 0:
            return "None"
        if score_pct < 25:
            return "Fresher"
        if score_pct < 50:
            return "Junior"
        if score_pct < 75:
            return "Middle"
        return "Senior"

    def _resolve_sfia_level(self, skill: str, level: str) -> int:
        ref = self._find_skill_reference(skill)
        base_level = self._base_level_label(level)

        if ref and isinstance(ref.get("levels"), list):
            for level_item in ref["levels"]:
                if not isinstance(level_item, dict):
                    continue
                if normalize_text(level_item.get("level")).lower() == base_level:
                    try:
                        return int(level_item.get("sfia") or 0)
                    except Exception:
                        return 0

        rules = self._load_score_rules()
        fallback = rules.get("sfia_fallback", {}) if isinstance(rules, dict) else {}
        mapped = self.map_level(level)
        if str(mapped) in fallback:
            return int(fallback[str(mapped)])
        return {0: 0, 1: 2, 2: 3, 3: 5}.get(mapped, 0)

    def _semantic_answer_score(self, skill: str, selected_level: str, answer_text: str) -> float:
        ref = self._find_skill_reference(skill)
        if not ref:
            return float(self.map_level(selected_level))

        target_label = self._base_level_label(selected_level)
        selected_desc = self._extract_level_descriptions(ref).get(target_label, "")

        if not selected_desc:
            return float(self.map_level(selected_level))

        answer_tokens = self._tokenize(answer_text)
        desc_tokens = self._tokenize(selected_desc)
        if not answer_tokens or not desc_tokens:
            return float(self.map_level(selected_level))

        overlap = len(answer_tokens & desc_tokens)
        ratio = overlap / max(1, len(desc_tokens))
        semantic_score = min(3.0, round(ratio * 6, 2))
        return semantic_score

    def _allowed_skills_for_profile(self, role: str, level: str) -> List[str]:
        framework = self._load_role_skill_framework()
        role_key = self._normalize_role_key(role or "", framework)
        level_key = self._normalize_level_key(level or "")
        role_skills = self._get_role_skills(role_key, level_key, framework)
        collected: List[str] = [
            normalize_text(item.get("skill"))
            for item in role_skills
            if isinstance(item, dict) and normalize_text(item.get("skill"))
        ]

        # Fallback/merge from role-specific skill references.
        skill_ref = self._load_skill_reference()
        if isinstance(skill_ref, dict):
            role_ref = skill_ref.get(role_key)
            if isinstance(role_ref, dict) and isinstance(role_ref.get("skills"), list):
                for skill_item in role_ref.get("skills", []):
                    if not isinstance(skill_item, dict):
                        continue
                    skill_name = normalize_text(skill_item.get("name"))
                    if skill_name:
                        collected.append(skill_name)

        deduped: List[str] = []
        seen: Set[str] = set()
        for skill in collected:
            key = skill.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(skill)
        return deduped

    def _best_skill_match(self, input_skill: str, question_text: str, allowed_skills: List[str]) -> str:
        in_skill = normalize_text(input_skill)
        if not allowed_skills:
            return in_skill or "General Skill"
        if in_skill and in_skill in allowed_skills:
            return in_skill

        combined_tokens = self._tokenize(f"{in_skill} {question_text}")
        best_skill = allowed_skills[0]
        best_score = -1.0

        for skill in allowed_skills:
            skill_tokens = self._tokenize(skill)
            overlap = len(combined_tokens & skill_tokens)
            coverage = overlap / max(1, len(skill_tokens))
            soft = 0.2 if in_skill and (in_skill.lower() in skill.lower() or skill.lower() in in_skill.lower()) else 0.0
            score = coverage + soft
            if score > best_score:
                best_score = score
                best_skill = skill

        return best_skill

    def _infer_level_from_answer(self, skill: str, answer_text: str) -> Dict[str, Any]:
        ref = self._find_skill_reference(skill)
        descriptions = self._extract_level_descriptions(ref)
        answer_tokens = self._tokenize(answer_text)

        if not descriptions or not answer_tokens:
            return {"idx": 0, "score": 0.0}

        best_idx = 0
        best_score = 0.0

        for idx, level_name in enumerate(LEVEL_ORDER):
            desc = descriptions.get(level_name, "")
            desc_tokens = self._tokenize(desc)
            if not desc_tokens:
                continue
            overlap = len(answer_tokens & desc_tokens)
            score = overlap / max(1, len(desc_tokens))
            if score > best_score:
                best_score = score
                best_idx = idx

        return {"idx": best_idx, "score": round(best_score, 4)}

    async def _infer_levels_from_answers_ai(self, items: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        if not self.llm_provider or not hasattr(self.llm_provider, "generate_content"):
            return {}
        if not items:
            return {}

        eval_items: List[Dict[str, Any]] = []
        for item in items:
            key = int(item.get("key", -1))
            skill = normalize_text(item.get("skill"))
            question = normalize_text(item.get("question"))
            answer = normalize_text(item.get("answer"))
            if key < 0 or not skill or not answer:
                continue

            ref = self._find_skill_reference(skill)
            desc = self._extract_level_descriptions(ref)
            if not desc:
                continue

            level_desc = {
                "1": desc.get("none", ""),
                "2": desc.get("basic", ""),
                "3": desc.get("intermediate", ""),
                "4": desc.get("advanced", ""),
            }
            eval_items.append(
                {
                    "key": key,
                    "skill": skill,
                    "question": question,
                    "answer": answer,
                    "levelDescriptions": level_desc,
                }
            )

        if not eval_items:
            return {}

        prompt = f"""
You are scoring candidate answers against skill level descriptions.
Return strict JSON only, no markdown.

For each item:
- Compare answer to level descriptions.
- Pick exactly one level in "0","1","2","3","4".
- Use "0" only for empty/off-topic/no evidence.
- confidence is from 0.0 to 1.0.
- desc is a short reason.

Input:
{json.dumps(eval_items, ensure_ascii=False)}

Output schema:
{{
  "results": [
    {{"key": 0, "level": "0", "confidence": 0.0, "desc": ""}}
  ]
}}
"""
        try:
            response_text, _ = await self.llm_provider.generate_content(
                prompt=prompt,
                model=HUGGINGFACE_LLAMA_3_3_70B_INSTRUCT_GROQ,
            )
            cleaned = self.llm_provider.clean_json_string(response_text)
            parsed = json.loads(cleaned)
        except Exception as ex:
            self.logger.warning(f"AI scoring batch failed: {ex}")
            return {}

        results = parsed.get("results", []) if isinstance(parsed, dict) else []
        if not isinstance(results, list):
            return {}

        output: Dict[int, Dict[str, Any]] = {}
        for row in results:
            if not isinstance(row, dict):
                continue
            try:
                key = int(row.get("key"))
            except Exception:
                continue

            level_raw = normalize_text(row.get("level"))
            if level_raw not in {"0", "1", "2", "3", "4"}:
                continue

            try:
                confidence = float(row.get("confidence", 0))
            except Exception:
                confidence = 0.0

            confidence = max(0.0, min(1.0, confidence))
            output[key] = {
                "idx": max(0, min(3, int(level_raw) - 1)) if level_raw != "0" else 0,
                "score": round(confidence, 4),
                "desc": normalize_text(row.get("desc")),
                "rawLevel": level_raw,
            }

        return output

    async def evaluate_survey_responses(
        self,
        request: SurveyResponsesDto,
        target_input: Optional[Dict[str, Any]] = None,
        gap_input: Optional[Dict[str, Any]] = None,
        missing_input: Optional[List[str]] = None,
    ) -> SurveySummaryResultDto:
        answer_payload = request.answer
        answer_responses = answer_payload.responses if answer_payload else []

        resolved_target: Dict[str, Any] = {}
        target_level = 0
        role_for_target = ""
        if answer_payload and answer_payload.profile:
            profile = answer_payload.profile
            role_for_target = normalize_text(profile.role)
            target_level = self._profile_target_level(profile.level)
            resolved_target = {
                "roles": [role_for_target] if role_for_target else [],
                "level": normalize_text(profile.level),
                "skillsTarget": [normalize_text(s) for s in (profile.techstack or []) if normalize_text(s)],
            }
        elif isinstance(target_input, dict):
            resolved_target = target_input
            target_level = self._profile_target_level(str(target_input.get("level", "")))
            role_raw = target_input.get("roles", [])
            role_for_target = normalize_text(role_raw[0]) if isinstance(role_raw, list) and role_raw else ""

        allowed_skills = self._allowed_skills_for_profile(role_for_target, resolved_target.get("level", ""))
        if not allowed_skills:
            seen_allowed: Set[str] = set()
            for resp in answer_responses:
                skill_name = normalize_text(resp.skill)
                if skill_name and skill_name.lower() not in seen_allowed:
                    allowed_skills.append(skill_name)
                    seen_allowed.add(skill_name.lower())

        ai_items: List[Dict[str, Any]] = [
            {"key": idx, "skill": item.skill, "question": item.question, "answer": item.answer}
            for idx, item in enumerate(answer_responses)
        ]
        ai_inferred_levels = await self._infer_levels_from_answers_ai(ai_items)

        by_phase: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        scored_responses: List[Dict[str, Any]] = []
        skill_actual_map: Dict[str, List[int]] = defaultdict(list)

        for idx, item in enumerate(answer_responses):
            phase = normalize_text(item.phase) or "general"
            mapped_skill = self._best_skill_match(
                input_skill=item.skill,
                question_text=item.question,
                allowed_skills=allowed_skills,
            )

            selected_level_input = normalize_text(item.selectedLevel)
            selected_actual = self._to_actual_level(selected_level_input)

            semantic = self._infer_level_from_answer(mapped_skill, item.answer)
            semantic_actual = int(semantic["idx"]) + 1 if float(semantic["score"]) >= 0.06 else 0

            ai_semantic = ai_inferred_levels.get(idx)
            ai_actual = (int(ai_semantic["idx"]) + 1) if (ai_semantic and float(ai_semantic["score"]) >= 0.35) else 0

            if selected_actual is None:
                final_actual = max(ai_actual, semantic_actual)
            else:
                final_actual = selected_actual
                if max(ai_actual, semantic_actual) > 0:
                    final_actual = max(final_actual, max(ai_actual, semantic_actual))

            final_actual = max(0, min(4, int(final_actual)))
            is_missing = final_actual == 0
            score_pct = self._achievement_score(final_actual, target_level)

            by_phase[phase].append(
                {
                    "skill": mapped_skill,
                    "actual": final_actual,
                    "scorePct": score_pct,
                    "missing": is_missing,
                }
            )
            skill_actual_map[mapped_skill].append(final_actual)

            scored_responses.append(
                {
                    "questionId": item.questionId,
                    "question": item.question,
                    "phase": phase,
                    "skill": mapped_skill,
                    "answer": item.answer,
                    "selectedLevel": str(final_actual),
                    "score": score_pct,
                    "isMissing": is_missing,
                }
            )

        lines: List[str] = []
        for phase, items in by_phase.items():
            avg_score = round(sum(x["scorePct"] for x in items) / len(items), 2) if items else 0.0
            weak_skills = sorted({x["skill"] for x in items if (not x["missing"]) and x["actual"] < max(1, target_level)})
            lines.append(
                f"{phase}: average score {avg_score}%. Consider improving: {', '.join(weak_skills) if weak_skills else 'none'}"
            )

        current_skills: List[Dict[str, Any]] = []
        computed_missing: List[str] = []
        computed_weak: List[str] = []
        for skill, actuals in skill_actual_map.items():
            avg_actual = round(sum(actuals) / len(actuals), 2)
            avg_actual_int = int(round(avg_actual))
            skill_score = self._achievement_score(avg_actual, target_level)
            current_skills.append(
                {
                    "skill": skill,
                    "level": str(avg_actual_int),
                    "score": skill_score,
                }
            )
            if avg_actual_int == 0:
                computed_missing.append(skill)
            elif target_level > 0 and avg_actual < target_level:
                computed_weak.append(skill)

        resolved_gap: Dict[str, Any] = {"weak": [], "missing": []}
        if isinstance(gap_input, dict):
            weak_in = gap_input.get("weak", [])
            missing_in = gap_input.get("missing", [])
            resolved_gap["weak"] = [normalize_text(x) for x in weak_in if normalize_text(x)] if isinstance(weak_in, list) else []
            resolved_gap["missing"] = [normalize_text(x) for x in missing_in if normalize_text(x)] if isinstance(missing_in, list) else []
        elif request.gap:
            resolved_gap["weak"] = [normalize_text(x) for x in request.gap.weak if normalize_text(x)]
            resolved_gap["missing"] = [normalize_text(x) for x in request.gap.missing if normalize_text(x)]
        elif isinstance(missing_input, list):
            resolved_gap["missing"] = [normalize_text(x) for x in missing_input if normalize_text(x)]

        if not resolved_gap["weak"]:
            resolved_gap["weak"] = sorted(set(computed_weak))
        if not resolved_gap["missing"]:
            resolved_gap["missing"] = sorted(set(computed_missing))

        overall_score = round(
            sum(x["score"] for x in scored_responses) / len(scored_responses),
            2,
        ) if scored_responses else 0.0
        overall_level = self._achievement_band(overall_score)

        scored_answer = {
            "profile": answer_payload.profile.model_dump() if answer_payload and answer_payload.profile else {},
            "responses": scored_responses,
            "overallScore": overall_score,
            "overallLevel": overall_level,
        }

        if resolved_gap["missing"]:
            lines.append(f"Missing skills (from gap): {', '.join(resolved_gap['missing'])}")

        return SurveySummaryResultDto(
            userId=request.userId,
            summaryText="\n".join(lines),
            answer=scored_answer,
            target=resolved_target,
            current={"skills": sorted(current_skills, key=lambda x: x["skill"])},
            gapJson={
                "missing": resolved_gap["missing"],
            },
        )

    async def evaluate_answer_json(
        self,
        answer: SurveyAnswerJsonDto,
        gap_input: Optional[Dict[str, Any]] = None,
    ) -> SurveySummaryResultDto:
        req = SurveyResponsesDto(answer=answer)
        return await self.evaluate_survey_responses(req, gap_input=gap_input)
