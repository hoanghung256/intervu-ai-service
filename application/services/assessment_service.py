import json
import os
import logging
from typing import Union, List
from infrastructure.model_provider.llm_provider import LLMProvider
from api.dtos import AssessmentRequest

class AssessmentService:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    def is_empty_request(self, req: AssessmentRequest):
        def _empty(v):
            if v is None:
                return True
            if isinstance(v, str):
                return v.strip() == "" or v == "string"
            if isinstance(v, (list, tuple)):
                return all((not str(x).strip() or str(x) == "string") for x in v)
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

        techstack_str = (
            ", ".join(request.techstack) if isinstance(request.techstack, (list, tuple)) else (request.techstack or "")
        )
        domain_str = (
            ", ".join(request.domain) if isinstance(request.domain, (list, tuple)) else (request.domain or "")
        )

        prompt = f"""
You are an intelligent technical interviewer inside a career development platform.

Your job is to guide a new user through an assessment to understand:
1. Their CURRENT level (Phase A)
2. Their TARGET expectations (Phase B)

IMPORTANT:
- Keep the OUTPUT FORMAT exactly as defined
- Total questions = 20
    - Phase A = 15 questions
    - Phase B = 5 questions
- Questions MUST feel like a real technical interview (NOT a survey)

-------------------------------------

=== CONTEXT ===
- The user is new
- Keep conversation natural and simple
- Do NOT overwhelm, but questions must be meaningful and practical

-------------------------------------

=== SKILL REFERENCE ===
{skill_reference_str}

-------------------------------------

=== INPUT ===
Role: {request.role}
Level: {request.level}
TechStack: {techstack_str}
Domain: {domain_str}
User Intent: {request.free_text}

-------------------------------------

=== TECHSTACK ENFORCEMENT (CRITICAL) ===

You MUST strictly filter all skills and questions based on TechStack.

STEP 1 — EXTRACT TECHSTACK
- Identify ALL technologies listed in TechStack

STEP 2 — MAP SKILLS
- From {{SKILL REFERENCE}}, ONLY select skills that:
    - Directly belong to the given TechStack
    - OR are universally required (e.g. API design, system design)

STEP 3 — HARD EXCLUSION RULE
- You are STRICTLY FORBIDDEN from:
    - Mentioning any technology NOT in TechStack
    - Asking about concepts that REQUIRE other technologies

EXAMPLES:

IF TechStack = "MySQL ONLY":
- ALLOWED:
    - SQL queries
    - indexing
    - normalization
    - transactions
    - query optimization

- FORBIDDEN:
    - Spring Boot
    - API controllers
    - Redis
    - authentication systems
    - ORM frameworks

IF TechStack = "Redis ONLY":
- ALLOWED:
    - caching strategies
    - TTL
    - pub/sub
    - streams

- FORBIDDEN:
    - SQL joins
    - REST APIs
    - controllers

STEP 4 — QUESTION VALIDATION (MANDATORY)

Before finalizing each question, you MUST check:
- Does this question ONLY rely on the given TechStack?
→ If NO → REMOVE or REWRITE

FAIL CONDITION:
If ANY question includes out-of-scope technology → the entire output is INVALID

-------------------------------------

=== HARD REQUIREMENT (VERY IMPORTANT) ===

- Questions MUST be strongly tied to the user's TechStack

- You MUST:
    - Identify key technologies from TechStack
    - Generate questions based on those technologies
    - Adapt concepts accordingly

EXAMPLES:

If TechStack includes:
- Java → focus on OOP, structure, backend patterns
- Spring Boot → API design, service/controller layers
- MySQL → queries, schema, optimization
- Redis → caching, performance

If TechStack includes:
- Node.js → async handling, API design, event loop
- React → state management, component design
- MongoDB → document modeling, aggregation

- DO NOT ask about technologies NOT present in TechStack
- DO NOT hardcode any specific technology
- ALWAYS adapt to the provided TechStack

- Focus ONLY on areas supported by the given TechStack
- Do NOT force API, caching, or system design if not applicable

-------------------------------------

=== QUESTION DESIGN STRATEGY ===

GENERAL RULE:
- Simulate a REAL backend interview
- Focus on practical ability (what user can DO)
- Avoid definitions unless tied to real usage

-------------------------------------

STEP 1 — CONTEXT QUESTION

- Ask 1 friendly, natural question

-------------------------------------

STEP 2 — PHASE A (CURRENT LEVEL) — 15 QUESTIONS

CORE LOGIC:

1. SKILL COVERAGE
- Select 5–7 skills from skill reference
- Must match Backend role
- Must align with:
    API Development, Database Design, Caching, System Design

- Each skill appears 2–3 times

2. DIFFICULTY PROGRESSION

- Q1–Q5: Basic usage
- Q6–Q10: Real project usage
- Q11–Q15: Problem solving / optimization / design

3. QUESTION STYLE (STRICT)

Each question MUST:
- Be scenario-based
- Be practical
- Mention tech context when possible

-------------------------------------

4. OPTIONS GENERATION (CRITICAL)

- DO NOT hardcode generic options
- AI MUST generate options dynamically based on each question

Each question MUST include exactly 4 options mapped to:
    - None
    - Basic
    - Intermediate
    - Advanced

STRICT RULES:

- Each option MUST:
    - Be SHORT (5–12 words max)
    - Represent a REAL action or approach
    - Be specific to the scenario in the question
    - Show a clear progression in capability
    - Act like a hint, NOT a full explanation

- Options should feel like:
    "different ways to solve the problem"

- DO NOT write:
    - generic self-evaluation (e.g. "I am good at this")
    - vague statements
    - long explanations

LEVEL GUIDELINE:

- None:
    → No experience or never done it

- Basic:
    → Naive / manual / incomplete approach

- Intermediate:
    → Practical, correct implementation

- Advanced:
    → Optimized, scalable, or well-designed solution

GOOD EXAMPLE:

Question:
"How would you cache product data in an API?"

Options:
- None: "Never used caching"
- Basic: "Cache manually in code"
- Intermediate: "Use Redis with TTL"
- Advanced: "Cache-aside with invalidation"

BAD EXAMPLE:
- "I know this"
- "I am experienced"

-------------------------------------

STEP 3 — PHASE B (TARGET) — LAST 5 QUESTIONS

- MUST be exactly 5 questions
- MUST use DIFFERENT skills from Phase A

LOGIC:

- Ask what level the user WANTS to reach
- Focus on:
    - scalability
    - performance
    - system design
    - architecture decisions

-------------------------------------

PHASE B OPTIONS (ALSO DYNAMIC)

Each question MUST include 4 levels:
- Beginner
- Comfortable
- Confident
- Expert

BUT:

- Each option MUST describe:
    - what the user wants to be able to DO
    - not just a label

Example:
- Beginner: "Understand basic concepts and follow tutorials"
- Comfortable: "Build and maintain features independently"
- Confident: "Handle real-world challenges and trade-offs"
- Expert: "Design systems and guide others"

-------------------------------------

=== STRICT RULES ===

- MUST only use skills from skill reference
- DO NOT invent new skills
- DO NOT duplicate skills between Phase A and Phase B
- Questions must be short, clear, and practical
- Avoid textbook definitions
- Avoid vague wording
- DO NOT generate generic questions

-------------------------------------

=== OUTPUT FORMAT ===
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
        response_text = await self.llm_provider.generate_content(prompt, model="gemini-2.0-flash")
        cleaned_json = self.llm_provider.clean_json_string(response_text)
        return json.loads(cleaned_json)
