
import logging
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Query, HTTPException, Body
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import fitz
import pymupdf4llm
import asyncio
from dotenv import load_dotenv
import httpx
import json
import os
import re
from google import genai
from openai import AsyncOpenAI
from typing import Optional, List
from datetime import datetime

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# The client gets the API key from the environment variable `GEMINI_API_KEY`.
gemini_client = genai.Client(api_key=GEMINI_API_KEY).aio

hg_client = AsyncOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
try:
    from deepgram import DeepgramClient
    deepgram_client = DeepgramClient(api_key=DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else None
except Exception as e:
    print(f"Warning: Deepgram client initialization failed: {e}")
    deepgram_client = None

MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "50"))
MIN_ANSWER_SENTENCES = int(os.getenv("MIN_ANSWER_SENTENCES", "2"))
MIN_ANSWER_WORDS = int(os.getenv("MIN_ANSWER_WORDS", "12"))

class AssessmentRequest(BaseModel):
    role: str
    level: str
    techstack: str
    domain: str
    selected_options: list[str]
    free_text: str

class QuestionDto(BaseModel):
    Title: str
    Content: str

class SimilarityCheckRequest(BaseModel):
    SimilarMatchQuestionList: List[QuestionDto]
    Question: QuestionDto

class HistoryStore:
    async def get_history(self, id: str) -> list[dict]:
        raise NotImplementedError

    async def append_message(self, id: str, message: dict, max_messages: Optional[int] = None) -> None:
        raise NotImplementedError

    async def clear_history(self, id: str) -> None:
        raise NotImplementedError

    async def get_last_cv(self, id: str) -> Optional[dict]:
        raise NotImplementedError

    async def set_last_cv(self, id: str, cv_json: Optional[dict]) -> None:
        raise NotImplementedError

    async def get_last_cv_pdf_url(self, id: str) -> Optional[str]:
        raise NotImplementedError

    async def set_last_cv_pdf_url(self, id: str, cv_pdf_url: Optional[str]) -> None:
        raise NotImplementedError

    async def get_transcript(self, id: str) -> Optional[dict]:
        raise NotImplementedError

    async def set_transcript(self, id: str, transcript_data: dict) -> None:
        raise NotImplementedError


class MemoryHistoryStore(HistoryStore):
    def __init__(self) -> None:
        self.histories: dict[str, list[dict]] = {}
        self.last_cvs: dict[str, Optional[dict]] = {}
        self.last_cv_pdf_urls: dict[str, Optional[str]] = {}
        self.transcripts: dict[str, dict] = {}

    async def get_history(self, id: str) -> list[dict]:
        return self.histories.get(id, [])

    async def append_message(self, id: str, message: dict, max_messages: Optional[int] = None) -> None:
        messages = self.histories.setdefault(id, [])
        messages.append(message)
        if max_messages and len(messages) > max_messages:
            self.histories[id] = messages[-max_messages:]

    async def clear_history(self, id: str) -> None:
        self.histories[id] = []

    async def get_last_cv(self, id: str) -> Optional[dict]:
        return self.last_cvs.get(id)

    async def set_last_cv(self, id: str, cv_json: Optional[dict]) -> None:
        self.last_cvs[id] = cv_json

    async def get_last_cv_pdf_url(self, id: str) -> Optional[str]:
        return self.last_cv_pdf_urls.get(id)

    async def set_last_cv_pdf_url(self, id: str, cv_pdf_url: Optional[str]) -> None:
        self.last_cv_pdf_urls[id] = cv_pdf_url

    async def get_transcript(self, id: str) -> Optional[dict]:
        return self.transcripts.get(id)

    async def set_transcript(self, id: str, transcript_data: dict) -> None:
        self.transcripts[id] = transcript_data


class PostgresHistoryStore(HistoryStore):
    def __init__(self, postgres_dsn: str, table_name: str) -> None:
        from sqlalchemy import DateTime, String, func, text
        from sqlalchemy.dialects.postgresql import JSONB
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
        from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$", table_name):
            raise ValueError("Invalid POSTGRES_HISTORY_TABLE. Use table or schema.table format.")

        schema_name = None
        simple_table_name = table_name
        if "." in table_name:
            schema_name, simple_table_name = table_name.split(".", 1)

        class Base(DeclarativeBase):
            pass

        table_args = {"schema": schema_name} if schema_name else {}

        class UserSession(Base):
            __tablename__ = simple_table_name
            __table_args__ = table_args

            id: Mapped[str] = mapped_column(String, primary_key=True)
            history: Mapped[list[dict]] = mapped_column(
                JSONB,
                nullable=False,
                server_default=text("'[]'::jsonb"),
                default=list,
            )
            last_cv: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
            last_cv_pdf_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
            transcript: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
            updated_at: Mapped[object] = mapped_column(
                DateTime(timezone=True),
                nullable=False,
                server_default=func.now(),
                onupdate=func.now(),
            )

        async_dsn = self._to_async_dsn(postgres_dsn)
        self.engine = create_async_engine(async_dsn, pool_pre_ping=True)
        self.SessionLocal = async_sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False)
        self.UserSession = UserSession
        self.Base = Base
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @staticmethod
    def _to_async_dsn(dsn: str) -> str:
        if dsn.startswith("postgresql+"):
            return dsn
        if dsn.startswith("postgresql:"):
            return re.sub(r"^postgresql:", "postgresql+psycopg:", dsn)
        return dsn

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with self.engine.begin() as conn:
                await conn.run_sync(self.Base.metadata.create_all)
            self._initialized = True

    async def get_history(self, id: str) -> list[dict]:
        await self._ensure_initialized()
        async with self.SessionLocal() as db:
            row = await db.get(self.UserSession, id)
            if not row:
                return []
            return row.history or []

    async def append_message(self, id: str, message: dict, max_messages: Optional[int] = None) -> None:
        await self._ensure_initialized()
        async with self.SessionLocal() as db:
            row = await db.get(self.UserSession, id)
            if not row:
                row = self.UserSession(id=id, history=[], last_cv=None, last_cv_pdf_url=None)
                db.add(row)
            history = list(row.history or [])
            history.append(message)
            # if max_messages:
            #     history = history[-max_messages:]
            row.history = history
            await db.commit()

    async def clear_history(self, id: str) -> None:
        await self._ensure_initialized()
        async with self.SessionLocal() as db:
            row = await db.get(self.UserSession, id)
            if not row:
                row = self.UserSession(id=id, history=[], last_cv=None, last_cv_pdf_url=None)
                db.add(row)
            else:
                row.history = []
            await db.commit()

    async def get_last_cv(self, id: str) -> Optional[dict]:
        await self._ensure_initialized()
        async with self.SessionLocal() as db:
            row = await db.get(self.UserSession, id)
            if not row:
                return None
            return row.last_cv

    async def set_last_cv(self, id: str, cv_json: Optional[dict]) -> None:
        await self._ensure_initialized()
        async with self.SessionLocal() as db:
            row = await db.get(self.UserSession, id)
            if not row:
                row = self.UserSession(id=id, history=[], last_cv=cv_json, last_cv_pdf_url=None)
                db.add(row)
            else:
                row.last_cv = cv_json
            await db.commit()

    async def get_last_cv_pdf_url(self, id: str) -> Optional[str]:
        await self._ensure_initialized()
        async with self.SessionLocal() as db:
            row = await db.get(self.UserSession, id)
            if not row:
                return None
            return row.last_cv_pdf_url

    async def set_last_cv_pdf_url(self, id: str, cv_pdf_url: Optional[str]) -> None:
        await self._ensure_initialized()
        async with self.SessionLocal() as db:
            row = await db.get(self.UserSession, id)
            if not row:
                row = self.UserSession(id=id, history=[], last_cv=None, last_cv_pdf_url=cv_pdf_url)
                db.add(row)
            else:
                row.last_cv_pdf_url = cv_pdf_url
            await db.commit()

    async def get_transcript(self, id: str) -> Optional[dict]:
        await self._ensure_initialized()
        async with self.SessionLocal() as db:
            row = await db.get(self.UserSession, id)
            if not row:
                return None
            return row.transcript

    async def set_transcript(self, id: str, transcript_data: dict) -> None:
        await self._ensure_initialized()
        async with self.SessionLocal() as db:
            row = await db.get(self.UserSession, id)
            if not row:
                row = self.UserSession(id=id, history=[], last_cv=None, last_cv_pdf_url=None, transcript=transcript_data)
                db.add(row)
            else:
                row.transcript = transcript_data
            await db.commit()


class MongoHistoryStore(HistoryStore):
    def __init__(self, mongo_uri: str, db_name: str, collection_name: str) -> None:
        from pymongo import MongoClient

        client = MongoClient(mongo_uri)
        self.collection = client[db_name][collection_name]

    async def get_history(self, id: str) -> list[dict]:
        doc = self.collection.find_one({"_id": id}, {"history": 1})
        if not doc:
            return []
        return doc.get("history", [])

    async def append_message(self, id: str, message: dict, max_messages: Optional[int] = None) -> None:
        history = await self.get_history(id)
        history.append(message)
        if max_messages:
            history = history[-max_messages:]
        self.collection.update_one(
            {"_id": id},
            {"$set": {"history": history}},
            upsert=True,
        )

    async def clear_history(self, id: str) -> None:
        self.collection.update_one({"_id": id}, {"$set": {"history": []}}, upsert=True)

    async def get_last_cv(self, id: str) -> Optional[dict]:
        doc = self.collection.find_one({"_id": id}, {"last_cv": 1})
        if not doc:
            return None
        return doc.get("last_cv")

    async def set_last_cv(self, id: str, cv_json: Optional[dict]) -> None:
        self.collection.update_one({"_id": id}, {"$set": {"last_cv": cv_json}}, upsert=True)

    async def get_last_cv_pdf_url(self, id: str) -> Optional[str]:
        doc = self.collection.find_one({"_id": id}, {"last_cv_pdf_url": 1})
        if not doc:
            return None
        return doc.get("last_cv_pdf_url")

    async def set_last_cv_pdf_url(self, id: str, cv_pdf_url: Optional[str]) -> None:
        self.collection.update_one({"_id": id}, {"$set": {"last_cv_pdf_url": cv_pdf_url}}, upsert=True)

    async def get_transcript(self, id: str) -> Optional[dict]:
        doc = self.collection.find_one({"_id": id}, {"transcript": 1})
        if not doc:
            return None
        return doc.get("transcript")

    async def set_transcript(self, id: str, transcript_data: dict) -> None:
        self.collection.update_one({"_id": id}, {"$set": {"transcript": transcript_data}}, upsert=True)


def init_history_store() -> HistoryStore:
    backend = os.getenv("HISTORY_BACKEND", "memory").lower()

    if backend in {"postgres", "postgresql"}:
        postgres_dsn = os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL") or "postgresql://postgres:postgres@localhost:5432/postgres"
        postgres_table = os.getenv("POSTGRES_HISTORY_TABLE", "InterviewUserSessions")
        try:
            return PostgresHistoryStore(postgres_dsn=postgres_dsn, table_name=postgres_table)
        except Exception as e:
            logging.error(f"Failed to initialize PostgreSQL store. Falling back to memory. Error: {e}")
            return MemoryHistoryStore()

    if backend in {"mongo", "mongodb"}:
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        mongo_db = os.getenv("MONGODB_DB", "interview_app")
        mongo_collection = os.getenv("MONGODB_COLLECTION", "user_sessions")
        try:
            return MongoHistoryStore(mongo_uri=mongo_uri, db_name=mongo_db, collection_name=mongo_collection)
        except Exception as e:
            logging.error(f"Failed to initialize MongoDB store. Falling back to memory. Error: {e}")
            return MemoryHistoryStore()

    return MemoryHistoryStore()


history_store = init_history_store()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI(
    title="FastAPI",
    description="FastAPI",
    version="1.0.0",
)
def clean_json_string(s: str) -> str:
    """
    Extract the first top-level JSON object from a string.
    Handles nested braces properly.
    """
    brace_count = 0
    start_idx = None

    for idx, char in enumerate(s):
        if char == '{':
            if brace_count == 0:
                start_idx = idx
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                # Extract substring and return
                return s[start_idx:idx+1]
        elif char == '[':
            if brace_count == 0:
                start_idx = idx
            brace_count += 1
        elif char == ']':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                 return s[start_idx:idx+1]

    # Fallback: return original string if no JSON found
    return s

def clean_json_array_string(s: str) -> str:
    """
    Extract the first top-level JSON object or array from a string.
    Handles nested braces/brackets properly.
    """

    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    s = s.strip()

    if not s:
        return "{}"

    start_char = '{'
    end_char = '}'
    if s.startswith('['):
        start_char = '['
        end_char = ']'

    count = 0

    for idx, char in enumerate(s):
        if char == start_char:
            count += 1
        elif char == end_char:
            count -= 1
            if count == 0:
                return s[:idx+1]

    # Fallback: return original string if no JSON found
    return s


def _extract_last_assistant_question(messages: list[dict]) -> Optional[str]:
    for item in reversed(messages):
        if item.get("role") == "assistant":
            content = (item.get("content") or "").strip()
            if content:
                return content
    return None


def _sentence_count(text: str) -> int:
    parts = re.split(r"[.!?]+", text.strip())
    return len([p for p in parts if p.strip()])


def _is_generic_answer(text: str) -> bool:
    normalized = text.strip().lower()
    generic_phrases = {
        "yes", "no", "ok", "okay", "sure", "maybe", "idk", "i don't know",
        "not sure", "n/a", "na", "none", "same", "all good", "fine",
    }
    if normalized in generic_phrases:
        return True
    if len(normalized.split()) <= 3 and normalized.endswith("?"):
        return True
    return False


def validate_candidate_answer(candidate_answer: Optional[str], previous_question: Optional[str]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if candidate_answer is None:
        return False, ["missing answer"]

    answer = candidate_answer.strip()
    if not answer:
        reasons.append("empty or whitespace")
        return False, reasons

    if _sentence_count(answer) < MIN_ANSWER_SENTENCES:
        reasons.append(f"fewer than {MIN_ANSWER_SENTENCES} sentences")
    if len(answer.split()) < MIN_ANSWER_WORDS:
        reasons.append(f"fewer than {MIN_ANSWER_WORDS} words")
    if _is_generic_answer(answer):
        reasons.append("too generic")

    # Lightweight topic-fit heuristic to catch obvious non-answers.
    if previous_question:
        q_tokens = {w for w in re.findall(r"[a-zA-Z]{4,}", previous_question.lower())}
        a_tokens = {w for w in re.findall(r"[a-zA-Z]{4,}", answer.lower())}
        if q_tokens and len(q_tokens.intersection(a_tokens)) == 0 and len(answer.split()) < 25:
            reasons.append("does not clearly address the previous question")

    return len(reasons) == 0, reasons


def extract_single_question(text: str) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return "Could you briefly introduce yourself and your relevant background?"
    if "?" in cleaned:
        return cleaned.split("?")[0].strip() + "?"
    return cleaned

async def parse_cv_to_json(cv_text: str) -> dict:
    prompt = f"""
You are a CV parsing assistant.

You will receive the text of a CV. Some personal information like names, emails, phone numbers, and addresses has already been anonymized. 

Your task is to extract structured information from the CV and output it as a JSON object with the following format:

{{
  "apply_for": {{"job_title": ""}},
  "skills": [],
  "languages": [],
  "experiences": [],
  "certifications": []
}}

Rules:
1. Only include information explicitly mentioned in the CV.
2. Preserve anonymized placeholders as-is.
3. Organize experiences in chronological order, most recent first.
4. If a field is not available, leave it empty or as an empty array/object.
5. Make JSON valid.
6. Output JSON only. No text outside JSON.
7. Do not add explanations or commentary.

Here is the CV text:

\"\"\"{cv_text}\"\"\"
"""

    response = await gemini_client.models.generate_content(
        model="gemma-3-27b-it",
        contents=[prompt]
    )
    structured_json = json.loads(clean_json_string(response.text))
    return structured_json

# async def generate_interview_questions(cv_text: str):
#     """
#     A background task to generate interview questions based on the CV.
#     """
#     prompt = f"""
# You are an interview question generator.
#
# Based on the following CV text, generate 5–10 relevant interview questions tailored to the candidate's skills, experience, and background.
#
# Return the result as a **JSON array of strings only**, with no additional text, explanation, or wrapping.
#
# Example format:
# {{
#   "questions": [
#     {{
#       "id": "q_001",
#       "type": "type1",
#       "difficulty": "difficulty1",
#       "target_skill": "skill1",
#       "experience_ref": "company1",
#       "question": "question1",
#       "answer_guide": "answer_guide1"
#     }},
#     {{
#       "id": "q_002",
#       "type": "type2",
#       "difficulty": "difficulty2",
#       "target_skill": "skill2",
#       "experience_ref": "company2",
#       "question": "question2",
#       "answer_guide": "answer_guide2"
#     }},
#     {{
#       "id": "...",
#       "type": "...",
#       "difficulty": "...",
#       "target_skill": "...",
#       "experience_ref": "...",
#       "question": "...",
#       "answer_guide": "..."
#     }}
#   ]
# }}
#
# CV Text:
# \"\"\"{cv_text}\"\"\"
#
# 1. Make JSON valid (use double quotes and proper syntax).
# 2. **Output JSON only. Do not include any text outside the JSON object.**
# 3. Do not add explanations, notes, or commentary.
# """
#     try:
#         response = await chat_with_fallback(prompt)
#         response = clean_json_string(response)
#         # For now, we just log the questions. In a real app, we might save them to a database.
#         logging.info(f"[BackgroundTask] Generated Interview Questions:\n{response}")
#     except Exception as e:
#         logging.error(f"[BackgroundTask] Error generating interview questions: {e}")

async def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Async wrapper for PyMuPDF Layout parser."""
    return await asyncio.to_thread(_extract_text_layout_aware, pdf_content)


def _extract_text_layout_aware(pdf_content: bytes) -> str:
    """Extracts structured text (headings, paragraphs) using PyMuPDF-Layout."""
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    md = pymupdf4llm.to_markdown(doc)  # Markdown preserves headings, bold, italic, lists
    doc.close()
    return md

@app.post("/api/extract-cv")
async def extract_cv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    id: str = Query("default"),
):
    content = await file.read()

    if file.filename.endswith(".pdf"):
        text = await extract_text_from_pdf(content)
    else:
        text = content.decode("utf-8")

    try:
        cv_json = await parse_cv_to_json(text)
        await history_store.set_last_cv(id, cv_json)
        # Add background task to run after the response is sent
        # background_tasks.add_task(generate_interview_questions, cv_json)
    except Exception as e:
        logging.error(f"Failed to parse CV to JSON: {e}")
        cv_json = {"error": "Failed to parse CV to JSON"}
        await history_store.set_last_cv(id, None)

    return cv_json

@app.get("/api/last-cv")
async def last_cv(id: str = Query("default")):
    cv_json = await history_store.get_last_cv(id)
    if cv_json is None:
        return {"error": "No CV has been parsed yet"}
    return cv_json

def is_empty_request(req: AssessmentRequest):
    return (
        req.role in ["", "string", None] and
        req.level in ["", "string", None] and
        req.techstack in ["", "string", None] and
        req.free_text in ["", "string", None]
    )

@app.post("/api/generate-assessment")
async def generate_assessment(request: AssessmentRequest):
    try:
        if is_empty_request(request):
            return {
                "status": "need_input",
                "question": "What are you trying to achieve right now, and what do you feel you're missing?"
            }

        with open("skill-references.json", "r", encoding="utf-8") as f:
            skill_reference = json.load(f)

        skill_reference_str = json.dumps(skill_reference, indent=2)

        prompt = f"""
You are an intelligent technical interviewer inside a career development platform.

Your job is to guide a new user through a short assessment to understand:
1. Their CURRENT level (Phase A)
2. Their TARGET expectations (Phase B)

The user has just signed up and may not fully understand the system.

-------------------------------------

=== CONTEXT ===
- The user is new
- You must guide them step by step
- Keep the conversation natural and simple
- Do NOT overwhelm the user

-------------------------------------

=== SKILL REFERENCE ===
{skill_reference_str}

-------------------------------------

    === INPUT (may be partial or empty) ===
    Role: {request.role}
    Level: {request.level}
    TechStack: {request.techstack}
    Domain: {request.domain}
    User Intent: {request.free_text}

-------------------------------------

=== STRATEGY ===

STEP 1 – CONTEXT QUESTION (MANDATORY)
- Ask 1 natural, friendly question
- Purpose: understand user's goal and background
- Example tone:
  "What are you trying to achieve right now?"
  "Can you tell me a bit about your experience and what you're aiming for?"

-------------------------------------

STEP 2 – PHASE A (CURRENT LEVEL)
- Generate exactly 5 questions
- Each question must:
  - Map to ONE skill from skill reference
  - Focus on REAL experience (what user can do)
- Questions must feel practical (not theoretical)
- Each question has 4 options:
  - None → no experience
  - Basic → simple usage
  - Intermediate → used in real projects
  - Advanced → confident in real-world scenarios

-------------------------------------

STEP 3 – PHASE B (TARGET)
- Generate exactly 5 questions
- ONLY after Phase A
- Each question must:
  - Map to ONE skill from skill reference
  - Ask what level user WANTS to reach
- Each question has 4 options:
  - Beginner
  - Comfortable
  - Confident
  - Expert

-------------------------------------

=== IMPORTANT RULES ===

- MUST only use skills from skill reference
- DO NOT invent new skills
- DO NOT duplicate skills across Phase A and Phase B
- Questions must be short, clear, and practical
- Tone must be friendly, not formal
- Avoid technical jargon if possible
- Make questions feel like a real conversation, not a survey form

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

        response = await gemini_client.models.generate_content(
            model="gemma-3-27b-it",
            contents=[prompt]
        )

        data = json.loads(clean_json_string(response.text))
        logging.info(f"LLM Assessment Response: {json.dumps(data, indent=2)}")

        if len(data.get("phaseA", [])) != 5:
            raise Exception("Invalid Phase A")

        if len(data.get("phaseB", [])) != 5:
            raise Exception("Invalid Phase B")

        return {
            "status": "success",
            "assessment": data
        }

    except Exception as e:
        logging.error(f"Error in generate_assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/last-cv-pdf-url")
async def set_last_cv_pdf_url(
    cv_pdf_url: str = Body(...),
    id: str = Query("default"),
):
    normalized = cv_pdf_url.strip()
    if not re.match(r"^https?://", normalized, re.IGNORECASE):
        raise HTTPException(status_code=400, detail="cv_pdf_url must be a valid http(s) URL")
    if not normalized.lower().split("?", 1)[0].endswith(".pdf"):
        raise HTTPException(status_code=400, detail="cv_pdf_url must point to a .pdf file")

    await history_store.set_last_cv_pdf_url(id, normalized)
    return {"status": "success", "id": id, "cv_pdf_url": normalized}


@app.get("/api/last-cv-pdf-url")
async def last_cv_pdf_url(id: str = Query("default")):
    cv_pdf_url = await history_store.get_last_cv_pdf_url(id)
    if cv_pdf_url is None:
        return {"error": "No CV PDF URL has been saved yet"}
    return {"id": id, "cv_pdf_url": cv_pdf_url}

@app.post("/api/transcript")
async def extract_transcript(
    file: UploadFile = File(...),
    id: str = Query("default"),
):
    if deepgram_client is None:
        raise HTTPException(status_code=503, detail="Deepgram service is not configured")

    try:
        audio_data = await file.read()

        response = await asyncio.to_thread(
            deepgram_client.listen.v1.media.transcribe_file,
            request=audio_data,
            model="nova-3",
            smart_format=True,
        )

        transcript_text = response.results.channels[0].alternatives[0].transcript

        prompt = f"""
        Extract from the transcript text all the question related to IT job which is Software Engineering, Frontend Dev, Backend Dev, Fullstack Dev, Business Analyst, Tester, etc.
        Refine the question so it is grammatically correct.
        Set the short name for question which is Problem Title to title, and the full description for the question which is Problem Statement as content both for the json array.
        
        IMPORTANT: For the 'content', provide only the question itself. Do not include the answer or any explanation.
        
        Transcript:
        {transcript_text}
        
        Output JSON only as an array of objects.
        Example: [{{ "title": "Question Title", "content": "The refined question text?" }}]
        """
        questions_list = []
        try:
            llm_response = await hg_client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct:novita",
                messages=[{"role": "user", "content": prompt}]
            )
            questions_list = json.loads(clean_json_string(llm_response.choices[0].message.content))
        except Exception as e:
            logging.error(f"Error extracting questions: {e}")

        transcript_data = {
            "text": transcript_text,
            "timestamp": datetime.utcnow().isoformat(),
            "filename": file.filename,
            "user_id": id,
            "metadata": {
                "model": "nova-3",
                "smart_format": True,
            }
        }

        await history_store.set_transcript(id, transcript_data)

        return {
            "status": "success",
            "transcript": transcript_text,
            "question list": questions_list,
        }
    except Exception as e:
        logging.error(f"Failed to extract transcript: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to extract transcript: {str(e)}")

@app.get("/api/transcript")
async def get_transcript(id: str = Query("default")):
    try:
        transcript_data = await history_store.get_transcript(id)
        if transcript_data is None:
            raise HTTPException(status_code=404, detail="Transcript not found")
        return transcript_data
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to retrieve transcript: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to retrieve transcript: {str(e)}")

@app.post("/api/check-similarity")
async def check_question_similarity(request: SimilarityCheckRequest):
    existing_questions_text = "\n".join(
        [f"- Title: {q.Title}\n  Content: {q.Content}"
         for i, q in enumerate(request.SimilarMatchQuestionList)]
    )

    # Improved Prompt using Chain of Thought (CoT) and structured reasoning
    prompt = f"""
    You are a technical interview question deduplication expert. Your task is to determine if a new "Target Question" is a duplicate of any question in a list of "Existing Questions".

    **Definition of Duplicate:**
    A question is a duplicate if it:
    1. Is semantically identical (same meaning, different words).
    2. Is a direct variation or sub-problem (e.g., "Reverse Linked List II" is a variation of "Reverse Linked List").
    3. Solves the exact same core algorithmic problem.
    4. IMPORTANT: Questions that use the same data structure (e.g., Linked List) but solve different problems (e.g., 'reordering' vs. 'reversing') are NOT duplicates.

    **Target Question:**
    - Title: "{request.Question.Title}"
    - Content: "{request.Question.Content}"

    **Existing Questions:**
    {existing_questions_text}

    **Your Task:**
    1. Analyze the Target Question to understand its core algorithmic goal.
    2. Iterate through EACH Existing Question and compare it with the Target Question.
    3. For each comparison, briefly reason whether they are duplicates based on the definition.
    4. Conclude with a final decision.

    **Output Format:**
    Provide a brief reasoning block followed by the final JSON object.
    
    Example Output:
    Reasoning:
    - Target vs Question 1: Different because...
    - Target vs Question 2: Duplicate because...
    
    ```json
    {{ "is_new": false }}
    ```
    (Or `{{ "is_new": true }}` if no duplicates found)
    """

    try:
        logging.info(f"Full prompt: {prompt}")
        # response = await gemini_client.models.generate_content(
        #     model="gemma-3-27b-it",
        #     contents=[prompt]
        # )

        response = await hg_client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct:novita",
            messages=[{"role": "user", "content": prompt}]
        )

        # llm_output = json.loads(clean_json_string(response.text))
        llm_output = json.loads(clean_json_string(response.choices[0].message.content))

        logging.info(f"LLM Output: {response.choices[0].message.content}")

        if llm_output.get("is_new", False):
            return {
                "title": request.Question.Title,
                "content": request.Question.Content
            }
        else:
            return {}

    except Exception as e:
        logging.error(f"Error checking similarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def interview_history(id: str = Query("default")):
    history = await history_store.get_history(id)
    if not history:
        return {"error": "No interview is ongoing yet"}
    return history

@app.post("/api/question")
async def questioning(
    user_answer: Optional[str] = Body(default=None, embed=True),
    id: str = Query("default"),
):
    last_parsed_cv = await history_store.get_last_cv(id)
    if last_parsed_cv is None:
        raise HTTPException(status_code=400, detail="No CV has been parsed yet for this id")

    history = await history_store.get_history(id)
    previous_question = _extract_last_assistant_question(history)
    is_first_turn = previous_question is None

    if user_answer is not None:
        await history_store.append_message(
            id,
            {"role": "user", "content": user_answer},
            max_messages=MAX_HISTORY_MESSAGES,
        )

    if is_first_turn:
        llm_messages = [{
            "role": "user",
            "content": f"""
        You are an AI interviewer.
        Ask exactly one question in one line.
        No explanation and no extra text.
        Use only the CV content.
        
        This is the first interview turn.
        Ask the candidate to briefly introduce themselves and explain their background for the applied role.
        
        CV:
        {last_parsed_cv}
        """,
        }]
    else:
        is_valid, reasons = validate_candidate_answer(user_answer, previous_question)
        if not is_valid:
            llm_messages = [{
                "role": "user",
                "content": f"""
            You are an AI interviewer.
            Ask exactly one question in one line.
            No explanation and no extra text.
            Stay on the SAME topic as the previous question.
            The latest answer is invalid, so ask the same question again or a close paraphrase.
            
            Invalid reason(s): {", ".join(reasons)}
            Previous question: {previous_question}
            Candidate answer: {user_answer}
            CV:
            {last_parsed_cv}
            """,
            }]
        else:
            updated_history = await history_store.get_history(id)
            llm_messages = [{
                "role": "user",
                "content": f"""
            You are an AI interviewer conducting a structured interview.
            Ask exactly one question in one line.
            No explanation and no extra text.
            Use only information present in the CV. Do not invent data.
            
            Interview stages in strict order:
            1) Self introduction
            2) CV and experience
            3) Technical and problem solving
            4) Mindset and teamwork
            
            Move to the next suitable question based on the transcript.
            Increase difficulty gradually.
            
            CV:
            {last_parsed_cv}
            
            Transcript:
            {updated_history}
            """,
            }]

    response = await hg_client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct:novita",
        messages=llm_messages
    )
    assistant_text = extract_single_question(response.choices[0].message.content)
    role = response.choices[0].message.role

    await history_store.append_message(
        id,
        {"role": role, "content": assistant_text},
        max_messages=MAX_HISTORY_MESSAGES,
    )
    return {
        "question": assistant_text
    }

@app.post("/api/history/clear")
async def clear_history(id: str = Query("default")):
    await history_store.clear_history(id)
    return {"status": "success", "message": "Interview history cleared."}

@app.get("/api/data")
def get_sample_data():
    return {
        "data": [
            {"id": 1, "name": "Sample Item 1", "value": 100},
            {"id": 2, "name": "Sample Item 2", "value": 200},
            {"id": 3, "name": "Sample Item 3", "value": 300}
        ],
        "total": 3,
        "timestamp": "2024-01-01T00:00:00Z"
    }


@app.get("/api/items/{item_id}")
def get_item(item_id: int):
    return {
        "item": {
            "id": item_id,
            "name": "Sample Item " + str(item_id),
            "value": item_id * 100
        },
        "timestamp": "2024-01-01T00:00:00Z"
    }


@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FastAPI</title>
        <link rel="icon" type="image/x-icon" href="/favicon.ico">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
                background-color: #000000;
                color: #ffffff;
                line-height: 1.6;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }
            
            header {
                border-bottom: 1px solid #333333;
                padding: 0;
            }
            
            nav {
                max-width: 1200px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                padding: 1rem 2rem;
                gap: 2rem;
            }
            
            .logo {
                font-size: 1.25rem;
                font-weight: 600;
                color: #ffffff;
                text-decoration: none;
            }
            
            .nav-links {
                display: flex;
                gap: 1.5rem;
                margin-left: auto;
            }
            
            .nav-links a {
                text-decoration: none;
                color: #888888;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                transition: all 0.2s ease;
                font-size: 0.875rem;
                font-weight: 500;
            }
            
            .nav-links a:hover {
                color: #ffffff;
                background-color: #111111;
            }
            
            main {
                flex: 1;
                max-width: 1200px;
                margin: 0 auto;
                padding: 4rem 2rem;
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
            }
            
            .hero {
                margin-bottom: 3rem;
            }
            
            .hero-code {
                margin-top: 2rem;
                width: 100%;
                max-width: 900px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            }
            
            .hero-code pre {
                background-color: #0a0a0a;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 1.5rem;
                text-align: left;
                grid-column: 1 / -1;
            }
            
            h1 {
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 1rem;
                background: linear-gradient(to right, #ffffff, #888888);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .subtitle {
                font-size: 1.25rem;
                color: #888888;
                margin-bottom: 2rem;
                max-width: 600px;
            }
            
            .cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1.5rem;
                width: 100%;
                max-width: 900px;
            }
            
            .card {
                background-color: #111111;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 1.5rem;
                transition: all 0.2s ease;
                text-align: left;
            }
            
            .card:hover {
                border-color: #555555;
                transform: translateY(-2px);
            }
            
            .card h3 {
                font-size: 1.125rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
                color: #ffffff;
            }
            
            .card p {
                color: #888888;
                font-size: 0.875rem;
                margin-bottom: 1rem;
            }
            
            .card a {
                display: inline-flex;
                align-items: center;
                color: #ffffff;
                text-decoration: none;
                font-size: 0.875rem;
                font-weight: 500;
                padding: 0.5rem 1rem;
                background-color: #222222;
                border-radius: 6px;
                border: 1px solid #333333;
                transition: all 0.2s ease;
            }
            
            .card a:hover {
                background-color: #333333;
                border-color: #555555;
            }
            
            .status-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                background-color: #0070f3;
                color: #ffffff;
                padding: 0.25rem 0.75rem;
                border-radius: 20px;
                font-size: 0.75rem;
                font-weight: 500;
                margin-bottom: 2rem;
            }
            
            .status-dot {
                width: 6px;
                height: 6px;
                background-color: #00ff88;
                border-radius: 50%;
            }
            
            pre {
                background-color: #0a0a0a;
                border: 1px solid #333333;
                border-radius: 6px;
                padding: 1rem;
                overflow-x: auto;
                margin: 0;
            }
            
            code {
                font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
                font-size: 0.85rem;
                line-height: 1.5;
                color: #ffffff;
            }
            
            /* Syntax highlighting */
            .keyword {
                color: #ff79c6;
            }
            
            .string {
                color: #f1fa8c;
            }
            
            .function {
                color: #50fa7b;
            }
            
            .class {
                color: #8be9fd;
            }
            
            .module {
                color: #8be9fd;
            }
            
            .variable {
                color: #f8f8f2;
            }
            
            .decorator {
                color: #ffb86c;
            }
            
            @media (max-width: 768px) {
                nav {
                    padding: 1rem;
                    flex-direction: column;
                    gap: 1rem;
                }
                
                .nav-links {
                    margin-left: 0;
                }
                
                main {
                    padding: 2rem 1rem;
                }
                
                h1 {
                    font-size: 2rem;
                }
                
                .hero-code {
                    grid-template-columns: 1fr;
                }
                
                .cards {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <header>
            <nav>
                <a href="/" class="logo">Vercel + FastAPI</a>
                <div class="nav-links">
                    <a href="/docs">API Docs</a>
                    <a href="/api/data">API</a>
                </div>
            </nav>
        </header>
        <main>
            <div class="hero">
                <h1>Vercel + FastAPI</h1>
                <div class="hero-code">
                    <pre><code><span class="keyword">from</span> <span class="module">fastapi</span> <span class="keyword">import</span> <span class="class">FastAPI</span>

<span class="variable">app</span> = <span class="class">FastAPI</span>()

<span class="decorator">@app.get</span>(<span class="string">"/"</span>)
<span class="keyword">def</span> <span class="function">read_root</span>():
    <span class="keyword">return</span> {<span class="string">"Python"</span>: <span class="string">"on Vercel"</span>}</code></pre>
                </div>
            </div>
            
            <div class="cards">
                <div class="card">
                    <h3>Interactive API Docs</h3>
                    <p>Explore this API's endpoints with the interactive Swagger UI. Test requests and view response schemas in real-time.</p>
                    <a href="/docs">Open Swagger UI →</a>
                </div>
                
                <div class="card">
                    <h3>Sample Data</h3>
                    <p>Access sample JSON data through our REST API. Perfect for testing and development purposes.</p>
                    <a href="/api/data">Get Data →</a>
                </div>
                
            </div>
        </main>
    </body>
    </html>
    """


import uvicorn
from api.api import create_app
app = create_app()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)