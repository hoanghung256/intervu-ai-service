import json
import logging
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Union
from fastapi import APIRouter, Depends, UploadFile, File, Form, Query, Body, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from api.dtos import (
    AssessmentRequest, AssessmentResponse, SimilarityCheckRequest,
    CVResponse, JDResponse, TranscriptResponse, AskRequest, AnswerResponse,
    ExtractQuestionsRequest, ExtractQuestionsResponse,
    CvEvaluationResponse, EvaluateAssessmentRequestDto, SurveySummaryResultDto, LLMUsage
)
from api.roadmap_dto import RoadmapRequest, RoadmapResponse, RoadmapProgressUpdateRequest
from infrastructure.model_provider.llm_provider import LLMProvider
from infrastructure.repository.history_repository import HistoryRepository
from infrastructure.repository.repository_factory import get_history_repository
from application.services.cv_service import CVService
from application.services.assessment_service import AssessmentService
from application.services.interview_service import InterviewService
from application.services.similarity_service import SimilarityService
from application.services.transcript_service import TranscriptService
from application.services.roadmap_service import RoadmapService
from application.services.roadmap_progress_service import RoadmapProgressService
from application.llm.inference_usecase import InferenceUseCase
from domain.entities.query import Query as QueryEntity
from infrastructure.external_services.deepgram_service import DeepgramService
from infrastructure.external_services.elevenlabs_service import ElevenLabsService
from infrastructure.env_constants import (
    ENV_TRANSCRIPTION_PROVIDER,
    DEFAULT_TRANSCRIPTION_PROVIDER,
    TRANSCRIPTION_PROVIDER_DEEPGRAM,
    TRANSCRIPTION_PROVIDER_ELEVENLABS,
)

router = APIRouter(prefix="/api")


def _is_pdf_upload(file: UploadFile) -> bool:
    filename = (file.filename or "").lower()
    return filename.endswith(".pdf") or file.content_type == "application/pdf"


def _decode_text_upload(content: bytes) -> str:
    # Support common plain-text encodings while rejecting binary payloads.
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("text-decode", content, 0, 1, "Unsupported or binary file")

# Dependency injection
def get_llm_provider() -> LLMProvider:
    return LLMProvider()

def get_history_repo() -> HistoryRepository:
    return get_history_repository()

def get_cv_service(llm_provider: LLMProvider = Depends(get_llm_provider)) -> CVService:
    return CVService(llm_provider)

def get_assessment_service(llm_provider: LLMProvider = Depends(get_llm_provider)) -> AssessmentService:
    return AssessmentService(llm_provider)

def get_interview_service(
    llm_provider: LLMProvider = Depends(get_llm_provider),
    history_repo: HistoryRepository = Depends(get_history_repo)
) -> InterviewService:
    return InterviewService(llm_provider, history_repo)

def get_similarity_service(llm_provider: LLMProvider = Depends(get_llm_provider)) -> SimilarityService:
    return SimilarityService(llm_provider)

def get_transcript_service(llm_provider: LLMProvider = Depends(get_llm_provider)) -> TranscriptService:
    return TranscriptService(llm_provider)

def get_roadmap_service(llm_provider: LLMProvider = Depends(get_llm_provider)) -> RoadmapService:
    return RoadmapService(llm_provider)

def get_roadmap_progress_service(llm_provider: LLMProvider = Depends(get_llm_provider)) -> RoadmapProgressService:
    return RoadmapProgressService(llm_provider)

def get_transcription_provider() -> Union[DeepgramService, ElevenLabsService]:
    provider_name = os.getenv(ENV_TRANSCRIPTION_PROVIDER, DEFAULT_TRANSCRIPTION_PROVIDER).strip().lower()

    if provider_name == TRANSCRIPTION_PROVIDER_ELEVENLABS:
        return ElevenLabsService()

    if provider_name != TRANSCRIPTION_PROVIDER_DEEPGRAM:
        logging.warning(
            f"Unknown transcription provider '{provider_name}'. "
            f"Falling back to '{TRANSCRIPTION_PROVIDER_DEEPGRAM}'."
        )

    return DeepgramService()

def get_inference_usecase(llm_provider: LLMProvider = Depends(get_llm_provider)) -> InferenceUseCase:
    return InferenceUseCase(llm_provider)

# --- Legacy/Initial Inference Endpoints ---

@router.post("/inference", response_model=AnswerResponse)
async def ask(payload: AskRequest, inference_usecase: InferenceUseCase = Depends(get_inference_usecase)) -> AnswerResponse:
    query = QueryEntity(prompt=payload.prompt)
    content, usage = await inference_usecase.generate_response(query)
    return AnswerResponse(answer=content, usage=LLMUsage(**usage))


@router.post("/extract-document", response_model=Union[CVResponse, JDResponse])
async def extract_document(
    file: UploadFile = File(...),
    doc_type: str = Form("cv"),
    id: str = Query("default"),
    cv_service: CVService = Depends(get_cv_service),
    history_repo: HistoryRepository = Depends(get_history_repo)
):
    valid_types = ["cv", "jd"]
    if doc_type.lower() not in valid_types:
        raise HTTPException(status_code=400, detail=f"doc_type must be one of {valid_types}")

    content = await file.read()
    filename = (file.filename or "upload").lower()

    if _is_pdf_upload(file):
        text = await cv_service.extract_text_from_pdf(content)
    else:
        allowed_text_exts = {".txt", ".md", ".markdown", ".json", ".yaml", ".yml", ".csv", ".log"}
        extension = Path(filename).suffix
        if extension and extension not in allowed_text_exts:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported document format for /extract-document. "
                    "Please upload a PDF, or a plain-text file (.txt/.md/.json/.yaml/.csv/.log)."
                ),
            )

        try:
            text = _decode_text_upload(content)
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unable to decode uploaded file as text. "
                    "Please upload a UTF text file or PDF."
                ),
            )

    try:
        parsed_json, usage = await cv_service.parse_document_to_json(text, doc_type)
        if doc_type.lower() == "cv":
            await history_repo.set_last_cv(id, parsed_json)
        parsed_json["usage"] = usage
        return parsed_json
    except RuntimeError as e:
        logging.error(f"Upstream LLM service unavailable while parsing {doc_type}: {e}")
        if doc_type.lower() == "cv":
            await history_repo.set_last_cv(id, None)
        raise HTTPException(
            status_code=503,
            detail="Document parsing service is temporarily unavailable. Please retry shortly."
        )
    except ValueError as e:
        logging.error(f"Invalid LLM output while parsing {doc_type}: {e}")
        if doc_type.lower() == "cv":
            await history_repo.set_last_cv(id, None)
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse {doc_type} content into structured JSON."
        )
    except Exception as e:
        import traceback
        logging.error(f"Failed to parse document to JSON: {e}")
        logging.error(traceback.format_exc())
        if doc_type.lower() == "cv":
            await history_repo.set_last_cv(id, None)
        raise HTTPException(status_code=500, detail=f"Unexpected error while parsing {doc_type}.")

@router.get("/last-cv")
async def get_last_cv(id: str = Query("default"), history_repo: HistoryRepository = Depends(get_history_repo)):
    cv_json = await history_repo.get_last_cv(id)
    if cv_json is None:
        return {"error": "No CV has been parsed yet"}
    return cv_json

@router.post("/generate-assessment", response_model=AssessmentResponse)
async def generate_assessment(
    request: AssessmentRequest,
    assessment_service: AssessmentService = Depends(get_assessment_service)
):
    try:
        if assessment_service.is_empty_request(request):
            return {
                "status": "need_input",
                "question": "What are you trying to achieve right now, and what do you feel you're missing?",
                "usage": LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            }

        assessment_data, usage = await assessment_service.generate_assessment(request)
        logging.info(f"Generated assessment: {assessment_data}")
        return {
            "status": "success",
            "context_question": assessment_data.get("contextQuestion", ""),
            "phaseA": assessment_data.get("phaseA", []),
            "phaseB": assessment_data.get("phaseB", []),
            "usage": LLMUsage(**usage),
        }
    except Exception as e:
        logging.error(f"Error in generate_assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate-assessment", response_model=SurveySummaryResultDto)
async def evaluate_assessment(
    request: EvaluateAssessmentRequestDto,
    assessment_service: AssessmentService = Depends(get_assessment_service)
):
    try:
        gap_payload = request.gapJson or {}
        return await assessment_service.evaluate_answer_json(
            answer=request.answer,
            gap_input=gap_payload,
        )
    except Exception as e:
        logging.error(f"Error in evaluate_assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/last-cv-pdf-url")
async def set_last_cv_pdf_url(
    cv_pdf_url: str = Body(..., embed=True),
    id: str = Query("default"),
    history_repo: HistoryRepository = Depends(get_history_repo)
):
    normalized = cv_pdf_url.strip()
    if not re.match(r"^https?://", normalized, re.IGNORECASE):
        raise HTTPException(status_code=400, detail="cv_pdf_url must be a valid http(s) URL")
    await history_repo.set_last_cv_pdf_url(id, normalized)
    return {"status": "success", "id": id, "cv_pdf_url": normalized}

@router.get("/last-cv-pdf-url")
async def get_last_cv_pdf_url(id: str = Query("default"), history_repo: HistoryRepository = Depends(get_history_repo)):
    cv_pdf_url = await history_repo.get_last_cv_pdf_url(id)
    if cv_pdf_url is None:
        return {"error": "No CV PDF URL has been saved yet"}
    return {"id": id, "cv_pdf_url": cv_pdf_url}

@router.post("/transcript", response_model=TranscriptResponse)
async def extract_transcript(
    file: UploadFile = File(...),
    id: str = Query("default"),
    transcription_provider: Union[DeepgramService, ElevenLabsService] = Depends(get_transcription_provider),
    transcript_service: TranscriptService = Depends(get_transcript_service),
    tags: Optional[str] = Form(None)
):
    if not transcription_provider.is_configured:
        provider_name = transcription_provider.provider_name
        raise HTTPException(status_code=503, detail=f"{provider_name} service is not configured")

    try:
        audio_data = await file.read()
        transcript_text = await transcription_provider.transcribe_file(audio_data)
        if not transcript_text:
            raise HTTPException(status_code=400, detail="Transcription failed")

        parsed_tags = []
        if tags:
            try:
                parsed_tags = json.loads(tags)
            except Exception:
                # If not JSON, maybe it's a comma separated string
                parsed_tags = [t.strip() for t in tags.split(",") if t.strip()]

        questions_list, usage = await transcript_service.extract_questions_from_text(transcript_text, parsed_tags)

        # Extract all tags used in the question list to return them at the top level
        used_tags = set()
        for q in questions_list:
            if "tags" in q and isinstance(q["tags"], list):
                used_tags.update(q["tags"])

        return {
            "status": "success",
            "transcript": transcript_text,
            "question_list": questions_list,
            "tags": list(used_tags),
            "usage": LLMUsage(**usage),
        }
    except Exception as e:
        logging.error(f"Failed to extract transcript: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/extract-questions", response_model=ExtractQuestionsResponse)
async def extract_questions_endpoint(
    request: ExtractQuestionsRequest,
    transcript_service: TranscriptService = Depends(get_transcript_service)
):
    try:
        questions, usage = await transcript_service.extract_questions_from_text(
            request.transcript_text,
            request.tags
        )

        # Extract used tags
        used_tags = set()
        for q in questions:
            if "tags" in q and isinstance(q["tags"], list):
                used_tags.update(q["tags"])

        return {
            "status": "success",
            "question_list": questions,
            "tags": list(used_tags),
            "usage": LLMUsage(**usage),
        }
    except Exception as e:
        logging.error(f"Failed to extract questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-roadmap", response_model=RoadmapResponse)
async def generate_roadmap(
    request: RoadmapRequest,
    roadmap_service: RoadmapService = Depends(get_roadmap_service)
):
    try:
        roadmap_data, usage = await roadmap_service.generate_roadmap(request)
        return {"status": "success", "roadmap": roadmap_data, "usage": LLMUsage(**usage)}
    except ValueError as e:
        logging.error(f"Invalid roadmap output: {e}")
        return {"status": "failed", "error": str(e)}
    except Exception as e:
        logging.error(f"Error in generate_roadmap: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update-roadmap-progress", response_model=RoadmapResponse)
async def update_roadmap_progress(
    request: RoadmapProgressUpdateRequest,
    roadmap_progress_service: RoadmapProgressService = Depends(get_roadmap_progress_service)
):
    try:
        updated_roadmap, usage = await roadmap_progress_service.update_roadmap_progress(request)
        return {"status": "success", "roadmap": updated_roadmap, "usage": LLMUsage(**usage)}
    except ValueError as e:
        logging.error(f"Invalid roadmap progress output: {e}")
        return {"status": "failed", "error": str(e)}
    except Exception as e:
        logging.error(f"Error in update_roadmap_progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-similarity")
async def check_question_similarity(
    request: SimilarityCheckRequest,
    similarity_service: SimilarityService = Depends(get_similarity_service)
):
    try:
        llm_output, usage = await similarity_service.check_similarity(request)
        usage_model = LLMUsage(**usage)
        if llm_output.get("is_new", False):
            return {
                "title": request.Question.Title,
                "content": request.Question.Content,
                "usage": usage_model,
            }
        return {"usage": usage_model}
    except Exception as e:
        logging.error(f"Error checking similarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_interview_history(id: str = Query("default"), history_repo: HistoryRepository = Depends(get_history_repo)):
    history = await history_repo.get_history(id)
    if not history:
        return {"error": "No interview history found"}
    return history

@router.post("/question")
async def get_question(
    user_answer: Optional[str] = Body(default=None, embed=True),
    id: str = Query("default"),
    interview_service: InterviewService = Depends(get_interview_service)
):
    try:
        question, usage = await interview_service.get_next_question(id, user_answer)
        return {"question": question, "usage": LLMUsage(**usage)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error in questioning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/history/clear")
async def clear_history(id: str = Query("default"), history_repo: HistoryRepository = Depends(get_history_repo)):
    await history_repo.clear_history(id)
    return {"status": "success", "message": "History cleared"}

@router.post("/evaluate-cv", response_model=CvEvaluationResponse)
async def evaluate_cv_endpoint(
    file: UploadFile = File(...),
    cv_service: CVService = Depends(get_cv_service)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported for CV evaluation.")

    try:
        content = await file.read()
        full_text = await cv_service.extract_text_from_pdf(content)
        
        evaluation, usage = await cv_service.evaluate_cv(full_text)
        return {
            "status": "success",
            **evaluation,
            "usage": LLMUsage(**usage),
        }
    except Exception as e:
        logging.error(f"Error in evaluate_cv_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
def redirect():
    return {"status": "ok"}
