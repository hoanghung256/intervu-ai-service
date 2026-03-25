from fastapi import APIRouter, Depends
from api.dtos import AnswerResponse, AskRequest
from domain.entities.query import Query
from application.llm.inference_usecase import InferenceUseCase

router = APIRouter()

def get_inference_usecase() -> InferenceUseCase:
    return InferenceUseCase()

@router.post("/inference", response_model=AnswerResponse)
def ask(payload: AskRequest, inference_usecase: InferenceUseCase = Depends(get_inference_usecase)) -> AnswerResponse:
    query = Query(prompt=payload.prompt)
    result = inference_usecase.generate_response(query)
    return AnswerResponse(answer=result)

@router.get("/")
def redirect():
    return {"status": "ok"}