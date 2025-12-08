from pydantic import BaseModel


class AskRequest(BaseModel):
    prompt: str


class AnswerResponse(BaseModel):
    answer: str
