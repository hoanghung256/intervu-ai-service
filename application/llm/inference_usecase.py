from domain.entities.query import Query
from infrastructure.model_provider.llm_provider import LLMProvider

class InferenceUseCase:
    def __init__(self, llm_model: LLMProvider | None = None):
        self.llm_model = llm_model or LLMProvider(model_name="stub-model")

    def generate_response(self, query: Query):
        return self.llm_model.generate_answer(query.prompt)