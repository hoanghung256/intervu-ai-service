from domain.entities.query import Query
from infrastructure.model_provider.llm_provider import LLMProvider

class InferenceUseCase:
    def __init__(self, llm_model: LLMProvider | None = None):
        self.llm_model = llm_model or LLMProvider()

    async def generate_response(self, query: Query):
        # Using the new LLMProvider's generate_content method
        content, usage = await self.llm_model.generate_content(query.prompt)
        return content, usage
