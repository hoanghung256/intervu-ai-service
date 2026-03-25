import json
import logging
from infrastructure.model_provider.llm_provider import LLMProvider
from api.dtos import SimilarityCheckRequest

class SimilarityService:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def check_similarity(self, request: SimilarityCheckRequest) -> dict:
        existing_questions_text = "\n".join(
            [f"- Title: {q.Title}\n  Content: {q.Content}"
             for q in request.SimilarMatchQuestionList]
        )

        prompt = f"""
        You are a technical interview question deduplication expert. Your task is to determine if a new "Target Question" is a duplicate of any question in a list of "Existing Questions".

        **Definition of Duplicate:**
        1. Semantically identical.
        2. Direct variation or sub-problem.
        3. Solves the exact same core algorithmic problem.
        4. Same data structure but DIFFERENT goal is NOT a duplicate.

        **Target Question:**
        - Title: "{request.Question.Title}"
        - Content: "{request.Question.Content}"

        **Existing Questions:**
        {existing_questions_text}

        **Output Format:**
        Provide reasoning followed by JSON.
        ```json
        {{ "is_new": false }}
        ```
        (Or `{{ "is_new": true }}` if no duplicates found)
        """

        try:
            response = await self.llm_provider.chat_completion([{"role": "user", "content": prompt}])
            content = response["content"]
            
            cleaned_json = self.llm_provider.clean_json_string(content)
            
            # If no JSON braces found, clean_json_string returns the whole string.
            # We should try to handle this or log it.
            if not (cleaned_json.startswith('{') or cleaned_json.startswith('[')):
                logging.error(f"LLM did not return JSON. Content: {content}")
                # Fallback or re-attempt? For now, return a default.
                return {"is_new": True}

            return json.loads(cleaned_json)
        except Exception as e:
            logging.error(f"Error in SimilarityService: {e}")
            return {"is_new": True}
