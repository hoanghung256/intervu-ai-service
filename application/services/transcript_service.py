import json
import logging
from typing import List, Dict
from infrastructure.model_provider.llm_provider import LLMProvider

class TranscriptService:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def extract_questions_from_text(self, transcript_text: str) -> List[Dict]:
        prompt = f"""
        Extract from the transcript text below all questions related to IT jobs (Software Engineering, Frontend Dev, Backend Dev, Fullstack Dev, Business Analyst, Tester, etc.).
        
        STRICT RULES:
        1. ONLY extract questions that are actually present or implied in the provided Transcript.
        2. If there are NO questions related to IT jobs in the Transcript, return an empty array: [].
        3. DO NOT make up or hallucinate any questions that are not in the text.
        4. Refine the extracted question so it is grammatically correct.
        5. Set the short name for the question as 'title' and the full description as 'content'.
        6. For the 'content', provide ONLY the question itself. Do not include answers or explanations.

        Transcript:
        {transcript_text}
        
        Output JSON only as an array of objects.
        Example if questions found: [{{ "title": "Question Title", "content": "The refined question text?" }}]
        Example if no questions found: []
        """
        try:
            llm_response = await self.llm_provider.chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
            cleaned_content = self.llm_provider.clean_json_string(llm_response["content"])
            if cleaned_content:
                questions_list = json.loads(cleaned_content)
                return questions_list
            return []
        except Exception as e:
            logging.error(f"Error extracting questions from transcript: {e}")
            return []
