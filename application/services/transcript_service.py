import json
import logging
from typing import List, Dict
from infrastructure.model_provider.llm_provider import LLMProvider

class TranscriptService:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def extract_questions_from_text(self, transcript_text: str) -> List[Dict]:
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
