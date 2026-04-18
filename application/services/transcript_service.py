import json
import logging
from typing import List, Dict
from infrastructure.model_provider.llm_provider import LLMProvider

class TranscriptService:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def extract_questions_from_text(self, transcript_text: str, tags: List[str] = None) -> List[Dict]:
        tags_instruction = ""
        if tags and len(tags) > 0:
            tags_instruction = f"""
        7. TAGGING: Assign relevant tags to each question ONLY from the following list: {", ".join(tags)}. 
           If no tags from the list are relevant, provide an empty array [].
        """

        prompt = f"""
        You are an expert technical AI specializing in creating interview question banks. 
        Analyze the transcript and extract professional, self-contained questions and tasks.

        TASK:
        1. EXTRACT ALL TECHNICAL QUESTIONS: Identify all technical inquiries, architecture probes, or problem-solving tasks.
        2. BANK-READY TITLES: Create specific, categorical titles that reflect the exact technical concept being discussed. 
           Avoid general titles like "System Overview" or "Database". 
        3. SELF-CONTAINED CONTENT: Rewrite each question to be a professional, standalone inquiry. If the question refers to a specific system mentioned in the transcript (like a "transaction processing system"), include that context so the question makes sense in a bank.
        4. GROUP CONTEXT: Merge follow-up choices or context into the main question (e.g., "Where does duplication happen? Is it client-side or API?").
        5. FIX TRANSCRIPTION ERRORS & NORMALIZE: 
           - "duplicate charts" -> "duplicate charges"
           - "csharp" -> "C#"
           - "dot net" -> ".NET"
           - "sequel" -> "SQL"
           - "b efs" -> "BFS"
        6. IGNORE FILLER: Skip "Okay", "Thank you", "Let's move on", "jump into it", etc.
        {tags_instruction}

        OUTPUT FORMAT:
        Return a JSON array of objects:
        [
          {{
            "title": "Specific Categorical Title",
            "content": "Professional, self-contained technical question",
            "tags": ["Tag1", "Tag2"]
          }}
        ]

        TRANSCRIPT:
        {transcript_text}

        Output JSON only. If no technical questions are found, return [].
        """
        _zero_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            llm_response = await self.llm_provider.chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
            usage = llm_response.get("usage", _zero_usage)
            cleaned_content = self.llm_provider.clean_json_string(llm_response["content"])
            if cleaned_content:
                questions_list = json.loads(cleaned_content)
                return questions_list, usage
            return [], usage
        except Exception as e:
            logging.error(f"Error extracting questions from transcript: {e}")
            return [], _zero_usage
