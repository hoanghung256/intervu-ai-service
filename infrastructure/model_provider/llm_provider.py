import os
import json
import re
import logging
from typing import Optional, List, Dict
from google import genai
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMProvider:
    def __init__(self, model_name: str = "gemma-3-27b-it"):
        self.model_name = model_name
        
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            try:
                self.gemini_client = genai.Client(api_key=gemini_api_key).aio
            except Exception as e:
                logging.error(f"Failed to initialize Gemini client: {e}")
                self.gemini_client = None
        else:
            logging.warning("GEMINI_API_KEY not found. Gemini services will be unavailable.")
            self.gemini_client = None

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                self.hg_client = AsyncOpenAI(
                    base_url="https://router.huggingface.co/v1",
                    api_key=hf_token,
                )
            except Exception as e:
                logging.error(f"Failed to initialize HuggingFace client: {e}")
                self.hg_client = None
        else:
            logging.warning("HF_TOKEN not found. HuggingFace services will be unavailable.")
            self.hg_client = None

    async def generate_content(self, prompt: str, model: Optional[str] = None) -> str:
        if not self.gemini_client:
            return "Gemini service is currently unavailable (API key missing or invalid)."
        
        model = model or self.model_name
        response = await self.gemini_client.models.generate_content(
            model=model,
            contents=[prompt]
        )
        return response.text

    async def chat_completion(self, messages: List[Dict], model: str = "meta-llama/Llama-3.1-8B-Instruct:novita") -> Dict:
        if not self.hg_client:
            return {
                "content": "HuggingFace service is currently unavailable (API key missing or invalid).",
                "role": "assistant"
            }

        response = await self.hg_client.chat.completions.create(
            model=model,
            messages=messages
        )
        return {
            "content": response.choices[0].message.content,
            "role": response.choices[0].message.role
        }

    @staticmethod
    def clean_json_string(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        s = s.strip()

        brace_count = 0
        start_idx = None

        for idx, char in enumerate(s):
            if char in '{[':
                if brace_count == 0:
                    start_idx = idx
                brace_count += 1
            elif char in '}]':
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    return s[start_idx:idx+1]
        return s
