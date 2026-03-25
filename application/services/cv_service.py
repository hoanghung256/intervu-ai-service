import fitz
import pymupdf4llm
import asyncio
import json
from infrastructure.model_provider.llm_provider import LLMProvider

class CVService:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        return await asyncio.to_thread(self._extract_text_layout_aware, pdf_content)

    def _extract_text_layout_aware(self, pdf_content: bytes) -> str:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        md = pymupdf4llm.to_markdown(doc)
        doc.close()
        return md

    async def parse_cv_to_json(self, cv_text: str) -> dict:
        prompt = f"""
You are a CV parsing assistant.
Extract structured information from the CV and output it as a JSON object with the following format:
{{
  "apply_for": {{"job_title": ""}},
  "skills": [],
  "languages": [],
  "experiences": [],
  "certifications": []
}}
Rules:
1. Only include information explicitly mentioned in the CV.
2. Preserve anonymized placeholders as-is.
 organizae experiences in chronological order, most recent first.
3. Make JSON valid.
4. Output JSON only. No text outside JSON.

CV text:
\"\"\"{cv_text}\"\"\"
"""
        response_text = await self.llm_provider.generate_content(prompt, model="gemma-3-27b-it")
        cleaned_json = self.llm_provider.clean_json_string(response_text)
        return json.loads(cleaned_json)
