import fitz
import pymupdf4llm
import asyncio
import json
import logging
from infrastructure.model_provider.llm_provider import LLMProvider

class CVService:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.logger = logging.getLogger(__name__)

    async def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        return await asyncio.to_thread(self._extract_text_layout_aware, pdf_content)

    def _extract_text_layout_aware(self, pdf_content: bytes) -> str:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        md = pymupdf4llm.to_markdown(doc)
        doc.close()
        return md

    async def parse_document_to_json(self, text: str, doc_type: str = "cv") -> dict:
        if doc_type.lower() == "jd":
            prompt = f"""
You are a Job Description (JD) parsing assistant.
Extract structured information from the JD and output it as a JSON object with the following format:
{{
  "job_title": "",
  "must_have_skills": [],
  "nice_to_have_skills": [],
  "required_yoe": 0.0,
  "core_responsibilities": [],
  "benefits": [],
  "company_culture": ""
}}
Rules:
1. Extract the primary job title.
2. Separate skills strictly into 'must_have_skills' and 'nice_to_have_skills'.
3. Estimate 'required_yoe' (years of experience) as a float number (e.g. 2.5). If not mentioned, use 0.0.
4. Output JSON only. No text outside JSON.

JD text:
\"\"\"{text}\"\"\"
"""
        else:
            prompt = f"""
You are a CV parsing assistant.
Extract structured information from the CV and output it as a JSON object with the following format:
{{
  "apply_for": {{"job_title": ""}},
  "skills": [],
  "languages": [],
  "experiences": [
    {{
      "company": "",
      "title": "",
      "dates": "",
      "description": ""
    }}
  ],
  "certifications": [],
  "total_years_of_experience": 0.0
}}
Rules:
1. Only include information explicitly mentioned in the CV.
2. Preserve anonymized placeholders as-is.
3. Organize experiences in chronological order, most recent first.
4. Estimate 'total_years_of_experience' across all relevant roles as a float number (e.g. 3.5).
5. Output JSON only. No text outside JSON.

CV text:
\"\"\"{text}\"\"\"
"""
        response_text = await self.llm_provider.generate_content(prompt, model="gemma-3-27b-it")
        cleaned_json = self.llm_provider.clean_json_string(response_text)
        try:
            return json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON. Raw output: '{response_text}'. Cleaned output: '{cleaned_json}'. Parse error: {e}")

    # Keep backward compatibility if needed, though we will update the endpoint
    async def parse_cv_to_json(self, cv_text: str) -> dict:
        return await self.parse_document_to_json(cv_text, "cv")

    async def evaluate_cv(self, full_text: str) -> dict:
        prompt = f"""
You are an expert Career Coach and Technical Mentor.
The following text contains a Job Description (JD) and a Candidate CV.
Identify both parts, then conduct a qualitative evaluation and provide constructive feedback DIRECTLY to the candidate.

=== DOCUMENT CONTENT ===
{full_text}

=== EVALUATION GUIDELINES ===
1. summary: A concise (2-3 sentences) overview of YOUR professional profile relative to this role. You MUST use second-person pronouns (You, Your, Yours). DO NOT refer to yourself in the third person or by name.
2. strengths: A list of YOUR key strengths and areas where you strongly match the job requirements.
3. gaps: A list of areas where you could improve or skills you might need to acquire to be more competitive for this role. Use a supportive, coaching tone.
4. reasoning: A detailed explanation of how this assessment was reached. Explain your match based on:
   - Your technical skill alignment
   - The depth and relevance of your past experiences
   - Your seniority level match
   - Your potential for growth or transferable skills
5. final_verdict: A qualitative recommendation for YOUR fit. Choose one of:
   - "Excellent Fit": You match or exceed almost all requirements.
   - "Strong Fit": You match most core requirements with minor gaps.
   - "Potential Fit": You have a solid foundation but may need specific training or more direct experience.
   - "Not a Match": There are significant gaps between your current profile and this specific role's core requirements.

=== CRITICAL RULES ===
- Use ONLY "You", "Your", and "Yours" when referring to the candidate. 
- NEVER use third-person names or "The candidate".
- Output ONLY a JSON object.
- DO NOT include any literal newlines or control characters inside the JSON strings. Use \\n if a newline is needed.
- Double-check that the JSON is valid before responding.

Output Structure:
{{
  "summary": "",
  "strengths": [],
  "gaps": [],
  "reasoning": "",
  "final_verdict": ""
}}
"""
        response_text = await self.llm_provider.chat_completion(
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response_text.get("content")
        if not content:
            self.logger.error("LLM returned an empty response for CV evaluation.")
            raise ValueError("The AI model failed to generate an evaluation. Please try again or with a different model.")

        self.logger.info(f"CV Evaluation LLM Response: {content}")
        
        cleaned_json = self.llm_provider.clean_json_string(content)
        if not cleaned_json:
            self.logger.error(f"Failed to find JSON in LLM response: {content}")
            raise ValueError("The AI model returned text that did not contain a valid JSON evaluation.")

        try:
            return json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse CV evaluation JSON: {e}. Cleaned output: {cleaned_json}")
            raise ValueError(f"Failed to parse the AI model's evaluation. Error: {str(e)}")
