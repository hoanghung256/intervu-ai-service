from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse
import spacy
from spacy.matcher import Matcher
from rapidfuzz import process, fuzz
import pdfplumber
import io
import re
from text_anonymizer import anonymize
from dotenv import load_dotenv
import os
import requests
import httpx
import json
from typing import List

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

app = FastAPI(
    title="FastAPI",
    description="FastAPI",
    version="1.0.0",
)

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

MODELS = [
        "tngtech/deepseek-r1t-chimera:free",
        "openai/gpt-oss-20b:free",
        "mistralai/mistral-7b-instruct:free",
        "qwen/qwen3-235b-a22b:free",
        "mistralai/mistral-small-3.2-24b-instruct",
        "tngtech/deepseek-r1t2-chimera:free",
        "qwen/qwen3-4b:free",
        "x-ai/grok-4.1-fast:free",
    ]

# Load spaCy model
# Define the path to the local spaCy model
SPACY_MODEL_PATH = os.path.join(os.getcwd(), "spacy_model")

# Load the spaCy model from the local directory
nlp = spacy.load(SPACY_MODEL_PATH)

# Known skills for fuzzy matching
KNOWN_SKILLS = [
    "Python", "JavaScript", "Java", "C++", "C#", "Ruby", "Go", "Swift", "Kotlin",
    "React", "Angular", "Vuejs", "Nodejs", "Django", "Flask", "Spring",
    "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL", "Git", "Docker", "Kubernetes",
    "AWS", "Azure", "Google Cloud", "Terraform", "Jenkins", "CI/CD"
]

WHITELIST = {
    # Original entries
    "FPT University", "Coursera", "English", "Japanese", "Web",
    "API", "OTP", "Google", "Servlets", "JSP", "Time",

    # Extra entries
    "TensorFlow", "PyTorch", "Keras", "GCP", "Linux", "GitHub",
    "Bitbucket", "Slack", "Zoom", "JIRA", "Confluence", "Redis"
}
WHITELIST.update(KNOWN_SKILLS)

async def chat_with_fallback(
    message: str,
    models: List[str],
    max_retries: int = 1,
    timeout: int = 10
) -> str:
    """
    Try multiple models in order until one responds successfully.
    Returns the assistant's message or raises an exception if all fail.
    """
    last_exception = None

    for model in models:
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers=HEADERS,
                        data=json.dumps({
                            "model": model,
                            "messages": [{"role": "user", "content": message}]
                        }),
                        timeout=timeout
                    )
                response.raise_for_status()
                data = response.json()

                # Extract content (adapt based on API response structure)
                # Some APIs return `choices` array
                # print(data)
                return data["choices"][0]["message"]["content"]

            except requests.exceptions.RequestException as e:
                last_exception = e
                print(f"[Warning] Model {model} attempt {attempt+1} failed: {e}")
                # Try again up to max_retries

        print(f"[Info] Switching to next model after {max_retries} failed attempts: {model}")

    raise RuntimeError(f"All models failed. Last exception: {last_exception}")

def clean_json_string(s: str) -> str:
    # Extract first {...} block
    match = re.search(r"\{.*\}", s, re.DOTALL)
    return match.group(0) if match else s

async def parse_cv_to_json(cv_text: str) -> dict:
    """
    Send anonymized CV text to the LLM and return structured JSON.
    """

    # Construct the prompt
    prompt = f"""
You are a CV parsing assistant.

You will receive the text of a CV. Some personal information like names, emails, phone numbers, and addresses has already been anonymized. 

Your task is to extract structured information from the CV and output it as a JSON object with the following format:

{{
  "summary": "<short professional summary or objective>",
  "skills": ["skill1", "skill2", "..."],
  "languages": ["language1", "language2", "..."],
  "experiences": [
    {{
      "job_title": "<Job title>",
      "company": "<Company name>",
      "description": "<Brief summary of responsibilities and achievements>"
    }}
  ],
  "certifications": ["certification1", "certification2", "..."]
}}

Rules:
1. Only include information explicitly mentioned in the CV. Do not make assumptions.
2. Preserve anonymized placeholders as-is.
3. Organize experiences and education in chronological order, most recent first.
4. If a field is not available, leave it as an empty string or empty array/object.
5. Make JSON valid (use double quotes and proper syntax).
6. **Output JSON only. Do not include any text outside the JSON object.**
7. Do not add explanations, notes, or commentary.

Here is the CV text:

\"\"\"{cv_text}\"\"\"
"""

    # Send request to OpenRouter API
    answer = await chat_with_fallback(prompt, MODELS)
    answer = clean_json_string(answer)
    try:
        structured_json = json.loads(answer)
    except json.JSONDecodeError:
        structured_json = {}  # fallback

    return structured_json

async def generate_interview_questions(cv_text: str):
    """
    A background task to generate interview questions based on the CV.
    """
    prompt = f"""
You are an interview question generator.

Based on the following CV text, generate 5–10 relevant interview questions tailored to the candidate's skills, experience, and background.

Return the result as a **JSON array of strings only**, with no additional text, explanation, or wrapping.

Example format:
{{
  "questions" : [
  "Question 1?",
  "Question 2?",
  "Question 3?"
  ],
}}

CV Text:
\"\"\"{cv_text}\"\"\"

1. Make JSON valid (use double quotes and proper syntax).
2. **Output JSON only. Do not include any text outside the JSON object.**
3. Do not add explanations, notes, or commentary.
"""
    try:
        response = await chat_with_fallback(prompt, MODELS)
        response = clean_json_string(response)
        # For now, we just print the questions. In a real app, we might save them to a database.
        print(f"[BackgroundTask] Generated Interview Questions:\n{response}")
    except Exception as e:
        print(f"[BackgroundTask] Error generating interview questions: {e}")

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extracts text from a PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # some pages may be empty
                text += page_text + "\n"
    return text

def anonymize_with_whitelist(text: str, whitelist: set[str]):
    anonymized_text, anonymization_map = anonymize(text)

    for placeholder, original in list(anonymization_map.items()):
        if original in whitelist:
            anonymized_text = anonymized_text.replace(placeholder, original)
            del anonymization_map[placeholder]

    return anonymized_text, anonymization_map

def extract_and_normalize_skills(text: str, known_skills: list[str], threshold: int = 85) -> list[str]:
    """
    Extracts entities using spaCy and normalizes them against a list of known skills
    using fuzzy matching.
    """
    doc = nlp(text)
    
    # 1. Extract potential skills using spaCy's NER (entities)
    potential_skills = {ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "LANGUAGE"]}
    
    # 2. Add any text that matches known skills exactly (case-insensitive)
    # This helps catch skills spaCy might have missed.
    text_lower = text.lower()
    for skill in known_skills:
        if skill.lower() in text_lower:
            potential_skills.add(skill)

    # 3. Normalize using fuzzy matching
    normalized_skills = set()
    for skill in potential_skills:
        match = process.extractOne(skill, known_skills, scorer=fuzz.token_set_ratio)
        if match and match[1] >= threshold:
            normalized_skills.add(match[0])
            
    return sorted(list(normalized_skills))

@app.post("/api/extract-cv")
async def extract_cv(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    content = await file.read()

    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(content)
    else:
        text = content.decode("utf-8")

    anonymized_text, anonymization_map = anonymize_with_whitelist(text, WHITELIST)
    sanitized = anonymized_text
    try:
        cv_json = await parse_cv_to_json(sanitized)
    except Exception as e:
        print(f"[Error] Failed to parse CV to JSON: {e}")
        cv_json = {"error": f"Failed to parse CV to JSON"}

    # Use spaCy and RapidFuzz to extract and normalize skills as per the document
    # This overrides or supplements the skills extracted by the LLM
    skills_from_nlp = extract_and_normalize_skills(sanitized, KNOWN_SKILLS)
    if "skills" in cv_json and isinstance(cv_json, dict):
        cv_json["skills"] = skills_from_nlp

    # Add background task to run after the response is sent
    background_tasks.add_task(generate_interview_questions, cv_json)

    return cv_json


@app.get("/api/data")
def get_sample_data():
    return {
        "data": [
            {"id": 1, "name": "Sample Item 1", "value": 100},
            {"id": 2, "name": "Sample Item 2", "value": 200},
            {"id": 3, "name": "Sample Item 3", "value": 300}
        ],
        "total": 3,
        "timestamp": "2024-01-01T00:00:00Z"
    }


@app.get("/api/items/{item_id}")
def get_item(item_id: int):
    return {
        "item": {
            "id": item_id,
            "name": "Sample Item " + str(item_id),
            "value": item_id * 100
        },
        "timestamp": "2024-01-01T00:00:00Z"
    }


@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FastAPI</title>
        <link rel="icon" type="image/x-icon" href="/favicon.ico">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
                background-color: #000000;
                color: #ffffff;
                line-height: 1.6;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }
            
            header {
                border-bottom: 1px solid #333333;
                padding: 0;
            }
            
            nav {
                max-width: 1200px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                padding: 1rem 2rem;
                gap: 2rem;
            }
            
            .logo {
                font-size: 1.25rem;
                font-weight: 600;
                color: #ffffff;
                text-decoration: none;
            }
            
            .nav-links {
                display: flex;
                gap: 1.5rem;
                margin-left: auto;
            }
            
            .nav-links a {
                text-decoration: none;
                color: #888888;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                transition: all 0.2s ease;
                font-size: 0.875rem;
                font-weight: 500;
            }
            
            .nav-links a:hover {
                color: #ffffff;
                background-color: #111111;
            }
            
            main {
                flex: 1;
                max-width: 1200px;
                margin: 0 auto;
                padding: 4rem 2rem;
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
            }
            
            .hero {
                margin-bottom: 3rem;
            }
            
            .hero-code {
                margin-top: 2rem;
                width: 100%;
                max-width: 900px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            }
            
            .hero-code pre {
                background-color: #0a0a0a;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 1.5rem;
                text-align: left;
                grid-column: 1 / -1;
            }
            
            h1 {
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 1rem;
                background: linear-gradient(to right, #ffffff, #888888);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .subtitle {
                font-size: 1.25rem;
                color: #888888;
                margin-bottom: 2rem;
                max-width: 600px;
            }
            
            .cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1.5rem;
                width: 100%;
                max-width: 900px;
            }
            
            .card {
                background-color: #111111;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 1.5rem;
                transition: all 0.2s ease;
                text-align: left;
            }
            
            .card:hover {
                border-color: #555555;
                transform: translateY(-2px);
            }
            
            .card h3 {
                font-size: 1.125rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
                color: #ffffff;
            }
            
            .card p {
                color: #888888;
                font-size: 0.875rem;
                margin-bottom: 1rem;
            }
            
            .card a {
                display: inline-flex;
                align-items: center;
                color: #ffffff;
                text-decoration: none;
                font-size: 0.875rem;
                font-weight: 500;
                padding: 0.5rem 1rem;
                background-color: #222222;
                border-radius: 6px;
                border: 1px solid #333333;
                transition: all 0.2s ease;
            }
            
            .card a:hover {
                background-color: #333333;
                border-color: #555555;
            }
            
            .status-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                background-color: #0070f3;
                color: #ffffff;
                padding: 0.25rem 0.75rem;
                border-radius: 20px;
                font-size: 0.75rem;
                font-weight: 500;
                margin-bottom: 2rem;
            }
            
            .status-dot {
                width: 6px;
                height: 6px;
                background-color: #00ff88;
                border-radius: 50%;
            }
            
            pre {
                background-color: #0a0a0a;
                border: 1px solid #333333;
                border-radius: 6px;
                padding: 1rem;
                overflow-x: auto;
                margin: 0;
            }
            
            code {
                font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
                font-size: 0.85rem;
                line-height: 1.5;
                color: #ffffff;
            }
            
            /* Syntax highlighting */
            .keyword {
                color: #ff79c6;
            }
            
            .string {
                color: #f1fa8c;
            }
            
            .function {
                color: #50fa7b;
            }
            
            .class {
                color: #8be9fd;
            }
            
            .module {
                color: #8be9fd;
            }
            
            .variable {
                color: #f8f8f2;
            }
            
            .decorator {
                color: #ffb86c;
            }
            
            @media (max-width: 768px) {
                nav {
                    padding: 1rem;
                    flex-direction: column;
                    gap: 1rem;
                }
                
                .nav-links {
                    margin-left: 0;
                }
                
                main {
                    padding: 2rem 1rem;
                }
                
                h1 {
                    font-size: 2rem;
                }
                
                .hero-code {
                    grid-template-columns: 1fr;
                }
                
                .cards {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <header>
            <nav>
                <a href="/" class="logo">Vercel + FastAPI</a>
                <div class="nav-links">
                    <a href="/docs">API Docs</a>
                    <a href="/api/data">API</a>
                </div>
            </nav>
        </header>
        <main>
            <div class="hero">
                <h1>Vercel + FastAPI</h1>
                <div class="hero-code">
                    <pre><code><span class="keyword">from</span> <span class="module">fastapi</span> <span class="keyword">import</span> <span class="class">FastAPI</span>

<span class="variable">app</span> = <span class="class">FastAPI</span>()

<span class="decorator">@app.get</span>(<span class="string">"/"</span>)
<span class="keyword">def</span> <span class="function">read_root</span>():
    <span class="keyword">return</span> {<span class="string">"Python"</span>: <span class="string">"on Vercel"</span>}</code></pre>
                </div>
            </div>
            
            <div class="cards">
                <div class="card">
                    <h3>Interactive API Docs</h3>
                    <p>Explore this API's endpoints with the interactive Swagger UI. Test requests and view response schemas in real-time.</p>
                    <a href="/docs">Open Swagger UI →</a>
                </div>
                
                <div class="card">
                    <h3>Sample Data</h3>
                    <p>Access sample JSON data through our REST API. Perfect for testing and development purposes.</p>
                    <a href="/api/data">Get Data →</a>
                </div>
                
            </div>
        </main>
    </body>
    </html>
    """