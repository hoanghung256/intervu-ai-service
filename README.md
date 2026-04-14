# AI service serve main LLM + MCP

# Actice venv

- For the first time create venv
  python -m venv venv
- Run to active
  venv\Scripts\Activate.ps1

### Install dependencies (for the first time clone project)

pip install -r requirements.txt

### Save dependencies for sync development environment

pip freeze > requirements.txt

# Run app

python main.py
or below command for auto hotload
uvicorn api.api:create_app --host 0.0.0.0 --port 8000 --reload
