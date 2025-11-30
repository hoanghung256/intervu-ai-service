## ⭐ Running Locally (Python + virtualenv + Uvicorn)

You can run the FastAPI application locally using a Python virtual environment and Uvicorn.

### 1. Create and activate a virtual environment

```terminal
python -m venv venv
source venv/bin/activate        # macOS / Linux
# OR
.\venv\Scripts\activate           # Windows
```
### 2. Install dependencies

```terminal
pip install -r requirements.txt
```

### 3. Run the FastAPI application with Uvicorn

```terminal
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```