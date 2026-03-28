import uvicorn
from api.api import create_app
from dotenv import load_dotenv
import os

load_dotenv()

app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
