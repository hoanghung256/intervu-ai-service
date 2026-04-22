import uvicorn
import logging
from api.api import create_app
from dotenv import load_dotenv
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

load_dotenv()

app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
