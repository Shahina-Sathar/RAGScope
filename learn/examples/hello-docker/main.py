import os

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def hello():
    # GREETING comes from an environment variable, just like RAGScope's API keys
    return {"message": os.getenv("GREETING", "Hello from inside a container!")}
