from fastapi import FastAPI
from pydantic import BaseModel
from rag import load_rag

app = FastAPI()

rag = load_rag()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    response = rag(request.message)
    return {
        "response": response
    }
