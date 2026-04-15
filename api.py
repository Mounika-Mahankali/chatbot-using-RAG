from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import shutil
import os
import traceback
from rag import load_rag, save_pdfs_to_db, generate_image_response
from multimodal import get_response
from db import SessionLocal, User, Chat, ChatSession

from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",   # frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load RAG only once 
rag = load_rag()

# Request Models

class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    message: str
    session_id: int


class RenameRequest(BaseModel):
    title: str


class SummaryRequest(BaseModel):
    conversation: str

class NewChatRequest(BaseModel):
    user_id: int


@app.get("/")
def home():
    return {"message": "API is running"}

@app.options("/{path:path}")
async def preflight_handler():
    return {"status": "ok"}
# Register

@app.post("/register")
def register(request: RegisterRequest):

    db = SessionLocal()

    try:

        user = User(
            username=request.username,
            password=request.password
        )

        db.add(user)
        db.commit()

        return {"message": "User registered successfully"}

    except Exception as e:
        db.rollback()
        return {"error": str(e)}

    finally:
        db.close()



# Login


@app.post("/login")
def login(request: LoginRequest):

    db = SessionLocal()

    try:

        user = db.query(User).filter_by(
            username=request.username,
            password=request.password
        ).first()

        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        return {
            "user_id": user.id,
            "message": "Login successful"
        }

    finally:
        db.close()


# New Chat

@app.post("/new-chat")
def new_chat(request: NewChatRequest):

    db = SessionLocal()

    try:

        chat = ChatSession(
            user_id=request.user_id,
            title="New Chat"
        )

        db.add(chat)
        db.commit()
        db.refresh(chat)

        return {"chat_id": chat.id}

    finally:
        db.close()

# Get Chats


@app.get("/chats/{user_id}")
def get_chats(user_id: int):

    db = SessionLocal()

    try:

        chats = db.query(ChatSession).filter_by(
            user_id=user_id
        ).all()

        return chats

    finally:
        db.close()


# Delete Chat


@app.delete("/chat/{chat_id}")
def delete_chat(chat_id: int):

    db = SessionLocal()

    try:

        db.query(Chat).filter_by(
            session_id=chat_id
        ).delete()

        db.query(ChatSession).filter_by(
            id=chat_id
        ).delete()

        db.commit()

        return {"message": "Chat deleted"}

    finally:
        db.close()



# Rename Chat

@app.put("/chat/{chat_id}")
def rename_chat(chat_id: int, request: RenameRequest):

    db = SessionLocal()

    try:

        chat = db.query(ChatSession).filter_by(
            id=chat_id
        ).first()

        chat.title = request.title

        db.commit()

        return {"message": "Chat renamed"}

    finally:
        db.close()


def generate_summary_background(session_id: int):

    db = SessionLocal()

    try:
        messages = db.query(Chat).filter_by(
            session_id=session_id
        ).all()

        if not messages:
            return

        conversation = ""

        for m in messages:
            conversation += f"User: {m.message}\n"
            conversation += f"Bot: {m.response}\n"

        summary = get_response(
            f"Summarize this conversation briefly:\n{conversation}"
        )

        session = db.query(ChatSession).filter_by(
            id=session_id
        ).first()

        if session:
            session.summary = summary
            db.commit()

    except Exception as e:
        print("Summary Error:", e)

    finally:
        db.close()

# Chat API (STABLE VERSION)


@app.post("/chat")
def chat(request: ChatRequest, background_tasks: BackgroundTasks):

    db = SessionLocal()

    try:
        print("Received message:", request.message)

        # Generate response safely
        try:
            response = rag(request.message)
        except Exception as e:
            raise Exception(f"RAG failed: {str(e)}")

        if not response:
            raise Exception("Empty response from RAG")

        # Save chat
        db.add(
            Chat(
                session_id=request.session_id,
                message=request.message,
                response=response
            )
        )

        db.commit()

        # Run summary in background
        background_tasks.add_task(
            generate_summary_background,
            request.session_id
        )

        return {"response": response}

    except Exception as e:
        db.rollback()

        print("Chat Error:")
        print(traceback.format_exc())   

        return {
            "error": "Something went wrong",
            "details": str(e)
        }

    finally:
        db.close()
# Chat History

@app.get("/chat-history/{session_id}")
def chat_history(session_id: int):

    db = SessionLocal()

    try:

        chats = db.query(Chat).filter_by(
            session_id=session_id
        ).all()

        return chats

    finally:
        db.close()

# Upload Docs

@app.post("/upload-docs")
async def upload_docs(
    session_id: int,
    file: UploadFile = File(...)
):
    try:

        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Save with session_id
        save_pdfs_to_db(
            [file_path],
            chat_session_id=session_id
        )

        return {
            "message": "Document uploaded successfully",
            "session_id": session_id
        }

    except Exception as e:
        return {"error": str(e)}


# Upload Image

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):

    try:

        image_bytes = await file.read()

        response = generate_image_response(
            "Describe this image",
            image_bytes
        )

        return {"response": response}

    except Exception as e:

        return {"error": str(e)}

# Summary API

@app.post("/summarize")
def summarize(request: SummaryRequest):

    try:

        response = get_response(
            f"Summarize this conversation:\n{request.conversation}"
        )

        return {"summary": response}

    except Exception as e:

        return {"error": str(e)}
