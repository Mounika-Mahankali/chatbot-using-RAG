import streamlit as st
import base64
from pathlib import Path
import json
from datetime import datetime

from db import SessionLocal, User, Chat, ChatSession
from multimodal import get_response #main LLM function calls groq api
from summary import summarize_chat
from langchain_memory import save_to_memory, summarize_memory

# RAG Imports
from rag import load_rag, save_pdfs_to_vector_db, generate_image_response,get_persistent_retriever
from vector_db import delete_vector_index
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import LLM
from typing import Optional, List
st.set_page_config(layout="wide")

db = SessionLocal()

# Initialize session state
if "user" not in st.session_state:
    st.session_state.user = None

if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

if "rename_id" not in st.session_state:
    st.session_state.rename_id = None

if "last_image" not in st.session_state:
    st.session_state.last_image = None

if "pdf_paths" not in st.session_state:
    st.session_state.pdf_paths = []

if "last_uploaded_pdfs" not in st.session_state:
    st.session_state.last_uploaded_pdfs = []

if "temp_image" not in st.session_state:
    st.session_state.temp_image = None

if "current_retriever" not in st.session_state:
    st.session_state.current_retriever = None


# RAG Custom LLM
# wraps groq api to langchain LLM
class CustomLLM(LLM):
    # sends req to groq ,gets response
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return get_response(prompt)

    def invoke(self, input, config=None, **kwargs):
        """Synchronous invoke method"""
        if isinstance(input, str):
            return self._call(input)
        elif hasattr(input, 'content'):
            return self._call(input.content)
        else:
            return self._call(str(input))

    async def ainvoke(self, input, config=None, **kwargs):
        """Asynchronous invoke method"""
        return self.invoke(input, config, **kwargs)

    def generate_prompt(self, prompts, stop=None, **kwargs):
        """Generate responses for multiple prompts"""
        from langchain_core.outputs import LLMResult
        from langchain_core.outputs import Generation

        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop)
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)

    async def agenerate_prompt(self, prompts, stop=None, **kwargs):
        """Async version of generate_prompt"""
        return self.generate_prompt(prompts, stop=stop, **kwargs)

    @property
    def _identifying_params(self):
        return {}

    @property
    def _llm_type(self):
        return "custom_groq"




# Full chat summarization function
def summarize_full_chat(session_id):

    chats = db.query(Chat).filter_by(session_id=session_id).all()

    if not chats:
        return "No chat available to summarize."

    conversation = ""

    for c in chats:
        conversation += f"User: {c.message}\nAssistant: {c.response}\n"

    prompt = f"""
Summarize the following conversation briefly:

{conversation}
"""

    return get_response(prompt)

# Load previous chats into memory
def load_memory_from_db(session_id):

    chats = db.query(Chat).filter_by(session_id=session_id).all()

    for chat in chats:
        save_to_memory(chat.message, chat.response)

# AUTH
def login(u, p):
    try:
        return db.query(User).filter_by(username=u, password=p).first()
    except Exception as e:
        st.error(str(e))


def register(u, p):
    try:
        db.add(User(username=u, password=p))
        db.commit()
    except Exception as e:
        st.error(str(e))


# LOGIN
if not st.session_state.user:

    st.markdown("<h2 style='text-align:center'>Login</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:

        mode = st.selectbox("Choose", ["Login", "Register"])
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if mode == "Register":

            if st.button("Register"):
                register(u, p)
                st.success("Registered!")

        else:

            if st.button("Login"):
                user = login(u, p)

                if user:
                    st.session_state.user = user.id
                    st.rerun()
                else:
                    st.error("Invalid credentials")


# MAIN
if st.session_state.user:

    st.sidebar.title("💬 Chats")

    # NEW CHAT
    if st.sidebar.button("➕ New Chat"):
        chat = ChatSession(
            user_id=st.session_state.user,
            title="New Chat",
            pdf_path=json.dumps(st.session_state.pdf_paths) if st.session_state.pdf_paths else None
        )
        db.add(chat)
        db.commit()

        st.session_state.chat_id = chat.id
        st.session_state.last_image = None

        st.rerun()

    # LOGOUT
    if st.sidebar.button("🚪 Logout"):
        st.session_state.user = None
        st.rerun()

    # CHAT LIST
    chats = db.query(ChatSession).filter_by(
        user_id=st.session_state.user
    ).all()

    for c in chats:

        col1, col2, col3 = st.sidebar.columns([4,1,1])

        if col1.button(c.title, key=f"open_{c.id}"):
            st.session_state.chat_id = c.id
            if c.pdf_path:
                try:
                    st.session_state.pdf_paths = json.loads(c.pdf_path)
                except json.JSONDecodeError:
                    # Backward compatibility: old single path
                    st.session_state.pdf_paths = [c.pdf_path]
            else:
                st.session_state.pdf_paths = []
            
            # Load persistent retriever for this session
            st.session_state.current_retriever = None
            
            st.rerun()

        if col2.button("✏️", key=f"rename_{c.id}"):
            st.session_state.rename_id = c.id

        if col3.button("🗑️", key=f"delete_{c.id}"):

            # Clean up vector database if exists
            if c.has_embeddings:
                delete_vector_index(c.id)
            
            db.query(Chat).filter_by(session_id=c.id).delete()
            db.query(ChatSession).filter_by(id=c.id).delete()
            db.commit()

            st.rerun()

        # RENAME
        if st.session_state.rename_id == c.id:

            new_name = st.sidebar.text_input(
                "New title",
                key=f"text_{c.id}"
            )

            if st.sidebar.button("Save", key=f"save_{c.id}"):

                c.title = new_name
                db.commit()

                st.session_state.rename_id = None

                st.rerun()

    # CHAT WINDOW
    if st.session_state.chat_id:

        st.title("💬 Chatbot")

        # Combined Upload Section
        st.markdown("---")
        with st.container():
            left_col, right_col = st.columns([2, 1])

            with left_col:
                st.markdown("### 📄 Document Upload")
                pdf_uploads = st.file_uploader(
                    "Upload PDFs for RAG Q&A",
                    type=["pdf","docx","txt","csv","pptx"],
                    accept_multiple_files=True,
                    help="Upload one or more PDFs to enable AI-powered questions & answers based on their content"
                )

                st.markdown("### 📋 Current Documents")
                if st.session_state.pdf_paths:
                    st.success(f"📄 **Active:** {len(st.session_state.pdf_paths)} document(s)")
                    for path in st.session_state.pdf_paths:
                        st.caption(f"• {Path(path).name}")

                if st.button("🗑️ Remove All Documents (use general chat)", key="remove_docs"):
                    st.session_state.pdf_paths = []
                    st.session_state.last_uploaded_pdfs = []
                    st.session_state.current_retriever = None

                    if st.session_state.chat_id:
                        current_chat = db.query(ChatSession).filter_by(id=st.session_state.chat_id).first()
                        if current_chat:
                            # Clean up vector database
                            if current_chat.has_embeddings:
                                delete_vector_index(current_chat.id)

                            current_chat.pdf_path = None
                            current_chat.has_embeddings = 0
                            current_chat.embeddings_updated_at = None
                            db.commit()

                    st.rerun()

                if not st.session_state.pdf_paths:
                    st.info("💬 General knowledge mode (no documents loaded)")
                    st.caption("Ask anything, not just PDF content")

            with right_col:
                st.markdown("### 🖼️ Image Upload (Multimodal)")
                uploaded_image = st.file_uploader(
                    "Upload an image to analyze",
                    type=["png", "jpg", "jpeg"],
                    key="image_upload_top",
                    help="Upload an image and ask questions about it!"
                )

                if uploaded_image:
                    image_bytes = uploaded_image.read()
                    st.session_state.temp_image = image_bytes

                if st.session_state.get('temp_image'):
                    st.image(st.session_state.temp_image, caption="Ready to analyze", width=180)
                    if st.button("🔍 Analyze Image", key="analyze_image_top", type="primary"):
                        try:
                            image_bytes = st.session_state.temp_image
                            image_base64 = base64.b64encode(image_bytes).decode()

                            st.session_state.last_image = image_bytes

                            with st.spinner("🤖 Analyzing image..."):
                                response = get_response(
                                    "Describe this image in detail and answer any questions about it.",
                                    image_bytes
                                )

                            db.add(
                                Chat(
                                    session_id=st.session_state.chat_id,
                                    message="📸 Image Analysis",
                                    response=response,
                                    image=image_base64
                                )
                            )

                            db.commit()
                            st.session_state.temp_image = None
                            st.rerun()

                        except Exception as e:
                            st.error(f"Error analyzing image: {str(e)}")

        if pdf_uploads and pdf_uploads != st.session_state.get('last_uploaded_pdfs'):
            try:
                # Save uploaded PDFs to uploads folder
                uploads_dir = Path(__file__).parent / "uploads"
                uploads_dir.mkdir(exist_ok=True)

                new_paths = []
                for pdf_upload in pdf_uploads:
                    pdf_path = uploads_dir / f"{st.session_state.user}_{pdf_upload.name}"
                    with open(pdf_path, "wb") as f:
                        f.write(pdf_upload.getbuffer())
                    new_paths.append(str(pdf_path))

                st.session_state.pdf_paths.extend(new_paths)
                st.session_state.last_uploaded_pdfs = pdf_uploads

                # Load documents for vector-less RAG
                if st.session_state.chat_id:
                    with st.spinner("Loading documents for Vector-less RAG..."):
                        current_chat = db.query(ChatSession).filter_by(id=st.session_state.chat_id).first()
                        if current_chat:

                            save_pdfs_to_vector_db(new_paths, st.session_state.chat_id)

                            current_chat.pdf_path = json.dumps(st.session_state.pdf_paths)
                            db.commit()

                st.success(f"✅ **{len(pdf_uploads)} uploaded successfully!**")
                st.info("📄 Documents loaded successfully.")

            except Exception as e:
                st.error(f"❌ Error uploading PDFs: {str(e)}")

        st.markdown("---")

        messages = db.query(Chat).filter_by(
            session_id=st.session_state.chat_id
        ).all()
        # Load memory from DB
        load_memory_from_db(st.session_state.chat_id)

        # DISPLAY CHAT
        for m in messages:

            with st.chat_message("user"):

                if m.image:
                    st.image(
                        base64.b64decode(m.image),
                        width=300
                    )
                else:
                    st.write(m.message)

            with st.chat_message("assistant"):
                st.write(m.response)

                # Show Logs Button
                if hasattr(m, "logs") and m.logs:

                    if st.button("📊 Logs", key=f"log_{m.id}"):

                        logs = json.loads(m.logs)

                        for log in logs:
                            st.write(
                                f"[RAG] {log['function']} | "
                                f"Start: {log['start']} | "
                                f"End: {log['end']} | "
                                f"Duration: {log['duration']} ms | "
                                f"Status: {log['status']}"
                            )

        
        # TEXT INPUT
        st.markdown("### 💬 Ask Questions")
        prompt = st.chat_input("Ask about the document or describe what you need...")

        if prompt:

            try:
                st.session_state.current_query_logs = []

                # Handle summarize prompt first
                if "summarize" in prompt.lower():

                    response = summarize_full_chat(st.session_state.chat_id)

                else:

                    image_data = st.session_state.last_image

                    # RAG + IMAGE LOGIC
                    if image_data:
                        image_data = st.session_state.last_image
                        st.session_state.last_image = None

                        response = generate_image_response(prompt, image_data)
                    else:
                        # Use persistent retriever if available
                        if st.session_state.current_retriever is not None:

                            # Always reload RAG for every query
                            qa = load_rag(chat_session_id=st.session_state.chat_id)

                            print("Calling RAG for query...")  

                            if hasattr(qa, "invoke"):
                                response = qa.invoke(prompt)
                            else:
                                response = qa(prompt)

                            save_to_memory(prompt, response)

                        else:

                            qa = load_rag(chat_session_id=st.session_state.chat_id)

                            if hasattr(qa, "invoke"):
                                response = qa.invoke(prompt)
                            else:
                                response = qa(prompt)

                            save_to_memory(prompt, response)


                # Save chat to DB with logs
                logs = json.dumps(st.session_state.get("current_query_logs", []))

                db.add(
                    Chat(
                        session_id=st.session_state.chat_id,
                        message=prompt,
                        response=response,
                        image=None,
                        logs=logs
                    )
                )

                db.commit()

                #  In-memory summary 
                memory_summary = summarize_memory()
                print("IN MEMORY SUMMARY:", memory_summary)

                # Store summary into DB
                session = db.query(ChatSession).filter_by(
                    id=st.session_state.chat_id
                ).first()

                if session:
                    session.summary = memory_summary
                    db.commit()

                if memory_summary:
                    st.markdown("---")
                    st.markdown("### 🧠 Conversation Summary")
                    st.info(memory_summary)

                st.rerun()

            except Exception as e:
                st.error(str(e))
