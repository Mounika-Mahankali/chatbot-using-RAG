import streamlit as st
import base64
from pathlib import Path
import json

from db import SessionLocal, User, Chat, ChatSession
from multimodal import get_response
from langchain_memory import save_to_memory, summarize_memory

# RAG Imports
from rag import load_rag, save_pdfs_to_db, generate_image_response

st.set_page_config(layout="wide")

db = SessionLocal()

# Session State

if "user" not in st.session_state:
    st.session_state.user = None

if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

if "last_image" not in st.session_state:
    st.session_state.last_image = None

if "pdf_paths" not in st.session_state:
    st.session_state.pdf_paths = []



# AUTH

def login(u, p):
    return db.query(User).filter_by(username=u, password=p).first()


def register(u, p):
    existing = db.query(User).filter_by(username=u).first()
    if existing:
        st.error("User already exists")
        return
    db.add(User(username=u, password=p))
    db.commit()


# LOGIN UI

if not st.session_state.user:

    st.title("🔐 Login")

    mode = st.selectbox("Choose", ["Login", "Register"])
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if mode == "Register":
        if st.button("Register", key="register_btn"):
            register(u, p)
            st.success("Registered!")

    else:
        if st.button("Login", key="login_btn"):
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
    if st.sidebar.button("➕ New Chat", key="new_chat_btn"):
        chat = ChatSession(
            user_id=st.session_state.user,
            title="New Chat"
        )
        db.add(chat)
        db.commit()
        st.session_state.chat_id = chat.id
        st.rerun()

    # LOGOUT
    if st.sidebar.button("🚪 Logout", key="logout_btn"):
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
            st.rerun()

        if col2.button("✏️", key=f"rename_{c.id}"):
            st.session_state.rename_id = c.id

        if col3.button("🗑️", key=f"delete_{c.id}"):

            db.query(Chat).filter_by(session_id=c.id).delete()
            db.query(ChatSession).filter_by(id=c.id).delete()
            db.commit()
            st.rerun()

    
    # CHAT WINDOW
    if st.session_state.chat_id:

        st.title("💬 Chatbot")

        # FILE UPLOAD
    
        st.markdown("### 📄 Upload Document")

        uploaded_file = st.file_uploader(
            "Upload file",
            type=["pdf", "txt", "docx"]
        )

        if uploaded_file:

            upload_dir = Path("uploads")
            upload_dir.mkdir(exist_ok=True)

            file_path = upload_dir / uploaded_file.name

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            save_pdfs_to_db(
                [str(file_path)],
                chat_session_id=st.session_state.chat_id
            )

            st.success("Document uploaded successfully")

        # IMAGE UPLOAD
        
        st.markdown("### 🖼️ Upload Image")

        image_file = st.file_uploader(
            "Upload image",
            type=["png", "jpg", "jpeg"]
        )

        if image_file:
            image_bytes = image_file.read()
            st.session_state.last_image = image_bytes
            st.image(image_bytes, width=200)

        # DISPLAY CHAT
        
        messages = db.query(Chat).filter_by(
            session_id=st.session_state.chat_id
        ).all()

        for m in messages:

            with st.chat_message("user"):
                st.write(m.message)

            with st.chat_message("assistant"):
                st.write(m.response)

        # USER INPUT
        prompt = st.chat_input("Ask something...")

        if prompt:

            try:

                # IMAGE MODE
                if st.session_state.last_image:
                    response = generate_image_response(
                        prompt,
                        st.session_state.last_image
                    )
                    st.session_state.last_image = None

                # RAG MODE
                else:
                    qa = load_rag(chat_session_id=st.session_state.chat_id)
                    response = qa(prompt)

                # SAVE CHAT
                db.add(
                    Chat(
                        session_id=st.session_state.chat_id,
                        message=prompt,
                        response=response
                    )
                )
                db.commit()

                # MEMORY SUMMARY
                save_to_memory(prompt, response)
                summary = summarize_memory()

                if summary:
                    st.info(f"🧠 Summary: {summary}")

                st.rerun()

            except Exception as e:
                st.error(str(e))
