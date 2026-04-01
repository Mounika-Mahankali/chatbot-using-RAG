from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import LLM
from typing import Optional, List
import streamlit as st 

from multimodal import get_response
from vector_db import get_retriever_from_session, process_and_save_pdfs

# Import Logger
from logger import log_execution


# Custom LLM
class CustomLLM(LLM):

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return get_response(prompt)

    @property
    def _identifying_params(self):
        return {}

    @property
    def _llm_type(self):
        return "custom_groq"


# Retrieval function with logging
@log_execution("retrieval")
def retrieve_documents(retriever, query):
    print("Retrieval function called")   
    return retriever.invoke(query)

@log_execution("llm_generation")
def generate_response(llm, prompt):
    return llm.invoke(prompt)


@log_execution("reranking")
def rerank_documents(query, docs, top_k=3):

    if not docs:
        return docs

    # Simple reranking logic 
    ranked_docs = sorted(
        docs,
        key=lambda x: len(x.page_content),
        reverse=True
    )

    return ranked_docs[:top_k]

@log_execution("image_llm_generation")
def generate_image_response(prompt, image):
    return get_response(prompt, image)

@log_execution("post_processing")
def post_process_response(response):
    return response.strip()


# Load RAG
def load_rag(chat_session_id=None):

    print("Loading RAG...")

    retriever = None

    # First use session retriever
    if "current_retriever" in st.session_state:
        retriever = st.session_state.current_retriever

    # fallback persistent retriever
    if retriever is None and chat_session_id:
        retriever = get_persistent_retriever(chat_session_id)

    print("Retriever:", retriever)

    llm = CustomLLM()

    # fallback general chat but still log
    def general_chat(query):
        return generate_response(llm, query)

    if retriever is None:
        print("No retriever found, using general chat")
        return general_chat

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def hybrid_rag(query):

        print("Hybrid RAG called")

        docs = retrieve_documents(retriever, query)

        docs = rerank_documents(query, docs)

        if not docs:
            response = generate_response(llm, query)
            return post_process_response(response)

        context = format_docs(docs)

        prompt = create_prompt(context, query)

        response = generate_response(llm, prompt)

        return post_process_response(response)

    return hybrid_rag


@log_execution("prompt_creation")
def create_prompt(context, query):
    return f"""
You are a helpful AI assistant.

Use the provided context if it is relevant to the user's question.
If the context is not relevant, answer using your general knowledge.

Context:
{context}

Question:
{query}

Answer:
"""

def get_retriever(chat_session_id):
    return get_retriever_from_session(chat_session_id)


def get_persistent_retriever(chat_session_id):
    return get_retriever_from_session(chat_session_id)


def save_pdfs_to_vector_db(pdf_paths, chat_session_id):
    return process_and_save_pdfs(pdf_paths, chat_session_id)
