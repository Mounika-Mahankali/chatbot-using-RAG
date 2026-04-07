from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import LLM
from typing import Optional, List
import streamlit as st 

from multimodal import get_response
from vector_db import process_and_save_pdfs

import pickle
import os

# Import Logger
from logger import log_execution

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredPowerPointLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from pathlib import Path

from rank_bm25 import BM25Okapi



# Store documents globally
vectorless_docs = []
CHUNKS_FILE = "vectorless_chunks.pkl"
# Load chunks if file exists
if os.path.exists(CHUNKS_FILE):
    with open(CHUNKS_FILE, "rb") as f:
        vectorless_docs = pickle.load(f)

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


@log_execution("llm_generation")
def generate_response(llm, prompt):
    return llm.invoke(prompt)


@log_execution("reranking")
def rerank_documents(query, docs, top_k=3):

    if not docs:
        return docs

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


# Load documents
def load_documents(file_path):

    file_path = str(file_path).lower()

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)

    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)

    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)

    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)

    elif file_path.endswith(".pptx"):
        loader = UnstructuredPowerPointLoader(file_path)

    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    return loader.load()


# Load vectorless docs
def load_vectorless_docs(pdf_paths):

    global vectorless_docs

    all_documents = []

    for file_path in pdf_paths:

        file_path = Path(file_path)

        documents = load_documents(file_path)

        if documents:
            all_documents.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    vectorless_docs = text_splitter.split_documents(all_documents)
    # Save chunks to disk
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(vectorless_docs, f)

    st.write("Chunks stored:", len(vectorless_docs))



# Load RAG
def load_rag(chat_session_id=None):

    print("Loading Vector-less RAG...")

    llm = CustomLLM()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def vectorless_rag(query):

        docs = bm25_retrieval(query)

        docs = rerank_documents(query, docs)

        if not docs:
            response = generate_response(llm, query)
            return post_process_response(response)

        context = format_docs(docs)

        prompt = create_prompt(context, query)

        response = generate_response(llm, prompt)

        return post_process_response(response)

    return vectorless_rag


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


# Save docs when uploaded
def save_pdfs_to_vector_db(pdf_paths, chat_session_id):

    load_vectorless_docs(pdf_paths)

def get_persistent_retriever(chat_session_id):
    return None


@log_execution("bm25_retrieval")
def bm25_retrieval(query):

    global vectorless_docs

    if not vectorless_docs:
        return []

    # Tokenize documents
    tokenized_docs = [
        doc.page_content.lower().split()
        for doc in vectorless_docs
    ]

    # Create BM25 object
    bm25 = BM25Okapi(tokenized_docs)

    # Tokenize query
    query_tokens = query.lower().split()

    # Get scores
    scores = bm25.get_scores(query_tokens)

    # Get top documents
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:3]

    return [vectorless_docs[i] for i in top_indices]
