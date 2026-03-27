import os
from pathlib import Path
import warnings

# Reduce HF/transformers startup noise for Streamlit
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("WANDB_DISABLED", "true")

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

warnings.filterwarnings("ignore", message=r".*UNEXPECTED.*")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Import persistent vector database functions
from vector_db import (
    get_retriever_from_session,
    process_and_save_pdfs,
    create_vector_index_from_pdfs,
    save_vector_index,
    get_embeddings
)


def get_retriever(pdf_paths=None):
    """
    Create a retriever from one or more PDF files.
    If no pdf_paths provided, returns None so general chat works.
    pdf_paths can be a single string or a list of strings.
    
    This function creates in-memory indices (legacy behavior).
    For persistent storage, use get_retriever_from_session() instead.
    """
    if not pdf_paths:
        return None

    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    all_documents = []
    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {pdf_path}.")
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        if documents:
            all_documents.extend(documents)

    if not all_documents:
        raise ValueError("No text extracted from the provided PDFs.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    texts = text_splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore.as_retriever()


def get_persistent_retriever(chat_session_id: int):
    """
    Get a retriever from persistent storage for a chat session.
    Loads pre-computed embeddings from disk.
    
    Args:
        chat_session_id: The chat session ID to load embeddings for
        
    Returns:
        Retriever object or None if no embeddings exist
    """
    return get_retriever_from_session(chat_session_id)


def save_pdfs_to_vector_db(pdf_paths: list, chat_session_id: int):
    """
    Process PDFs and save their embeddings to persistent storage.
    
    Args:
        pdf_paths: List of PDF file paths
        chat_session_id: Chat session ID to associate with
        
    Returns:
        Path to saved vector index
    """
    return process_and_save_pdfs(pdf_paths, chat_session_id)