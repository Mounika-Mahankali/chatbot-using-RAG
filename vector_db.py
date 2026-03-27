"""
Persistent Vector Database Management for RAG System
Handles creating, saving, loading, and updating FAISS indices
Supports PDF, DOCX, TXT, CSV, PPTX
"""

import os
from pathlib import Path
import shutil 

import os

import os
from pathlib import Path
import shutil
from typing import Optional, List
import warnings

warnings.filterwarnings("ignore")

# Document Loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredPowerPointLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


# Directory for storing vector indices
VECTOR_DB_DIR = Path(__file__).parent / "vector_indices"
VECTOR_DB_DIR.mkdir(exist_ok=True)

# Embeddings model configuration
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_embeddings():
    """Initialize and return embeddings model"""
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)


# NEW FUNCTION — Load Multiple File Formats
def load_documents(file_path):
    """Load documents based on file type"""

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


def get_index_path(chat_session_id: int) -> Path:
    """Get the directory path for a chat session's vector index"""
    return VECTOR_DB_DIR / f"session_{chat_session_id}"


def create_vector_index_from_pdfs(pdf_paths: List[str]) -> FAISS:
    """Create vector index from multiple file types"""

    if not pdf_paths:
        return None

    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    all_documents = []

    # Load documents from files
    for file_path in pdf_paths:

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found at {file_path}")

        documents = load_documents(file_path)

        if documents:
            all_documents.extend(documents)

    if not all_documents:
        raise ValueError("No text extracted from provided documents.")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    texts = text_splitter.split_documents(all_documents)

    # Create embeddings
    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore


def add_documents_to_index(vectorstore: FAISS, pdf_paths: List[str]) -> FAISS:

    if not pdf_paths:
        return vectorstore

    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    all_documents = []

    for file_path in pdf_paths:

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found at {file_path}")

        documents = load_documents(file_path)

        if documents:
            all_documents.extend(documents)

    if not all_documents:
        return vectorstore

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    texts = text_splitter.split_documents(all_documents)

    vectorstore.add_documents(texts)

    return vectorstore


def save_vector_index(vectorstore: FAISS, chat_session_id: int) -> Path:

    index_path = get_index_path(chat_session_id)

    if index_path.exists():
        shutil.rmtree(index_path)

    index_path.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(index_path))

    return index_path


def load_vector_index(chat_session_id: int) -> Optional[FAISS]:

    index_path = get_index_path(chat_session_id)

    if not index_path.exists():
        return None

    embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def delete_vector_index(chat_session_id: int) -> bool:

    index_path = get_index_path(chat_session_id)

    if index_path.exists():
        shutil.rmtree(index_path)
        return True

    return False


def get_retriever_from_session(chat_session_id: int):

    vectorstore = load_vector_index(chat_session_id)

    if vectorstore is None:
        return None

    return vectorstore.as_retriever(search_kwargs={"k": 3})


def process_and_save_pdfs(pdf_paths: List[str], chat_session_id: int):

    existing_index = load_vector_index(chat_session_id)

    if existing_index is not None:

        updated_index = add_documents_to_index(existing_index, pdf_paths)

    else:

        updated_index = create_vector_index_from_pdfs(pdf_paths)

    if updated_index is not None:

        save_vector_index(updated_index, chat_session_id)

        return get_index_path(chat_session_id)

    return None
from pathlib import Path
import shutil
from typing import Optional, List
import warnings

warnings.filterwarnings("ignore")

# Document Loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredPowerPointLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


# Directory for storing vector indices
VECTOR_DB_DIR = Path(__file__).parent / "vector_indices"
VECTOR_DB_DIR.mkdir(exist_ok=True)

# Embeddings model configuration
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_embeddings():
    """Initialize and return embeddings model"""
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)


# NEW FUNCTION — Load Multiple File Formats
def load_documents(file_path):
    """Load documents based on file type"""

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


def get_index_path(chat_session_id: int) -> Path:
    """Get the directory path for a chat session's vector index"""
    return VECTOR_DB_DIR / f"session_{chat_session_id}"


def create_vector_index_from_pdfs(pdf_paths: List[str]) -> FAISS:
    """Create vector index from multiple file types"""

    if not pdf_paths:
        return None

    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    all_documents = []

    # Load documents from files
    for file_path in pdf_paths:

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found at {file_path}")

        documents = load_documents(file_path)

        if documents:
            all_documents.extend(documents)

    if not all_documents:
        raise ValueError("No text extracted from provided documents.")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    texts = text_splitter.split_documents(all_documents)

    # Create embeddings
    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore


def add_documents_to_index(vectorstore: FAISS, pdf_paths: List[str]) -> FAISS:

    if not pdf_paths:
        return vectorstore

    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    all_documents = []

    for file_path in pdf_paths:

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found at {file_path}")

        documents = load_documents(file_path)

        if documents:
            all_documents.extend(documents)

    if not all_documents:
        return vectorstore

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    texts = text_splitter.split_documents(all_documents)

    vectorstore.add_documents(texts)

    return vectorstore


def save_vector_index(vectorstore: FAISS, chat_session_id: int) -> Path:

    index_path = get_index_path(chat_session_id)

    if index_path.exists():
        shutil.rmtree(index_path)

    index_path.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(index_path))

    return index_path


def load_vector_index(chat_session_id: int) -> Optional[FAISS]:

    index_path = get_index_path(chat_session_id)

    if not index_path.exists():
        return None

    embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def delete_vector_index(chat_session_id: int) -> bool:

    index_path = get_index_path(chat_session_id)

    if index_path.exists():
        shutil.rmtree(index_path)
        return True

    return False


def get_retriever_from_session(chat_session_id: int):

    vectorstore = load_vector_index(chat_session_id)

    if vectorstore is None:
        return None

    return vectorstore.as_retriever(search_kwargs={"k": 3})


def process_and_save_pdfs(pdf_paths: List[str], chat_session_id: int):

    existing_index = load_vector_index(chat_session_id)

    if existing_index is not None:

        updated_index = add_documents_to_index(existing_index, pdf_paths)

    else:

        updated_index = create_vector_index_from_pdfs(pdf_paths)

    if updated_index is not None:

        save_vector_index(updated_index, chat_session_id)

        return get_index_path(chat_session_id)

    return None
from typing import Optional, List
import warnings

warnings.filterwarnings("ignore")

# Document Loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredPowerPointLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


# Directory for storing vector indices
VECTOR_DB_DIR = Path(__file__).parent / "vector_indices"
VECTOR_DB_DIR.mkdir(exist_ok=True)

# Embeddings model configuration
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_embeddings():
    """Initialize and return embeddings model"""
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)


# NEW FUNCTION — Load Multiple File Formats
def load_documents(file_path):
    """Load documents based on file type"""

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


def get_index_path(chat_session_id: int) -> Path:
    """Get the directory path for a chat session's vector index"""
    return VECTOR_DB_DIR / f"session_{chat_session_id}"


def create_vector_index_from_pdfs(pdf_paths: List[str]) -> FAISS:
    """Create vector index from multiple file types"""

    if not pdf_paths:
        return None

    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    all_documents = []

    # Load documents from files
    for file_path in pdf_paths:

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found at {file_path}")

        documents = load_documents(file_path)

        if documents:
            all_documents.extend(documents)

    if not all_documents:
        raise ValueError("No text extracted from provided documents.")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    texts = text_splitter.split_documents(all_documents)

    # Create embeddings
    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore


def add_documents_to_index(vectorstore: FAISS, pdf_paths: List[str]) -> FAISS:

    if not pdf_paths:
        return vectorstore

    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    all_documents = []

    for file_path in pdf_paths:

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found at {file_path}")

        documents = load_documents(file_path)

        if documents:
            all_documents.extend(documents)

    if not all_documents:
        return vectorstore

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    texts = text_splitter.split_documents(all_documents)

    vectorstore.add_documents(texts)

    return vectorstore


def save_vector_index(vectorstore: FAISS, chat_session_id: int) -> Path:

    index_path = get_index_path(chat_session_id)

    if index_path.exists():
        shutil.rmtree(index_path)

    index_path.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(index_path))

    return index_path


def load_vector_index(chat_session_id: int) -> Optional[FAISS]:

    index_path = get_index_path(chat_session_id)

    if not index_path.exists():
        return None

    embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def delete_vector_index(chat_session_id: int) -> bool:

    index_path = get_index_path(chat_session_id)

    if index_path.exists():
        shutil.rmtree(index_path)
        return True

    return False


def get_retriever_from_session(chat_session_id: int):

    vectorstore = load_vector_index(chat_session_id)

    if vectorstore is None:
        return None

    return vectorstore.as_retriever(search_kwargs={"k": 3})


def process_and_save_pdfs(pdf_paths: List[str], chat_session_id: int):

    existing_index = load_vector_index(chat_session_id)

    if existing_index is not None:

        updated_index = add_documents_to_index(existing_index, pdf_paths)

    else:

        updated_index = create_vector_index_from_pdfs(pdf_paths)

    if updated_index is not None:

        save_vector_index(updated_index, chat_session_id)

        return get_index_path(chat_session_id)

    return None