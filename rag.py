from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import LLM
from typing import Optional, List

from multimodal import get_response
from vector_db import get_retriever_from_session, process_and_save_pdfs


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


# Load RAG
def load_rag(chat_session_id: int = None, pdf_paths=None):

    retriever = None
    
    if chat_session_id:
        retriever = get_persistent_retriever(chat_session_id)

    if retriever is None and pdf_paths:
        retriever = get_retriever(pdf_paths)

    llm = CustomLLM()

    # No docs, general chat
    if retriever is None:
        return lambda query: llm.invoke(query)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def hybrid_rag(query):

        docs = retriever.invoke(query)

        # If no docs,normal chat
        if not docs:
            return llm.invoke(query)

        # Optional: check relevance length
        context = format_docs(docs)

        # If context too small,fallback
        if len(context.strip()) < 20:
            return llm.invoke(query)

        prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}
"""

        return llm.invoke(prompt)

    return hybrid_rag
# Keep old function names for compatibility

def get_retriever(chat_session_id):
    return get_retriever_from_session(chat_session_id)


def get_persistent_retriever(chat_session_id):
    return get_retriever_from_session(chat_session_id)


def save_pdfs_to_vector_db(pdf_paths, chat_session_id):
    return process_and_save_pdfs(pdf_paths, chat_session_id)
