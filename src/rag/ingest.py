"""Simple RAG ingestion utilities using LangChain + Chroma/FAISS.

Functions:
- create_vectorstore_from_texts(texts, metadatas=None, persist_dir='data/vector_store')

This module expects OPENAI_API_KEY (and optional OPENAI_API_BASE) in env.
"""
import os
from typing import List, Optional

from langchain_document.docstore.document import Document
from langchain_core.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import Chroma, FAISS


def create_vectorstore_from_texts(texts: List[str], metadatas: Optional[List[dict]] = None, persist_dir: str = "data/vector_store"):
    """Create a persistent vector store from a list of texts. Tries Chroma, falls back to FAISS.

    Returns the vectorstore object (Chroma or FAISS).
    """
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = OpenAIEmbeddings()
    docs = [Document(page_content=t, metadata=(metadatas[i] if metadatas and i < len(metadatas) else {}))
            for i, t in enumerate(texts)]

    # Prefer Chroma (persisted directory) if available
    try:
        vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
        # Chroma persists automatically for many versions; call persist if implemented
        try:
            vectordb.persist()
        except Exception:
            pass
        return vectordb
    except Exception:
        # Fallback to FAISS (in-memory with local save)
        vectordb = FAISS.from_documents(docs, embeddings)
        try:
            vectordb.save_local(persist_dir)
        except Exception:
            pass
        return vectordb
