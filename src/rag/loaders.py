"""Document loading and chunking utilities for ingestion.

Provides:
- load_documents_from_dir(directory) -> List[langchain.Document]
- split_documents_to_texts(documents, chunk_size=1000, chunk_overlap=200) -> (texts, metadatas)

Tries multiple LangChain loaders based on file extension; falls back to plain text read.
"""
from typing import List, Tuple
import os
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def _load_file_basic(path: str) -> List[Document]:
    """Fallback loader: read as text."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return [Document(page_content=content, metadata={"source": path})]
    except Exception as e:
        logger.exception("Failed to read file %s: %s", path, e)
        return []


def load_documents_from_dir(directory: str, recursive: bool = True) -> List[Document]:
    """Load documents from a directory using available loaders.
    Returns a list of langchain.Document objects with metadata.source set to filepath.
    """
    documents: List[Document] = []
    if not os.path.isdir(directory):
        logger.warning("Directory not found: %s", directory)
        return documents

    for root, dirs, files in os.walk(directory):
        for fname in files:
            fpath = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            docs_for_file = []

            # Try specialized loaders when available
            try:
                if ext in [".pdf"]:
                    try:
                        # prefer PyPDFLoader if available
                        from langchain_community.document_loaders import PyPDFLoader
                        docs_for_file = PyPDFLoader(fpath).load()
                    except Exception:
                        from langchain_community.document_loaders import UnstructuredPDFLoader
                        docs_for_file = UnstructuredPDFLoader(fpath).load()
                elif ext in [".txt", ".md"]:
                    from langchain_community.document_loaders import TextLoader
                    docs_for_file = TextLoader(fpath, encoding='utf-8').load()
                elif ext in [".docx", ".doc"]:
                    try:
                        from langchain_community.document_loaders import Docx2txtLoader
                        docs_for_file = Docx2txtLoader(fpath).load()
                    except Exception:
                        # if loader not available, fallback
                        docs_for_file = _load_file_basic(fpath)
                else:
                    # unknown extension -> try plain text
                    docs_for_file = _load_file_basic(fpath)
            except Exception as e:
                logger.exception("Loader failed for %s: %s", fpath, e)
                docs_for_file = _load_file_basic(fpath)

            # Ensure metadata.source exists
            for d in docs_for_file:
                if not getattr(d, 'metadata', None):
                    d.metadata = {}
                d.metadata['source'] = fpath
            documents.extend(docs_for_file)

        if not recursive:
            break

    logger.info("Loaded %d raw documents from %s", len(documents), directory)
    return documents


def split_documents_to_texts(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> Tuple[List[str], List[dict]]:
    """Split documents into text chunks and return texts + metadatas aligned lists."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts: List[str] = []
    metadatas: List[dict] = []
    for doc in documents:
        splits = splitter.split_text(doc.page_content)
        for s in splits:
            texts.append(s)
            meta = dict(doc.metadata) if getattr(doc, 'metadata', None) else {}
            # keep optional short preview
            meta.setdefault('source_preview', (s[:200] + '...') if len(s) > 200 else s)
            metadatas.append(meta)
    logger.info("Split into %d chunks", len(texts))
    return texts, metadatas
