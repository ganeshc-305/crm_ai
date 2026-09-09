"""RAG Retriever wrapper that loads a persisted Chroma or FAISS store and performs similarity search."""
import os
from typing import List

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.docstore.document import Document


class RAGRetriever:
    def __init__(self, persist_dir: str = "data/vector_store", embedding_client=None):
        self.persist_dir = persist_dir
        self.embedding_client = embedding_client or OpenAIEmbeddings()
        self.vectordb = None
        self._load_vectorstore()

    def _load_vectorstore(self):
        # Try Chroma first
        try:
            if os.path.isdir(self.persist_dir):
                self.vectordb = Chroma(persist_directory=self.persist_dir, embedding_function=self.embedding_client)
                return
        except Exception:
            self.vectordb = None

        # Try FAISS
        try:
            if os.path.isdir(self.persist_dir):
                self.vectordb = FAISS.load_local(self.persist_dir, self.embedding_client)
                return
        except Exception:
            self.vectordb = None

    def is_ready(self) -> bool:
        return self.vectordb is not None

    def query(self, text: str, k: int = 4) -> List[Document]:
        """Run similarity search and return list of langchain Documents."""
        if not self.vectordb:
            raise RuntimeError("Vector store not loaded. Run ingestion first or check persist_dir.")
        return self.vectordb.similarity_search(text, k=k)
