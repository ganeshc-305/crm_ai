"""DocumentAgent: handles document ingestion and metadata extraction."""
from typing import Any, Dict, List
import os
import logging

from src.rag.ingest import create_vectorstore_from_texts

logger = logging.getLogger(__name__)

class DocumentAgent:
    def __init__(self, name: str = "document_agent", persist_dir: str = "data/vector_store"):
        self.name = name
        self.persist_dir = persist_dir

    def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Supported tasks:
        - {"type": "ingest_texts", "texts": [...], "metadatas": [...]} -> ingests texts into vector store
        """
        ttype = task.get("type")
        if ttype == "ingest_texts":
            texts = task.get("texts", [])
            metadatas = task.get("metadatas")
            if not texts:
                return {"error": "no texts provided"}
            try:
                vs = create_vectorstore_from_texts(texts, metadatas=metadatas, persist_dir=self.persist_dir)
                logger.info("Ingested %d documents", len(texts))
                return {"status": "ok", "ingested": len(texts)}
            except Exception as e:
                logger.exception("Ingestion failed: %s", e)
                return {"error": str(e)}
        else:
            return {"error": "unsupported task type", "task_type": ttype}
