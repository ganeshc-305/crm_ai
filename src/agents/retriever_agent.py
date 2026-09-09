"""RetrieverAgent: wraps RAGRetriever to answer queries."""
from typing import Any, Dict, List
import logging

from src.rag.retrieve import RAGRetriever

logger = logging.getLogger(__name__)

class RetrieverAgent:
    def __init__(self, name: str = "retriever_agent", persist_dir: str = "data/vector_store"):
        self.name = name
        self.retriever = RAGRetriever(persist_dir=persist_dir)

    def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Supported tasks:
        - {"type": "query", "query": "text", "k": 3}
        """
        ttype = task.get("type")
        if ttype == "query":
            q = task.get("query", "")
            k = task.get("k", 4)
            if not q:
                return {"error": "no query provided"}
            try:
                docs = self.retriever.query(q, k=k)
                results = [ {"content": d.page_content, "metadata": d.metadata} for d in docs ]
                return {"status": "ok", "results": results}
            except Exception as e:
                logger.exception("Query failed: %s", e)
                return {"error": str(e)}
        else:
            return {"error": "unsupported task type", "task_type": ttype}
